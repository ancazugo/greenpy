from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import save_temp_file, find_overlapping_files, get_geometries, view_suffix, drop_geo_views
from .utils.sedona_rdd import create_spatial_rdds, count_trees_rdd


def concatenate_trees_for_boundary(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    cfg: GreenPyConfig,
    geo_boundary_gdf: gpd.GeoDataFrame,
    # tile-mode optional
    output_areas_os_tile_overlay_df: pd.DataFrame | None = None,
    tree_vector_paths_df: pd.DataFrame | None = None,
) -> object:
    """Load tree files for the boundary and register the `geo_trees_<geo_code>` Spark temp view."""
    logger.debug(f"Getting trees for {geo_code}")

    trees_dir = Path(cfg.data.trees_dir)

    if trees_dir.is_file():
        suffix = trees_dir.suffix.lower()
        geo_trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
        geo_trees_gdf = geo_trees_gdf.to_crs(cfg.crs)
    elif cfg.tile_system.enabled and output_areas_os_tile_overlay_df is not None and tree_vector_paths_df is not None:
        geo_tile_lst = (
            output_areas_os_tile_overlay_df[output_areas_os_tile_overlay_df[geo_level] == geo_code]
            ["TILE_NAME_5KM_int"].str.upper().unique().tolist()
        )
        tree_paths = (
            tree_vector_paths_df[tree_vector_paths_df["TILE_NAME"].isin(geo_tile_lst)]
            .drop_duplicates(subset=["TILE_NAME"])["path"].tolist()
        )
        parts = [gpd.read_file(p) for p in tree_paths]
        parts = [g.to_crs(cfg.crs) if g.crs is not None else g.set_crs(cfg.crs) for g in parts]
        geo_trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg.crs)
    else:
        tree_paths = find_overlapping_files(geo_boundary_gdf, trees_dir)
        parts = [gpd.read_file(p) for p in tree_paths]
        parts = [g.to_crs(cfg.crs) if g.crs is not None else g.set_crs(cfg.crs) for g in parts]
        geo_trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg.crs)
    geo_trees_gdf = geo_trees_gdf[geo_trees_gdf.geometry.notna()].reset_index(drop=True)
    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf).withColumn("tree_id", monotonically_increasing_id())
    geo_trees_sdf.createOrReplaceTempView(f"geo_trees_{view_suffix(geo_code)}")

    return geo_trees_sdf


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    sub_geo_level: str,
    geo_code: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    overwrite: bool = True,
    # tile-mode optional
    output_areas_os_tile_overlay_df: pd.DataFrame | None = None,
    tree_vector_paths_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Count all trees per sub_geo_level unit within one geo_code (no size filtering).

    Writes `Tree_count_<geo_code>.csv` with columns <sub_geo_level>,
    tree_count. Returns the DataFrame, the cached CSV when overwrite is
    False, or None on error.
    """
    start_time = time.time()
    logger.info(f"Tree_count: processing {geo_code}")

    out_path = output_dir / f"Tree_count_{geo_code}.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        sub_geo_sdf = sedona.sql(
            f"""SELECT {sub_geo_level}, geometry FROM boundaries WHERE {geo_level} = '{geo_code}'"""
        )

        geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, dissolve=True)
        geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry="geometry", crs=cfg.crs)

        geo_trees_sdf = concatenate_trees_for_boundary(
            sedona, geo_level, geo_code, cfg, geo_boundary_gdf,
            output_areas_os_tile_overlay_df, tree_vector_paths_df,
        )
        sub_geo_rdd, geo_trees_rdd = create_spatial_rdds(sub_geo_sdf, geo_trees_sdf, build_on_spatial_partitioned_rdd=True)
        geo_tree_count_sdf = count_trees_rdd(sedona, sub_geo_rdd, geo_trees_rdd, sub_geo_level, using_index=True)
        geo_tree_count_df = save_temp_file(geo_tree_count_sdf, out_path)

        end_time = time.time()
        logger.info(f"Tree_count: {geo_code} — {geo_tree_count_df['tree_count'].sum()} trees in {end_time - start_time:.2f}s")
        return geo_tree_count_df

    except Exception:
        logger.exception(f"Tree_count: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
