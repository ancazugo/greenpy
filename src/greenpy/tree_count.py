from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import save_temp_file, get_geometries, load_trees_gdf, view_suffix, drop_geo_views
from .utils.sedona_rdd import create_spatial_rdds, count_trees_rdd


def concatenate_trees_for_boundary(
    sedona: SparkSession,
    geo_code: str,
    cfg: GreenPyConfig,
    geo_boundary_gdf: gpd.GeoDataFrame,
) -> object:
    """Load trees for the boundary and register the `geo_trees_<geo_code>` Spark temp view.

    Trees are reduced to their centroids (as in T3), so a crown straddling two
    sub-geo units is counted once, in the unit containing its centre (one of
    them when the centre lies exactly on the shared edge).
    """
    logger.debug(f"Getting trees for {geo_code}")

    geo_trees_gdf = load_trees_gdf(Path(cfg.data.trees_dir), geo_boundary_gdf, cfg)
    geo_trees_gdf = geo_trees_gdf[geo_trees_gdf.geometry.notna()].reset_index(drop=True)
    geo_trees_gdf = gpd.GeoDataFrame(geometry=geo_trees_gdf.geometry.centroid, crs=cfg.crs)
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
) -> pd.DataFrame | None:
    """Count all trees per sub_geo_level unit within one geo_code (no size filtering).

    Each tree is counted once, in the unit containing its centroid; units with
    no trees are absent from the CSV (Merge reports them as 0).

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

        geo_trees_sdf = concatenate_trees_for_boundary(sedona, geo_code, cfg, geo_boundary_gdf)
        sub_geo_rdd, geo_trees_rdd = create_spatial_rdds(sub_geo_sdf, geo_trees_sdf, build_on_spatial_partitioned_rdd=True)
        geo_tree_count_sdf = count_trees_rdd(
            sedona, sub_geo_rdd, geo_trees_rdd, sub_geo_level, using_index=True, unique_objects=True
        )
        geo_tree_count_df = save_temp_file(geo_tree_count_sdf, out_path)

        end_time = time.time()
        logger.info(f"Tree_count: {geo_code} — {geo_tree_count_df['tree_count'].sum()} trees in {end_time - start_time:.2f}s")
        return geo_tree_count_df

    except Exception:
        logger.exception(f"Tree_count: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
