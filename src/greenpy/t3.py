from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from .config.schema import GreenPyConfig
from .utils.data_processing import filter_buffer_geometries, get_geometries, find_overlapping_files, save_temp_file
from .utils.sedona_rdd import create_spatial_rdds, count_trees_rdd


def process_vom_tiles(
    sedona: SparkSession, trees_path_lst: list, cfg: GreenPyConfig, tree_area: int = 10, tree_height: int = 3
) -> gpd.GeoDataFrame | None:
    """Read and filter tree vector files, register as Spark temp view."""
    logger.debug(f"Reading {len(trees_path_lst)} tree tile files")

    if not trees_path_lst:
        return None

    col = cfg.columns
    if len(trees_path_lst) > 1:
        trees_gdf_lst = [gpd.read_file(p, layer=col.tree_layer) for p in trees_path_lst]
        merged_trees_gdf = gpd.GeoDataFrame(pd.concat(trees_gdf_lst, ignore_index=True))
    else:
        merged_trees_gdf = gpd.read_file(trees_path_lst[0], layer=col.tree_layer)

    geo_trees_gdf = merged_trees_gdf[
        (merged_trees_gdf["tree_area"] > tree_area) & (merged_trees_gdf["tree_height"] > tree_height)
    ].reset_index(drop=True)
    geo_trees_gdf["tree_id"] = range(len(geo_trees_gdf))
    geo_trees_gdf["geometry"] = geo_trees_gdf["geometry"].centroid

    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
    geo_trees_sdf.createOrReplaceTempView("geo_trees")

    return geo_trees_gdf


def read_trees_unique(
    sedona: SparkSession, trees_dir: Path, geo_boundary_gdf: gpd.GeoDataFrame, cfg: GreenPyConfig,
    tree_area: int = 10, tree_height: int = 3,
) -> DataFrame:
    """Load tree files overlapping the boundary, filter by height/area, register as Spark temp view."""
    logger.debug("Reading tree vector files overlapping boundary")

    col = cfg.columns

    if trees_dir.is_file():
        suffix = trees_dir.suffix.lower()
        geo_trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
        geo_trees_gdf = geo_trees_gdf.rename(columns={
            col.tree_height_col: "tree_height",
            col.tree_area_col: "tree_area",
            col.tree_id_col: "tree_id",
        })
    elif cfg.tile_system.enabled:
        tree_paths = list(trees_dir.glob("*.gpkg"))
        logger.debug(f"Found {len(tree_paths)} tree tile files")
        geo_trees_gdf = pd.concat([gpd.read_file(p) for p in tree_paths], ignore_index=True)
    else:
        tree_paths = find_overlapping_files(geo_boundary_gdf, trees_dir, pattern="*.gpkg")
        logger.debug(f"Found {len(tree_paths)} tree vector files")
        geo_trees_gdf = pd.concat([gpd.read_file(p) for p in tree_paths], ignore_index=True)
    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
    if "geom" in geo_trees_sdf.columns:
        geo_trees_sdf = geo_trees_sdf.withColumnRenamed("geom", "geometry")
    geo_trees_sdf.createOrReplaceTempView("geo_trees")
    geo_trees_sdf = sedona.sql(
        f"""SELECT tree_id, tree_area, tree_height, ST_Centroid(geometry) AS geometry
            FROM geo_trees
            WHERE tree_area > {tree_area} AND tree_height > {tree_height} AND geometry IS NOT NULL"""
    )
    geo_trees_sdf = geo_trees_sdf.withColumn("tree_id", monotonically_increasing_id())
    geo_trees_sdf.createOrReplaceTempView("geo_trees")

    return geo_trees_sdf


def count_trees(sedona: SparkSession) -> pd.DataFrame:
    """Count trees per building using SQL join on Spark temp views."""
    trees_within_buffer_sdf = sedona.sql(
        """
        SELECT b.building_id, COUNT(t.tree_id) AS tree_count
        FROM buildings_buffers b
        LEFT JOIN geo_trees t
        ON ST_Intersects(b.geometry, t.geometry)
        GROUP BY b.building_id
        """
    )
    return trees_within_buffer_sdf.toPandas()


def _attach_sub_geo_level(
    sedona, geo_tree_count_df: pd.DataFrame, geo_level: str, geo_code: str, sub_geo_level: str
) -> pd.DataFrame:
    """Join sub_geo_level column onto per-building results using geo_buildings Spark view."""
    building_level_df = sedona.sql(
        f"""
        SELECT b.building_id, bnd.{sub_geo_level}
        FROM geo_buildings b
        JOIN boundaries bnd ON ST_Contains(bnd.geometry, ST_Centroid(b.geometry))
        WHERE bnd.{geo_level} = '{geo_code}'
        """
    ).toPandas()
    return geo_tree_count_df.merge(building_level_df, on="building_id", how="left")


def process_geo_code(
    sedona: SparkSession,
    query_method: str,
    geo_level: str,
    geo_code: str,
    sub_geo_level: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    buffer: int = 100,
    tree_area: int = 10,
    tree_height: int = 3,
    overwrite: bool = True,
    # tile-mode optional args
    overlapping_tiles_lst: list | None = None,
    output_areas_os_tile_overlay_df: pd.DataFrame | None = None,
    vom_raster_paths_df: pd.DataFrame | None = None,
    tree_vector_paths_df: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    start_time = time.time()
    logger.info(f"T3: processing {geo_code} with buffer {buffer}m")

    out_path = output_dir / f"T3_{geo_code}_{buffer}m.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        get_geometries(sedona, geo_level, geo_code, dissolve=True)
        geo_buildings_buffer_sdf = filter_buffer_geometries(
            sedona, geo_level, geo_code, "buildings", buffer, id_col="building_id"
        )

        if query_method == "sql":
            logger.debug("Executing T3 query using SQL")
            trees_dir = Path(cfg.data.trees_dir)
            geo_boundary_gdf = gpd.GeoDataFrame(
                get_geometries(sedona, geo_level, geo_code, dissolve=True).toPandas(),
                geometry="geometry", crs=cfg.crs,
            )
            read_trees_unique(sedona, trees_dir, geo_boundary_gdf, cfg, tree_area, tree_height)
            geo_tree_count_df = count_trees(sedona)
            geo_tree_count_df.rename(columns={"tree_count": f"tree_count_{buffer}m"}, inplace=True)
            geo_tree_count_df = _attach_sub_geo_level(sedona, geo_tree_count_df, geo_level, geo_code, sub_geo_level)
            geo_tree_count_df.to_csv(out_path, index=False)

        elif query_method == "rdd":
            logger.debug("Executing T3 query using Spatial RDD")
            if cfg.tile_system.enabled and overlapping_tiles_lst is not None:
                trees_dir = Path(cfg.data.trees_dir)
                # Tile-mode: filter by tile name substrings
                tree_paths = [p for p in trees_dir.glob("*.gpkg") if any(t in p.name for t in overlapping_tiles_lst)]
                geo_trees_gdf = pd.concat([gpd.read_file(p) for p in tree_paths], ignore_index=True)
                geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
                geo_trees_sdf.createOrReplaceTempView("geo_trees")
                geo_trees_sdf = sedona.sql(
                    f"""SELECT tree_id, tree_area, tree_height, ST_Centroid(geometry) AS geometry
                        FROM geo_trees WHERE tree_area > {tree_area} AND tree_height > {tree_height} AND geometry IS NOT NULL"""
                )
                geo_trees_sdf = geo_trees_sdf.withColumn("tree_id", monotonically_increasing_id())
                geo_trees_sdf.createOrReplaceTempView("geo_trees")
            else:
                trees_dir = Path(cfg.data.trees_dir)
                geo_boundary_gdf = gpd.GeoDataFrame(
                    get_geometries(sedona, geo_level, geo_code, dissolve=True).toPandas(),
                    geometry="geometry", crs=cfg.crs,
                )
                geo_trees_sdf = read_trees_unique(sedona, trees_dir, geo_boundary_gdf, cfg, tree_area, tree_height)

            geo_buildings_buffer_rdd, geo_trees_rdd = create_spatial_rdds(
                geo_buildings_buffer_sdf, geo_trees_sdf, build_on_spatial_partitioned_rdd=True
            )
            geo_tree_count_sdf = count_trees_rdd(sedona, geo_buildings_buffer_rdd, geo_trees_rdd, "building_id", using_index=True)
            geo_tree_count_df = save_temp_file(geo_tree_count_sdf, out_path)
            geo_tree_count_df.rename(columns={"tree_count": f"tree_count_{buffer}m"}, inplace=True)
            geo_tree_count_df = _attach_sub_geo_level(sedona, geo_tree_count_df, geo_level, geo_code, sub_geo_level)
            geo_tree_count_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"T3: {geo_code} — {len(geo_tree_count_df)} records in {end_time - start_time:.2f}s")
        return geo_tree_count_df

    except Exception as e:
        logger.error(f"T3: error processing {geo_code}: {e}")
