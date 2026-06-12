from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import monotonically_increasing_id

from .config.schema import GreenPyConfig
from .utils.data_processing import (
    filter_buffer_geometries,
    get_geometries,
    find_overlapping_files,
    rename_tree_columns,
    view_suffix,
    drop_geo_views,
)
from .utils.sedona_rdd import create_spatial_rdds, count_trees_rdd


def read_trees_unique(
    sedona: SparkSession,
    trees_dir: Path,
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    geo_code: str,
    tree_area: int = 10,
    tree_height: int = 3,
    tree_paths: list[Path] | None = None,
) -> DataFrame:
    """Load trees for the geo_code boundary and register the `geo_trees_<geo_code>` view.

    Reads either a single tree file, an explicit list of tile paths, or the
    files in trees_dir overlapping the boundary. Columns are renamed to the
    canonical tree_height/tree_area names and reprojected to cfg.crs, then
    filtered (area > tree_area, height > tree_height) and reduced to centroids
    with a generated tree_id.
    """
    logger.debug(f"Reading tree vector files for {geo_code}")

    if tree_paths is not None:
        parts = [gpd.read_file(p) for p in tree_paths]
        geo_trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))
    elif trees_dir.is_file():
        suffix = trees_dir.suffix.lower()
        geo_trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
    elif cfg.tile_system.enabled:
        paths = list(trees_dir.glob("*.gpkg"))
        logger.debug(f"Found {len(paths)} tree tile files")
        parts = [gpd.read_file(p) for p in paths]
        geo_trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))
    else:
        paths = find_overlapping_files(geo_boundary_gdf, trees_dir, pattern="*.gpkg")
        logger.debug(f"Found {len(paths)} tree vector files")
        parts = [gpd.read_file(p) for p in paths]
        geo_trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True))

    geo_trees_gdf = rename_tree_columns(geo_trees_gdf, cfg)
    geo_trees_gdf = geo_trees_gdf.to_crs(cfg.crs) if geo_trees_gdf.crs is not None else geo_trees_gdf.set_crs(cfg.crs)

    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
    if "geom" in geo_trees_sdf.columns:
        geo_trees_sdf = geo_trees_sdf.withColumnRenamed("geom", "geometry")
    geo_trees_sdf = (
        geo_trees_sdf
        .where(f"tree_area > {tree_area} AND tree_height > {tree_height} AND geometry IS NOT NULL")
        .selectExpr("ST_Centroid(geometry) AS geometry", "tree_area", "tree_height")
        .withColumn("tree_id", monotonically_increasing_id())
    )
    geo_trees_sdf.createOrReplaceTempView(f"geo_trees_{view_suffix(geo_code)}")

    return geo_trees_sdf


def count_trees(sedona: SparkSession, geo_code: str) -> pd.DataFrame:
    """Count trees per building buffer using a SQL spatial join.

    LEFT JOIN semantics: buildings whose buffer contains no trees are returned
    with tree_count 0 (unlike the RDD path, which omits them).
    """
    sfx = view_suffix(geo_code)
    trees_within_buffer_sdf = sedona.sql(
        f"""
        SELECT b.building_id, COUNT(t.tree_id) AS tree_count
        FROM buildings_buffers_{sfx} b
        LEFT JOIN geo_trees_{sfx} t
        ON ST_Intersects(b.geometry, t.geometry)
        GROUP BY b.building_id
        """
    )
    return trees_within_buffer_sdf.toPandas()


def _attach_sub_geo_level(
    sedona, geo_tree_count_df: pd.DataFrame, geo_level: str, geo_code: str, sub_geo_level: str
) -> pd.DataFrame:
    """Join the sub_geo_level code of each building (by centroid containment) onto the counts."""
    sfx = view_suffix(geo_code)
    building_level_df = sedona.sql(
        f"""
        SELECT b.building_id, bnd.{sub_geo_level}
        FROM geo_buildings_{sfx} b
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
) -> pd.DataFrame | None:
    """Compute T3 (trees within `buffer` metres of each building) for one geo_code.

    Writes `T3_<geo_code>_<buffer>m.csv` to output_dir with columns
    building_id, tree_count_<buffer>m, <sub_geo_level>. Returns the DataFrame,
    the cached CSV when it exists and overwrite is False, or None on error.
    """
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

        trees_dir = Path(cfg.data.trees_dir)
        geo_boundary_gdf = gpd.GeoDataFrame(
            get_geometries(sedona, geo_level, geo_code, dissolve=True).toPandas(),
            geometry="geometry", crs=cfg.crs,
        )

        tree_paths = None
        if cfg.tile_system.enabled and overlapping_tiles_lst is not None:
            # Tile-mode: pre-filter tile files by name substrings
            tree_paths = [p for p in trees_dir.glob("*.gpkg") if any(t in p.name for t in overlapping_tiles_lst)]

        geo_trees_sdf = read_trees_unique(
            sedona, trees_dir, geo_boundary_gdf, cfg, geo_code, tree_area, tree_height, tree_paths=tree_paths
        )

        if query_method == "sql":
            logger.debug("Executing T3 query using SQL")
            geo_tree_count_df = count_trees(sedona, geo_code)
        else:
            logger.debug("Executing T3 query using Spatial RDD")
            geo_buildings_buffer_rdd, geo_trees_rdd = create_spatial_rdds(
                geo_buildings_buffer_sdf, geo_trees_sdf, build_on_spatial_partitioned_rdd=True
            )
            geo_tree_count_sdf = count_trees_rdd(sedona, geo_buildings_buffer_rdd, geo_trees_rdd, "building_id", using_index=True)
            geo_tree_count_df = geo_tree_count_sdf.toPandas()

        geo_tree_count_df.rename(columns={"tree_count": f"tree_count_{buffer}m"}, inplace=True)
        geo_tree_count_df = _attach_sub_geo_level(sedona, geo_tree_count_df, geo_level, geo_code, sub_geo_level)
        geo_tree_count_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"T3: {geo_code} — {len(geo_tree_count_df)} records in {end_time - start_time:.2f}s")
        return geo_tree_count_df

    except Exception:
        logger.exception(f"T3: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
