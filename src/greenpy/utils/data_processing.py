import re
import shutil
import tempfile
import pandas as pd
from loguru import logger
import geopandas as gpd
from pathlib import Path
from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame


def view_suffix(geo_code: str) -> str:
    """Sanitize a geo code into a valid Spark temp-view name suffix.

    Per-geo views are suffixed with this so parallel workers processing
    different geo codes never overwrite each other's views.
    """
    return re.sub(r"\W", "_", str(geo_code))


def drop_geo_views(sedona: SparkSession, geo_code: str) -> None:
    """Drop all per-geo temp views created while processing geo_code."""
    sfx = view_suffix(geo_code)
    for name in (
        f"geo_boundary_{sfx}",
        f"geo_sub_boundaries_{sfx}",
        f"geo_buildings_{sfx}",
        f"buildings_buffers_{sfx}",
        f"geo_trees_{sfx}",
        f"raw_tiles_{sfx}",
        f"binary_tiles_{sfx}",
        f"buildings_partitioned_{sfx}",
        f"t30b_trees_{sfx}",
        f"vis_rays_{sfx}",
        f"vis_obstacles_{sfx}",
        f"vis_crowns_{sfx}",
    ):
        sedona.catalog.dropTempView(name)


def rename_tree_columns(gdf: gpd.GeoDataFrame, cfg) -> gpd.GeoDataFrame:
    """Rename user-configured tree columns to canonical tree_height/tree_area/tree_id."""
    col = cfg.columns
    mapping = {
        col.tree_height_col: "tree_height",
        col.tree_area_col: "tree_area",
        col.tree_id_col: "tree_id",
    }
    return gdf.rename(columns={k: v for k, v in mapping.items() if k in gdf.columns})


def load_trees_gdf(
    trees_dir: Path,
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg,
) -> gpd.GeoDataFrame:
    """Read tree vector files for a boundary into one GeoDataFrame in cfg.crs.

    Reads either a single tree file or the .gpkg/.parquet files in trees_dir
    (e.g. written by the Trees process) whose extent overlaps the boundary.
    Callers that look beyond the boundary (e.g. building buffers) must pass
    the boundary buffered accordingly, or trees in neighbouring tiles are
    missed. Columns are renamed to the canonical tree_height/tree_area/tree_id
    names; original geometries are kept.
    """
    if trees_dir.is_file():
        suffix = trees_dir.suffix.lower()
        trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
    else:
        paths = find_overlapping_files(geo_boundary_gdf, trees_dir, pattern="*.gpkg")
        paths += find_overlapping_files(geo_boundary_gdf, trees_dir, pattern="*.parquet")
        logger.debug(f"Found {len(paths)} tree vector files")
        if not paths:
            raise FileNotFoundError(f"No tree .gpkg/.parquet files in {trees_dir} overlap the boundary")
        parts = [gpd.read_parquet(p) if p.suffix == ".parquet" else gpd.read_file(p) for p in paths]
        parts = [g.to_crs(cfg.crs) if g.crs is not None else g.set_crs(cfg.crs) for g in parts]
        trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg.crs)

    trees_gdf = rename_tree_columns(trees_gdf, cfg)
    return trees_gdf.to_crs(cfg.crs) if trees_gdf.crs is not None else trees_gdf.set_crs(cfg.crs)


def _vector_bounds(path: Path) -> tuple:
    """File extent without reading features: GeoParquet bbox metadata, or the OGR layer extent."""
    if path.suffix == ".parquet":
        import json
        import pyarrow.parquet as pq
        geo = json.loads(pq.read_schema(path).metadata[b"geo"])
        bbox = geo["columns"][geo["primary_column"]].get("bbox")
        return tuple(bbox) if bbox else tuple(gpd.read_parquet(path).total_bounds)
    return tuple(gpd.read_file(path, rows=0).total_bounds)  # fast: reads no features


def find_overlapping_files(boundary_gdf: gpd.GeoDataFrame, files_dir: Path, pattern: str = "*.gpkg") -> list[Path]:
    """
    Returns paths from files_dir whose bounding box intersects boundary_gdf.
    Uses a spatial index over file extents — no tile naming convention required.
    """
    files = list(files_dir.glob(pattern))
    if not files:
        return []

    extents = []
    for p in files:
        try:
            bbox = _vector_bounds(p)
            extents.append({"path": p, "minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3]})
        except Exception as e:
            logger.warning(f"Could not read extent of {p}: {e}")

    if not extents:
        return []

    from shapely.geometry import box
    extent_gdf = gpd.GeoDataFrame(
        extents,
        geometry=[box(e["minx"], e["miny"], e["maxx"], e["maxy"]) for e in extents],
        crs=boundary_gdf.crs,
    )
    dissolved = boundary_gdf.dissolve()
    hits = extent_gdf[extent_gdf.intersects(dissolved.geometry.iloc[0])]
    return hits["path"].tolist()


# Rasters that sit beside CHM tiles but are not heights: Defra VOM ships a
# uint8 hillshade (VOM_HS_*) next to every height tile, and "*.tif" picks both up.
NON_CHM_PREFIXES = ("VOM_HS_",)


def is_chm_tile(path) -> bool:
    """False for known non-height rasters stored alongside CHM tiles (e.g. Defra VOM hillshades)."""
    return not Path(path).name.startswith(NON_CHM_PREFIXES)


def find_overlapping_rasters(boundary_gdf: gpd.GeoDataFrame, files_dir: Path, pattern: str = "*.tif") -> list[Path]:
    """Returns raster paths under files_dir (searched recursively) whose bounding box intersects boundary_gdf.

    Tile bounds are transformed to the boundary CRS before the test, so tiles
    stored in another CRS are still selected correctly. Known non-height
    rasters (see is_chm_tile) are skipped.
    """
    import rioxarray as rxr
    files = sorted(p for p in files_dir.rglob(pattern) if is_chm_tile(p))
    if not files:
        return []

    from shapely.geometry import box
    dissolved = boundary_gdf.dissolve().geometry.iloc[0]
    result = []
    for p in files:
        try:
            rast = rxr.open_rasterio(p)
            if rast.rio.crs is not None and boundary_gdf.crs is not None:
                bounds = rast.rio.transform_bounds(boundary_gdf.crs)
            else:
                bounds = rast.rio.bounds()
            bbox_geom = box(*bounds)
            if bbox_geom.intersects(dissolved):
                result.append(p)
        except Exception as e:
            logger.warning(f"Could not read bounds of {p}: {e}")
    return result


def filter_buffer_geometries(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    table_name: str,
    buffer: int | None = None,
    id_col: str = "building_id",
) -> DataFrame:
    """Filter table_name features belonging to the geo_code, optionally buffered.

    Buildings are selected through the `boundaries_buildings_overlay` lookup
    (representative point in the unit), so a building straddling two geo codes
    is processed — and output — by exactly one of them, the same unit Merge
    aggregates it into. Other tables are selected by intersection with the
    `geo_boundary_<geo_code>` view created by get_geometries() or
    get_sub_geo_boundaries(). Registers `geo_<table_name>_<geo_code>` and, when a
    buffer is given, `<table_name>_buffers_<geo_code>` (geometry buffered by
    `buffer` metres plus id_col).
    """
    sfx = view_suffix(geo_code)
    if table_name == "buildings":
        query = f"""
        SELECT b.* FROM buildings b
        JOIN boundaries_buildings_overlay o ON b.building_id = o.building_id
        WHERE o.{geo_level} = '{geo_code}'
        """
    else:
        query = f"""
        SELECT b.* FROM {table_name} b, geo_boundary_{sfx} g
        WHERE ST_Intersects(b.geometry, g.geometry)
        """
    geo_sdf = sedona.sql(query)
    geo_sdf.createOrReplaceTempView(f"geo_{table_name}_{sfx}")

    if buffer:
        geo_buffer_sdf = sedona.sql(
            f"""
            SELECT ST_Buffer(b.geometry, {buffer}) AS geometry, b.{id_col}
            FROM geo_{table_name}_{sfx} b
            """
        )
        geo_buffer_sdf.createOrReplaceTempView(f"{table_name}_buffers_{sfx}")
        return geo_buffer_sdf

    return geo_sdf


def get_geometries(sedona: SparkSession, geo_level: str, geo_code: str, dissolve: bool = True) -> DataFrame:
    """Select boundary rows where geo_level = geo_code, optionally dissolved to one geometry.

    Registers the result as the `geo_boundary_<geo_code>` temp view used by
    filter_buffer_geometries().
    """
    query = "ST_Union_Aggr(geometry) AS geometry" if dissolve else "*"
    geo_boundary_sdf = sedona.sql(
        f"""
        SELECT {query}
        FROM boundaries
        WHERE {geo_level} = '{geo_code}'
        """
    )
    geo_boundary_sdf.createOrReplaceTempView(f"geo_boundary_{view_suffix(geo_code)}")
    return geo_boundary_sdf


def get_sub_geo_boundaries(
    sedona: SparkSession, geo_level: str, geo_code: str, sub_geo_level: str
) -> DataFrame:
    """Return one dissolved geometry per sub_geo_level unit within geo_level = geo_code.

    Also refreshes the `geo_boundary_<geo_code>` temp view (whole-region
    dissolve) so that filter_buffer_geometries() continues to work correctly.
    """
    sfx = view_suffix(geo_code)
    sdf = sedona.sql(
        f"""
        SELECT {sub_geo_level}, ST_Union_Aggr(geometry) AS geometry
        FROM boundaries
        WHERE {geo_level} = '{geo_code}'
        GROUP BY {sub_geo_level}
        """
    )
    sdf.createOrReplaceTempView(f"geo_sub_boundaries_{sfx}")

    sedona.sql(
        f"""
        SELECT ST_Union_Aggr(geometry) AS geometry
        FROM boundaries
        WHERE {geo_level} = '{geo_code}'
        """
    ).createOrReplaceTempView(f"geo_boundary_{sfx}")

    return sdf


def save_csv_as_parquet(in_directory: Path, path_pattern: str, out_path: Path) -> pd.DataFrame:
    """Concatenate all CSVs matching path_pattern into a single parquet file."""
    csv_files = list(in_directory.glob(path_pattern))
    dataframes_lst = [pd.read_csv(file) for file in csv_files]
    concatenated_df = pd.concat(dataframes_lst, ignore_index=True)
    concatenated_df.to_parquet(out_path, index=False)
    return concatenated_df


def save_temp_file(spark_df: DataFrame, output_path: Path, coalesce: int = 1, file_format: str = "csv") -> pd.DataFrame:
    """
    Saves a Spark DataFrame to a single named file and returns a Pandas DataFrame.
    Workaround for Spark writing to a directory instead of a single file.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        spark_df.coalesce(coalesce) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .format(file_format) \
            .save(str(temp_path))

        part_files = list(temp_path.glob(f"part-*.{file_format}*"))
        if not part_files:
            raise FileNotFoundError(f"No part file found with format '{file_format}' in {temp_dir}")
        shutil.move(part_files[0], output_path)

    if file_format == "parquet":
        return pd.read_parquet(output_path)
    elif file_format == "csv":
        return pd.read_csv(output_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")
