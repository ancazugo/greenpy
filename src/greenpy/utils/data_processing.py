import logging
import shutil
import tempfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame


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
            bbox = gpd.read_file(p, rows=0).total_bounds  # fast: reads no features
            extents.append({"path": p, "minx": bbox[0], "miny": bbox[1], "maxx": bbox[2], "maxy": bbox[3]})
        except Exception as e:
            logging.warning(f"Could not read extent of {p}: {e}")

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


def find_overlapping_rasters(boundary_gdf: gpd.GeoDataFrame, files_dir: Path, pattern: str = "*.tif") -> list[Path]:
    """Returns raster paths from files_dir whose bounding box intersects boundary_gdf."""
    import rioxarray as rxr
    files = list(files_dir.glob(pattern))
    if not files:
        return []

    from shapely.geometry import box
    dissolved = boundary_gdf.dissolve().geometry.iloc[0]
    result = []
    for p in files:
        try:
            rast = rxr.open_rasterio(p)
            bounds = rast.rio.bounds()
            bbox_geom = box(*bounds)
            if bbox_geom.intersects(dissolved):
                result.append(p)
        except Exception as e:
            logging.warning(f"Could not read bounds of {p}: {e}")
    return result


def filter_buffer_geometries(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    table_name: str,
    buffer: int | None = None,
    id_col: str = "building_id",
) -> DataFrame:
    geo_sdf = sedona.sql(
        f"""
        SELECT b.* FROM {table_name} b, geo_boundary g
        WHERE ST_Intersects(b.geometry, g.geometry)
        """
    )
    geo_sdf.createOrReplaceTempView(f"geo_{table_name}")

    if buffer:
        geo_buffer_sdf = sedona.sql(
            f"""
            SELECT ST_Buffer(b.geometry, {buffer}) AS geometry, b.{id_col}
            FROM geo_{table_name} b
            """
        )
        geo_buffer_sdf.createOrReplaceTempView(f"{table_name}_buffers")
        return geo_buffer_sdf

    return geo_sdf


def get_geometries(sedona: SparkSession, geo_level: str, geo_code: str, dissolve: bool = True) -> DataFrame:
    query = "ST_Union_Aggr(geometry) AS geometry" if dissolve else "*"
    geo_boundary_sdf = sedona.sql(
        f"""
        SELECT {query}
        FROM boundaries
        WHERE {geo_level} = '{geo_code}'
        """
    )
    geo_boundary_sdf.createOrReplaceTempView("geo_boundary")
    return geo_boundary_sdf


def save_csv_as_parquet(in_directory: Path, path_pattern: str, out_path: Path) -> pd.DataFrame:
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
