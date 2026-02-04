
from utils.paths import vom_raster_paths_parquet, tree_vector_paths_parquet, vom_unzipped_dir

import re
import logging
import pandas as pd
from pathlib import Path

from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType, IntegerType

def extract_vom_type(file_name: str|Path) -> str:
    """
    Classifies the type of VOM (Vegetation Object Model) based on the file name.

    Args:
        file_name (str | Path): The name of the file to classify.

    Returns:
        str: 'HS' if the file name contains 'VOM_HS_', otherwise 'CHM'.
    """

    if 'VOM_HS_' in file_name:
        return 'HS'
    else:
        return 'CHM'

@udf(StringType())
def extract_vom_type_udf(file_name: str|Path) -> str:
    """
    Spark DataFrame function wrapper to extract the type of VOM (Vegetation Object Model) based on the file name.

    Args:
        file_name (str | Path): The name of the file to classify.

    Returns:
        str: 'HS' if the file name contains 'VOM_HS_', otherwise 'CHM'.
    """
    return extract_vom_type(file_name)

def extract_grid_reference(filename: str|Path) -> str|None:
    """
    Extracts a grid reference from a given filename.
    The function searches for a pattern in the filename that matches 'VOM' or 'VOM_HS'
    followed by an underscore, a two-letter code, a four-digit number, and another underscore.
    If such a pattern is found, it returns the grid reference (the two-letter code and the four-digit number).
    If no match is found, it returns None.

    Args:
        filename (str | Path): The name of the file from which to extract the grid reference.

    Returns:
        str | None: The extracted grid reference if a match is found, otherwise None.
    """

    match = re.search(r'VOM_([A-Z]{2}\d{4})_', filename)
    if match:
        return match.group(1)
    return None

@udf(StringType())
def extract_grid_reference_udf(filename: str|Path) -> str|None:
    """
    Spark DataFrame function wrapper to extract the grid reference from a given filename.

    Args:
        filename (str | Path): The name of the file from which to extract the grid reference.

    Returns:
        str | None: The extracted grid reference if a match is found, otherwise None.
    """
    return extract_grid_reference(filename)

def extract_year(file_path: str|Path) -> int:
    """
    Extracts the year from the given file path.

    Args:
        file_path (str): The file path containing the year.

    Returns:
        int: The extracted year, or None if not found.
    """
    match = re.search(r'/(\d{4})/', file_path)
    if match:
        return int(match.group(1))
    return None

@udf(IntegerType())
def extract_year_udf(file_path: str|Path) -> int:
    """
    Spark DataFrame function wrapper to extract the year from the given file path.

    Args:
        file_path (str | Path): The file path containing the year.

    Returns:
        int: The extracted year, or None if not found.
    """
    return extract_year(file_path)

def generate_vom_paths_df(sedona: SparkSession) -> pd.DataFrame:
    """
    Generates the VOM paths DataFrame.

    Args:
        sedona (SparkSession): The Spark session.

    Returns:
        pd.DataFrame: The VOM paths DataFrame.
    """

    logging.debug("Generating VOM paths DataFrame")

    vom_sdf = sedona.read.format("binaryFile").load(f"{str(vom_unzipped_dir)}/*/*.tif")
    vom_sdf.createOrReplaceTempView("vom")

    vom_raster_paths_sdf = sedona.sql("""SELECT path FROM vom""")
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("file_type", extract_vom_type_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("TILE_NAME", extract_grid_reference_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("year", extract_year_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.filter(vom_raster_paths_sdf["file_type"] == "CHM")
    vom_raster_paths_df = vom_raster_paths_sdf.toPandas()
    vom_raster_paths_df['path'] = vom_raster_paths_df['path'].str.replace('file:', '', regex=False)
    vom_raster_paths_df.sort_values(by=['TILE_NAME', 'year'], ascending=[True, False], inplace=True)
    vom_raster_paths_df.reset_index(drop=True, inplace=True)

    vom_raster_paths_df.to_parquet(vom_raster_paths_parquet, index=False)

    return vom_raster_paths_df

def generate_tree_paths_df(trees_dir: Path) -> pd.DataFrame:
    """
    Generates the tree paths DataFrame.

    Args:
        trees_dir (Path): The directory containing the tree files.
        
    Returns:
        pd.DataFrame: The tree paths DataFrame.
    """
    logging.debug("Generating tree paths DataFrame")
    
    tree_paths = [x for x in trees_dir.glob("*.gpkg")]
    tree_metadata = []
    for path in tree_paths:
        match = re.search(r'VOM_trees_([A-Z]{2}\d{4})_(\d{4})\.gpkg', path.name)
        if match:
            tile_name, year = match.groups()
            tree_metadata.append({'TILE_NAME': tile_name, 'year': int(year), 'path': str(path)})

    tree_vector_paths_df = pd.DataFrame(tree_metadata)
    tree_vector_paths_df.sort_values(by=['TILE_NAME', 'year'], ascending=[True, False], inplace=True)
    tree_vector_paths_df.reset_index(drop=True, inplace=True)

    tree_vector_paths_df.to_parquet(tree_vector_paths_parquet, index=False)

    return tree_vector_paths_df