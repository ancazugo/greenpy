"""
UK-specific VOM (Vegetation Object Model) raster and vector file processing.
Requires tile_system.enabled = True in config.
"""

import re
import logging
import pandas as pd
from pathlib import Path

from pyspark.sql.functions import udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import StringType, IntegerType


def extract_vom_type(file_name):
    return "HS" if "VOM_HS_" in file_name else "CHM"


@udf(StringType())
def extract_vom_type_udf(file_name):
    return "HS" if "VOM_HS_" in file_name else "CHM"


def extract_grid_reference(filename: str, pattern: str = r"VOM_([A-Z]{2}\d{4})_") -> str | None:
    """Extract grid reference from filename using the provided regex pattern."""
    match = re.search(pattern, filename)
    return match.group(1) if match else None


def make_extract_grid_reference_udf(pattern: str = r"VOM_([A-Z]{2}\d{4})_"):
    @udf(StringType())
    def _udf(filename: str) -> str | None:
        import re
        m = re.search(pattern, filename)
        return m.group(1) if m else None
    return _udf


def extract_year(file_path: str) -> int | None:
    match = re.search(r"/(\d{4})/", file_path)
    return int(match.group(1)) if match else None


@udf(IntegerType())
def extract_year_udf(file_path):
    import re
    match = re.search(r"/(\d{4})/", file_path)
    return int(match.group(1)) if match else None


def translate_tile_name(tile_name: str) -> str:
    """
    Translate between UK OS grid tile name formats:
    - TL0045 ↔ TL04NW
    """
    NS_dict = {"S": "0", "N": "5"}
    EW_dict = {"W": "0", "E": "5"}

    assert len(tile_name) == 6
    code = tile_name[2:6].upper()

    try:
        int(code)
        NS_dict = {v: k for k, v in NS_dict.items()}
        EW_dict = {v: k for k, v in EW_dict.items()}
        direction_code = code[0] + code[2] + NS_dict[code[3]] + EW_dict[code[1]]
        return tile_name[:2].upper() + direction_code
    except ValueError:
        number_code = code[0] + EW_dict[code[3]] + code[1] + NS_dict[code[2]]
        return tile_name[:2].lower() + number_code


def generate_vom_paths_df(sedona: SparkSession, chm_tiles_dir: Path, tile_name_pattern: str = r"VOM_([A-Z]{2}\d{4})_") -> pd.DataFrame:
    """Scan VOM raster directory and return a metadata DataFrame."""
    logging.debug("Generating VOM paths DataFrame")

    vom_sdf = sedona.read.format("binaryFile").load(f"{str(chm_tiles_dir)}/*/*.tif")
    vom_sdf.createOrReplaceTempView("vom")

    extract_ref_udf = make_extract_grid_reference_udf(tile_name_pattern)
    vom_raster_paths_sdf = sedona.sql("SELECT path FROM vom")
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("file_type", extract_vom_type_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("TILE_NAME", extract_ref_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.withColumn("year", extract_year_udf(vom_sdf["path"]))
    vom_raster_paths_sdf = vom_raster_paths_sdf.filter(vom_raster_paths_sdf["file_type"] == "CHM")

    vom_df = vom_raster_paths_sdf.toPandas()
    vom_df["path"] = vom_df["path"].str.replace("file:", "", regex=False)
    vom_df.sort_values(["TILE_NAME", "year"], ascending=[True, False], inplace=True)
    vom_df.reset_index(drop=True, inplace=True)
    return vom_df


def generate_tree_paths_df(
    trees_dir: Path,
    file_pattern: str = r"VOM_trees_([A-Z]{2}\d{4})_(\d{4})\.gpkg",
) -> pd.DataFrame:
    """Scan tree vector directory and return a metadata DataFrame."""
    logging.debug("Generating tree paths DataFrame")

    tree_paths = list(trees_dir.glob("*.gpkg"))
    metadata = []
    for path in tree_paths:
        match = re.search(file_pattern, path.name)
        if match:
            tile_name, year = match.groups()
            metadata.append({"TILE_NAME": tile_name, "year": int(year), "path": str(path)})

    df = pd.DataFrame(metadata)
    if not df.empty:
        df.sort_values(["TILE_NAME", "year"], ascending=[True, False], inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df
