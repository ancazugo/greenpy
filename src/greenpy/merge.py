"""
Final aggregation pipeline: merges T3, T30, T300, spectral, and tree count outputs.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import save_temp_file


def merge_output_csv(sedona: SparkSession, cfg: GreenPyConfig, t3_buffer_lst: list[int], file_format: str = "parquet") -> None:
    """Merge per-geo CSV files from each module into consolidated parquet files."""
    logging.info("Merging module CSV outputs into parquet")

    db_dir = Path(cfg.output.base_dir) / "database"
    base = Path(cfg.output.base_dir)

    for buffer in t3_buffer_lst:
        t3_parquet = db_dir / f"T3_{buffer}m.parquet"
        t3_sdf = sedona.read.format("csv").option("header", True).load(str(base / "T3") + f"/*{buffer}m.csv")
        t3_sdf = t3_sdf.withColumnRenamed("tree_count", f"tree_count_{buffer}m")
        t3_sdf.createOrReplaceTempView(f"t3_{buffer}m")
        save_temp_file(t3_sdf, t3_parquet, coalesce=1, file_format=file_format)

    for name in ["T30", "T300", "Spectral", "Tree_count"]:
        sdf = sedona.read.format("csv").option("header", True).load(str(base / name))
        sdf.createOrReplaceTempView(name.lower())
        save_temp_file(sdf, db_dir / f"{name}.parquet", coalesce=1, file_format=file_format)


def read_parquet_files(sedona: SparkSession, cfg: GreenPyConfig, t3_buffer_lst: list[int]) -> dict:
    """Load consolidated parquet files and register as Spark temp views."""
    db_dir = Path(cfg.output.base_dir) / "database"

    for name in ["T30", "T300", "Spectral", "Tree_count"]:
        sdf = sedona.read.format("parquet").load(str(db_dir / f"{name}.parquet"))
        sdf.createOrReplaceTempView(name.lower())

    buildings_parquet = db_dir / "buildings.parquet"
    if buildings_parquet.exists():
        sedona.read.format("parquet").load(str(buildings_parquet)).createOrReplaceTempView("buildings")

    census_parquet = db_dir / "census_boundaries.parquet"
    if census_parquet.exists():
        sedona.read.format("parquet").load(str(census_parquet)).createOrReplaceTempView("boundaries")

    overlay_parquet = db_dir / "census_buildings_overlay.parquet"
    if overlay_parquet.exists():
        sedona.read.format("parquet").load(str(overlay_parquet)).createOrReplaceTempView("boundaries_buildings_overlay")

    sdf_dict = {}
    for buffer in t3_buffer_lst:
        t3_parquet = db_dir / f"T3_{buffer}m.parquet"
        sdf = sedona.read.format("parquet").load(str(t3_parquet))
        sdf.createOrReplaceTempView(f"t3_{buffer}m")
        sdf_dict[f"t3_{buffer}m"] = sdf

    return sdf_dict


def aggregate_t30(sedona: SparkSession, geo_level: str) -> DataFrame:
    t30_agg = sedona.sql(f"""
    SELECT {geo_level},
    ROUND(SUM(canopy_cover * total_pixels) / SUM(total_pixels), 2) AS canopy_cover
    FROM t30
    GROUP BY {geo_level}
    """)
    t30_agg.createOrReplaceTempView("t30_agg")
    return t30_agg


def aggregate_tree_count(sedona: SparkSession, geo_level: str, sub_geo_level: str) -> DataFrame:
    tree_count_agg = sedona.sql(f"""
    SELECT b.{geo_level}, SUM(t.tree_count) AS total_trees
    FROM tree_count t
    LEFT JOIN boundaries b ON t.{sub_geo_level} = b.{sub_geo_level}
    GROUP BY {geo_level}
    """)
    tree_count_agg.createOrReplaceTempView("tree_count_agg")
    return tree_count_agg


def merge_t30_and_spectral(sedona: SparkSession, geo_level: str) -> DataFrame:
    t30_spectral = sedona.sql(f"""
    SELECT s.*, t30_agg.canopy_cover
    FROM t30_agg
    RIGHT JOIN spectral s ON t30_agg.{geo_level} = s.{geo_level}
    """)
    t30_spectral.createOrReplaceTempView("t30_spectral")
    return t30_spectral


def merge_t3_and_t300(sedona: SparkSession, t3_buffer_lst: list[int]) -> DataFrame:
    tree_count_cols = ", ".join([f"t3_{b}m.tree_count_{b}m" for b in t3_buffer_lst])
    joins = "\n".join([
        f"FULL JOIN t3_{b}m ON t300.building_id = t3_{b}m.building_id"
        for b in t3_buffer_lst
    ])
    query = f"""
    SELECT t300.*, b.distance_water, b.map_use, b.building_area, {tree_count_cols}
    FROM t300
    JOIN buildings b ON t300.building_id = b.building_id
    {joins}
    """
    t3_300 = sedona.sql(query)
    t3_300 = t3_300.fillna({f"tree_count_{b}m": 0 for b in t3_buffer_lst})
    t3_300.createOrReplaceTempView("t3_300")
    return t3_300


def aggregate_t3_300_by_boundaries(sedona: SparkSession, geo_level: str, t3_buffer_lst: list[int]) -> DataFrame:
    t3_300_boundaries = sedona.sql(f"""
    SELECT DISTINCT bbo.{geo_level}, t3_300.* FROM t3_300
    LEFT JOIN boundaries_buildings_overlay bbo ON t3_300.building_id = bbo.building_id
    """)
    t3_300_boundaries.createOrReplaceTempView("t3_300_boundaries")

    avg_tree_cols = ", ".join([f"ROUND(AVG(tree_count_{b}m), 2) as tree_count_{b}m" for b in t3_buffer_lst])
    t3_300_agg = sedona.sql(f"""
    SELECT {geo_level}, {avg_tree_cols},
    ROUND(AVG(distance_manhattan), 2) as park_distance_manhattan,
    ROUND(AVG(distance_euclidean), 2) as park_distance_euclidean,
    ROUND(AVG(distance_water), 2) as water_distance
    FROM t3_300_boundaries
    GROUP BY {geo_level}
    """)
    t3_300_agg.createOrReplaceTempView("t3_300_agg")
    return t3_300_agg


def merge_all(sedona: SparkSession, geo_level: str) -> DataFrame:
    result = sedona.sql(f"""
    SELECT t3_300_agg.*, ts.canopy_cover,
    ROUND(ts.NDBI, 2) as NDBI, ROUND(ts.NDVI, 2) as NDVI, ROUND(ts.NDWI, 2) as NDWI,
    tca.total_trees
    FROM t3_300_agg
    JOIN t30_spectral ts ON t3_300_agg.{geo_level} = ts.{geo_level}
    JOIN tree_count_agg tca ON t3_300_agg.{geo_level} = tca.{geo_level}
    """)
    result.createOrReplaceTempView("t3_30_300_spectral")
    return result


def process_data(
    sedona: SparkSession,
    cfg: GreenPyConfig,
    geo_level: str,
    sub_geo_level: str,
    t3_buffer_lst: list[int] = None,
) -> pd.DataFrame:
    if t3_buffer_lst is None:
        t3_buffer_lst = [10, 25, 50, 75, 100]

    logging.info("Starting merge pipeline")

    read_parquet_files(sedona, cfg, t3_buffer_lst)
    aggregate_t30(sedona, geo_level)
    aggregate_tree_count(sedona, geo_level, sub_geo_level)
    merge_t30_and_spectral(sedona, geo_level)
    merge_t3_and_t300(sedona, t3_buffer_lst)
    aggregate_t3_300_by_boundaries(sedona, geo_level, t3_buffer_lst)
    result_sdf = merge_all(sedona, geo_level)

    result_df = result_sdf.toPandas()
    tree_cols = [f"tree_count_{b}m" for b in t3_buffer_lst]
    result_df[tree_cols] = result_df[tree_cols].fillna(0)

    spectral_cols = ["NDBI", "NDVI", "NDWI"]
    present_spectral = [c for c in spectral_cols if c in result_df.columns]
    columns = (
        [geo_level, "total_trees"]
        + tree_cols
        + ["canopy_cover", "park_distance_manhattan", "park_distance_euclidean", "water_distance"]
        + present_spectral
    )
    result_df = result_df[[c for c in columns if c in result_df.columns]]

    db_dir = Path(cfg.output.base_dir) / "database"
    result_df.to_parquet(db_dir / "T3_30_300_spectral.parquet", index=False)

    logging.info("Merge pipeline completed")
    return result_df
