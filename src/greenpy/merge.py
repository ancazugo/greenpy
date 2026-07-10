"""
Final aggregation pipeline: merges T3, T30, T300, spectral, and tree count outputs.

Spectral and T30_buildings are optional throughout — when their outputs don't
exist the merged table simply omits the corresponding columns.
"""

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .pipeline import build_buildings_overlay, ensure_dggs_files
from .utils.data_processing import save_temp_file


def merge_output_csv(sedona: SparkSession, cfg: GreenPyConfig, t3_buffer_lst: list[int], file_format: str = "parquet") -> None:
    """Consolidate the per-geo CSV files of each module into one parquet per module.

    Modules with no CSV output (e.g. Spectral when it was never run) are
    skipped with a warning instead of failing.
    """
    logger.info("Merging module CSV outputs into parquet")

    db_dir = Path(cfg.output.base_dir) / "database"
    base = Path(cfg.output.base_dir)

    for buffer in t3_buffer_lst:
        if not list((base / "T3").glob(f"*_{buffer}m.csv")):
            logger.warning(f"No T3 CSV outputs found for buffer {buffer}m — skipping")
            continue
        t3_parquet = db_dir / f"T3_{buffer}m.parquet"
        t3_sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(str(base / "T3") + f"/*{buffer}m.csv")
        save_temp_file(t3_sdf, t3_parquet, coalesce=1, file_format=file_format)

    for name in ["T30", "T300", "Spectral", "Tree_count", "Visibility"]:
        if not list((base / name).glob("*.csv")):
            logger.warning(f"No {name} CSV outputs found — skipping")
            continue
        sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(str(base / name))
        save_temp_file(sdf, db_dir / f"{name}.parquet", coalesce=1, file_format=file_format)

    for buffer in _t30_buildings_buffers(base / "T30_buildings", "csv"):
        sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(
            str(base / "T30_buildings") + f"/*_{buffer}m.csv"
        )
        save_temp_file(sdf, db_dir / f"T30_buildings_{buffer}m.parquet", coalesce=1, file_format=file_format)


def _t30_buildings_buffers(directory: Path, extension: str) -> list[int]:
    """Buffer radii for which T30_buildings outputs exist in directory (from filenames)."""
    if not directory.exists():
        return []
    return sorted({
        int(m.group(1))
        for f in directory.glob(f"T30_buildings_*m.{extension}")
        if (m := re.search(rf"_(\d+)m\.{extension}$", f.name))
    })


def read_parquet_files(
    sedona: SparkSession, cfg: GreenPyConfig, t3_buffer_lst: list[int],
    dggs: str | None = None, dggs_resolution: int | None = None,
) -> dict:
    """Register available consolidated parquet files as Spark temp views.

    Returns a dict with a boolean per optional table ('spectral') and raises
    if a required module output is missing. Rebuilds the buildings overlay
    lookup if the parquet cache predates it. When a DGGS is given, the
    boundaries and overlay views use the grid cells.
    """
    db_dir = Path(cfg.output.base_dir) / "database"

    missing = []
    for name in ["T30", "T300", "Tree_count"]:
        p = db_dir / f"{name}.parquet"
        if not p.exists():
            missing.append(name)
            continue
        sedona.read.format("parquet").load(str(p)).createOrReplaceTempView(name.lower())

    for buffer in t3_buffer_lst:
        p = db_dir / f"T3_{buffer}m.parquet"
        if not p.exists():
            missing.append(f"T3 ({buffer}m)")
            continue
        sedona.read.format("parquet").load(str(p)).createOrReplaceTempView(f"t3_{buffer}m")

    if missing:
        raise FileNotFoundError(
            f"Missing consolidated outputs for: {missing}. Run those modules before Merge."
        )

    spectral_parquet = db_dir / "Spectral.parquet"
    has_spectral = spectral_parquet.exists()
    if has_spectral:
        sedona.read.format("parquet").load(str(spectral_parquet)).createOrReplaceTempView("spectral")
    else:
        logger.warning("No Spectral output found — merged table will omit spectral indices")

    t30_buildings_buffers = _t30_buildings_buffers(db_dir, "parquet")
    for buffer in t30_buildings_buffers:
        sedona.read.format("parquet").load(
            str(db_dir / f"T30_buildings_{buffer}m.parquet")
        ).createOrReplaceTempView(f"t30_buildings_{buffer}m")
    if not t30_buildings_buffers:
        logger.warning("No T30_buildings output found — merged table will omit per-building canopy cover")

    sedona.read.format("geoparquet").load(str(db_dir / "buildings.parquet")).createOrReplaceTempView("buildings")

    if dggs is not None:
        boundaries_parquet, overlay_parquet = ensure_dggs_files(sedona, db_dir, cfg, dggs, dggs_resolution)
    else:
        boundaries_parquet = db_dir / "census_boundaries.parquet"
        overlay_parquet = db_dir / "census_buildings_overlay.parquet"
        if not overlay_parquet.exists():
            build_buildings_overlay(db_dir, cfg)

    sedona.read.format("geoparquet").load(str(boundaries_parquet)).createOrReplaceTempView("boundaries")
    sedona.read.format("parquet").load(str(overlay_parquet)).createOrReplaceTempView("boundaries_buildings_overlay")

    return {"has_spectral": has_spectral, "t30_buildings_buffers": t30_buildings_buffers}


def _sub_to_geo_join(geo_level: str, sub_geo_level: str) -> str:
    """SQL fragment joining a sub_geo_level table alias `t` up to geo_level via boundaries."""
    return f"""
    LEFT JOIN (SELECT DISTINCT {sub_geo_level}, {geo_level} FROM boundaries) b
    ON t.{sub_geo_level} = b.{sub_geo_level}
    """


def aggregate_t30(sedona: SparkSession, geo_level: str, sub_geo_level: str) -> DataFrame:
    """Aggregate sub_geo_level canopy cover to geo_level, weighted by valid pixel counts."""
    if geo_level == sub_geo_level:
        query = f"""
        SELECT {geo_level},
        ROUND(SUM(canopy_cover * total_pixels) / SUM(total_pixels), 2) AS canopy_cover
        FROM t30
        GROUP BY {geo_level}
        """
    else:
        query = f"""
        SELECT b.{geo_level},
        ROUND(SUM(t.canopy_cover * t.total_pixels) / SUM(t.total_pixels), 2) AS canopy_cover
        FROM t30 t
        {_sub_to_geo_join(geo_level, sub_geo_level)}
        GROUP BY b.{geo_level}
        """
    t30_agg = sedona.sql(query)
    t30_agg.createOrReplaceTempView("t30_agg")
    return t30_agg


def aggregate_tree_count(sedona: SparkSession, geo_level: str, sub_geo_level: str) -> DataFrame:
    """Sum sub_geo_level tree counts up to geo_level."""
    if geo_level == sub_geo_level:
        query = f"SELECT {geo_level}, SUM(tree_count) AS total_trees FROM tree_count GROUP BY {geo_level}"
    else:
        query = f"""
        SELECT b.{geo_level}, SUM(t.tree_count) AS total_trees
        FROM tree_count t
        {_sub_to_geo_join(geo_level, sub_geo_level)}
        GROUP BY b.{geo_level}
        """
    tree_count_agg = sedona.sql(query)
    tree_count_agg.createOrReplaceTempView("tree_count_agg")
    return tree_count_agg


def aggregate_t30_buildings(sedona: SparkSession, geo_level: str, buffers: list[int]) -> DataFrame | None:
    """Average per-building canopy cover up to geo_level via the buildings overlay.

    One `building_canopy_cover_<buffer>m` column per buffer (named to avoid
    clashing with T30's per-area `canopy_cover`). Returns None when no
    T30_buildings output exists.
    """
    if not buffers:
        return None

    per_buffer = [
        sedona.sql(f"""
        SELECT bbo.{geo_level},
        ROUND(AVG(t.canopy_cover), 2) AS building_canopy_cover_{buffer}m
        FROM t30_buildings_{buffer}m t
        LEFT JOIN boundaries_buildings_overlay bbo ON t.building_id = bbo.building_id
        GROUP BY bbo.{geo_level}
        """)
        for buffer in buffers
    ]
    # full outer join: buffers may have run over different geo code sets
    t30b_agg = per_buffer[0]
    for sdf in per_buffer[1:]:
        t30b_agg = t30b_agg.join(sdf, on=geo_level, how="full")
    t30b_agg.createOrReplaceTempView("t30_buildings_agg")
    return t30b_agg


def merge_t30_and_spectral(sedona: SparkSession, geo_level: str, sub_geo_level: str, has_spectral: bool) -> DataFrame:
    """Attach geo_level-averaged spectral indices to aggregated canopy cover.

    Returns canopy cover alone when no Spectral output is available.
    """
    if not has_spectral:
        t30_spectral = sedona.sql(f"SELECT {geo_level}, canopy_cover FROM t30_agg")
        t30_spectral.createOrReplaceTempView("t30_spectral")
        return t30_spectral

    index_cols = [c for c in sedona.table("spectral").columns if c not in (geo_level, sub_geo_level)]
    avg_cols = ", ".join(f"ROUND(AVG(t.{c}), 2) AS {c}" for c in index_cols)
    if geo_level == sub_geo_level:
        sedona.sql(
            f"SELECT t.{geo_level}, {avg_cols} FROM spectral t GROUP BY t.{geo_level}"
        ).createOrReplaceTempView("spectral_agg")
    else:
        sedona.sql(f"""
        SELECT b.{geo_level}, {avg_cols}
        FROM spectral t
        {_sub_to_geo_join(geo_level, sub_geo_level)}
        GROUP BY b.{geo_level}
        """).createOrReplaceTempView("spectral_agg")

    sel = ", ".join(f"s.{c}" for c in index_cols)
    t30_spectral = sedona.sql(f"""
    SELECT t.{geo_level}, t.canopy_cover, {sel}
    FROM t30_agg t
    LEFT JOIN spectral_agg s ON t.{geo_level} = s.{geo_level}
    """)
    t30_spectral.createOrReplaceTempView("t30_spectral")
    return t30_spectral


def merge_t3_and_t300(sedona: SparkSession, t3_buffer_lst: list[int]) -> DataFrame:
    """Join per-building T300 distances with T3 tree counts (one column per buffer).

    Optional building attributes (distance_water, map_use, building_area) are
    included only when present in the buildings dataset.
    """
    extra_cols = [c for c in ("distance_water", "map_use", "building_area") if c in sedona.table("buildings").columns]
    extra_sel = "".join(f", b.{c}" for c in extra_cols)
    tree_count_cols = ", ".join([f"t3_{b}m.tree_count_{b}m" for b in t3_buffer_lst])
    joins = "\n".join([
        f"LEFT JOIN t3_{b}m ON t300.building_id = t3_{b}m.building_id"
        for b in t3_buffer_lst
    ])
    query = f"""
    SELECT t300.*{extra_sel}, {tree_count_cols}
    FROM t300
    JOIN buildings b ON t300.building_id = b.building_id
    {joins}
    """
    t3_300 = sedona.sql(query)
    t3_300 = t3_300.fillna({f"tree_count_{b}m": 0 for b in t3_buffer_lst})
    t3_300.createOrReplaceTempView("t3_300")
    return t3_300


def aggregate_t3_300_by_boundaries(sedona: SparkSession, geo_level: str, t3_buffer_lst: list[int]) -> DataFrame:
    """Average per-building T3/T300 metrics up to geo_level via the buildings overlay."""
    if geo_level in sedona.table("t3_300").columns:
        # geo_level == sub_geo_level (e.g. DGGS cells): already attached per building
        t3_300_boundaries = sedona.table("t3_300")
    else:
        t3_300_boundaries = sedona.sql(f"""
        SELECT DISTINCT bbo.{geo_level}, t3_300.* FROM t3_300
        LEFT JOIN boundaries_buildings_overlay bbo ON t3_300.building_id = bbo.building_id
        """)
    t3_300_boundaries.createOrReplaceTempView("t3_300_boundaries")

    avg_tree_cols = ", ".join([f"ROUND(AVG(tree_count_{b}m), 2) as tree_count_{b}m" for b in t3_buffer_lst])
    water_col = (
        ",\n    ROUND(AVG(distance_water), 2) as water_distance"
        if "distance_water" in t3_300_boundaries.columns else ""
    )
    t3_300_agg = sedona.sql(f"""
    SELECT {geo_level}, {avg_tree_cols},
    ROUND(AVG(distance_manhattan), 2) as park_distance_manhattan,
    ROUND(AVG(distance_euclidean), 2) as park_distance_euclidean{water_col}
    FROM t3_300_boundaries
    GROUP BY {geo_level}
    """)
    t3_300_agg.createOrReplaceTempView("t3_300_agg")
    return t3_300_agg


def merge_all(sedona: SparkSession, geo_level: str, t30_buildings_buffers: list[int] | None = None) -> DataFrame:
    """Join the building-level aggregates with canopy cover, spectral indices, and tree totals."""
    ts_cols = [c for c in sedona.table("t30_spectral").columns if c != geo_level]
    sel = "".join(f", ts.{c}" for c in ts_cols)
    t30b_sel, t30b_join = "", ""
    if t30_buildings_buffers:
        t30b_sel = "".join(f", tb.building_canopy_cover_{b}m" for b in t30_buildings_buffers)
        t30b_join = f"LEFT JOIN t30_buildings_agg tb ON t3_300_agg.{geo_level} = tb.{geo_level}"
    result = sedona.sql(f"""
    SELECT t3_300_agg.*{sel}{t30b_sel}, tca.total_trees
    FROM t3_300_agg
    LEFT JOIN t30_spectral ts ON t3_300_agg.{geo_level} = ts.{geo_level}
    LEFT JOIN tree_count_agg tca ON t3_300_agg.{geo_level} = tca.{geo_level}
    {t30b_join}
    """)
    result.createOrReplaceTempView("t3_30_300_spectral")
    return result


def process_data(
    sedona: SparkSession,
    cfg: GreenPyConfig,
    geo_level: str,
    sub_geo_level: str,
    t3_buffer_lst: list[int] = None,
    dggs: str | None = None,
    dggs_resolution: int | None = None,
) -> pd.DataFrame:
    """Run the full merge pipeline and write database/T3_30_300_spectral.parquet.

    Produces one row per geo_level unit with total trees, mean T3 tree counts
    per buffer, canopy cover, park distances, and (when available) water
    distance and spectral indices.
    """
    if t3_buffer_lst is None:
        t3_buffer_lst = [10, 25, 50, 75, 100]

    logger.info("Starting merge pipeline")

    tables = read_parquet_files(sedona, cfg, t3_buffer_lst, dggs=dggs, dggs_resolution=dggs_resolution)
    aggregate_t30(sedona, geo_level, sub_geo_level)
    aggregate_t30_buildings(sedona, geo_level, tables["t30_buildings_buffers"])
    aggregate_tree_count(sedona, geo_level, sub_geo_level)
    merge_t30_and_spectral(sedona, geo_level, sub_geo_level, tables["has_spectral"])
    merge_t3_and_t300(sedona, t3_buffer_lst)
    aggregate_t3_300_by_boundaries(sedona, geo_level, t3_buffer_lst)
    result_sdf = merge_all(sedona, geo_level, tables["t30_buildings_buffers"])

    result_df = result_sdf.toPandas()
    tree_cols = [f"tree_count_{b}m" for b in t3_buffer_lst]
    result_df[tree_cols] = result_df[tree_cols].fillna(0)

    leading = [geo_level, "total_trees"] + tree_cols + ["canopy_cover"] + [
        f"building_canopy_cover_{b}m" for b in tables["t30_buildings_buffers"]
    ] + ["park_distance_manhattan", "park_distance_euclidean", "water_distance"]
    ordered = [c for c in leading if c in result_df.columns]
    ordered += [c for c in result_df.columns if c not in ordered]
    result_df = result_df[ordered]

    db_dir = Path(cfg.output.base_dir) / "database"
    result_df.to_parquet(db_dir / "T3_30_300_spectral.parquet", index=False)

    logger.info("Merge pipeline completed")
    return result_df
