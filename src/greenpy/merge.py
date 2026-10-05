"""
Final aggregation pipeline: merges T3, T30, T300, spectral, and tree count outputs.

Spectral, T30_buildings and Visibility are optional throughout — when their
outputs don't exist the merged table simply omits the corresponding columns.
Besides the per-unit means, the 3-30-300 rule is evaluated per building
(compute_compliance) and summarised as the share of buildings meeting it.
"""

import re
from pathlib import Path

import pandas as pd
from loguru import logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .pipeline import build_buildings_overlay, ensure_dggs_files, unit_output_name
from .utils.data_processing import save_temp_file


def merge_output_csv(
    sedona: SparkSession, cfg: GreenPyConfig, t3_buffer_lst: list[int], file_format: str = "parquet",
    dggs: str | None = None, dggs_resolution: int | None = None,
) -> None:
    """Consolidate the per-geo CSV files of each module into one parquet per module.

    Modules with no CSV output (e.g. Spectral when it was never run) are
    skipped with a warning instead of failing. With a DGGS, T30 and Tree_count
    are read from (and written to) their grid-suffixed names, e.g. T30_h3_9.
    """
    logger.info("Merging module CSV outputs into parquet")

    db_dir = Path(cfg.output.base_dir) / "database"
    base = Path(cfg.output.base_dir)

    for buffer in t3_buffer_lst:
        if not list((base / "T3").glob(f"*_{buffer}m.csv")):
            logger.warning(f"No T3 CSV outputs found for buffer {buffer}m — skipping")
            continue
        t3_parquet = db_dir / f"T3_{buffer}m.parquet"
        # the underscore matters: "*50m.csv" would also match T3_<code>_150m.csv
        t3_sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(str(base / "T3") + f"/*_{buffer}m.csv")
        save_temp_file(t3_sdf, t3_parquet, coalesce=1, file_format=file_format)

    for module in ["T30", "T300", "Spectral", "Tree_count"]:
        name = unit_output_name(module, dggs, dggs_resolution)
        if not list((base / name).glob("*.csv")):
            logger.warning(f"No {name} CSV outputs found — skipping")
            continue
        sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(str(base / name))
        save_temp_file(sdf, db_dir / f"{name}.parquet", coalesce=1, file_format=file_format)

    # per-buffer modules: one consolidated parquet per buffer radius
    for name in ["T30_buildings", "Visibility"]:
        for buffer in _buffers_in(base / name, name, "csv"):
            sdf = sedona.read.format("csv").option("header", True).option("inferSchema", True).load(
                str(base / name) + f"/*_{buffer}m.csv"
            )
            save_temp_file(sdf, db_dir / f"{name}_{buffer}m.parquet", coalesce=1, file_format=file_format)


def _buffers_in(directory: Path, prefix: str, extension: str) -> list[int]:
    """Buffer radii for which `<prefix>_*<buffer>m.<extension>` outputs exist in directory."""
    if not directory.exists():
        return []
    return sorted({
        int(m.group(1))
        for f in directory.glob(f"{prefix}_*m.{extension}")
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
    for module in ["T30", "T300", "Tree_count"]:
        name = unit_output_name(module, dggs, dggs_resolution)
        p = db_dir / f"{name}.parquet"
        if not p.exists():
            missing.append(name)
            continue
        sedona.read.format("parquet").load(str(p)).createOrReplaceTempView(module.lower())

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

    t30_buildings_buffers = _buffers_in(db_dir, "T30_buildings", "parquet")
    for buffer in t30_buildings_buffers:
        sedona.read.format("parquet").load(
            str(db_dir / f"T30_buildings_{buffer}m.parquet")
        ).createOrReplaceTempView(f"t30_buildings_{buffer}m")
    if not t30_buildings_buffers:
        logger.warning("No T30_buildings output found — merged table will omit per-building canopy cover")

    visibility_buffers = _buffers_in(db_dir, "Visibility", "parquet")
    for buffer in visibility_buffers:
        sedona.read.format("parquet").load(
            str(db_dir / f"Visibility_{buffer}m.parquet")
        ).createOrReplaceTempView(f"visibility_{buffer}m")

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

    return {
        "has_spectral": has_spectral,
        "t30_buildings_buffers": t30_buildings_buffers,
        "visibility_buffers": visibility_buffers,
    }


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


def _aggregate_building_buffers(
    sedona: SparkSession, geo_level: str, buffers: list[int], view_prefix: str,
    columns: list[tuple[str, str]], agg_view: str, ratios: list[tuple[str, str, str]] = (),
) -> DataFrame | None:
    """Average per-building metrics up to geo_level via the buildings overlay, one column per metric and buffer.

    columns holds (value_col, out_prefix) pairs; ratios holds (numerator_col,
    denominator_col, out_prefix) triples, aggregated as 100 * SUM / SUM (null
    where the denominator sums to 0). Columns missing from a buffer's table are
    skipped. Output columns are `<out_prefix>_<buffer>m`. Returns None when no
    buffer exists.
    """
    if not buffers:
        return None

    def select(buffer: int) -> str:
        present = set(sedona.table(f"{view_prefix}_{buffer}m").columns)
        return ", ".join([
            *(f"ROUND(AVG(t.{value_col}), 2) AS {out_prefix}_{buffer}m"
              for value_col, out_prefix in columns if value_col in present),
            *(f"ROUND(100 * SUM(t.{num}) / NULLIF(SUM(t.{den}), 0), 2) AS {out_prefix}_{buffer}m"
              for num, den, out_prefix in ratios if {num, den} <= present),
        ])

    per_buffer = [
        sedona.sql(f"""
        SELECT bbo.{geo_level}, {select(buffer)}
        FROM {view_prefix}_{buffer}m t
        LEFT JOIN boundaries_buildings_overlay bbo ON t.building_id = bbo.building_id
        GROUP BY bbo.{geo_level}
        """)
        for buffer in buffers
    ]
    # full outer join: buffers may have run over different geo code sets
    agg = per_buffer[0]
    for sdf in per_buffer[1:]:
        agg = agg.join(sdf, on=geo_level, how="full")
    agg.createOrReplaceTempView(agg_view)
    return agg


def aggregate_t30_buildings(sedona: SparkSession, geo_level: str, buffers: list[int]) -> DataFrame | None:
    """Average per-building canopy cover up to geo_level via the buildings overlay.

    One `building_canopy_cover_<buffer>m` column per buffer (named to avoid
    clashing with T30's per-area `canopy_cover`). Returns None when no
    T30_buildings output exists.
    """
    return _aggregate_building_buffers(
        sedona, geo_level, buffers, "t30_buildings", [("canopy_cover", "building_canopy_cover")], "t30_buildings_agg"
    )


def aggregate_visibility(sedona: SparkSession, geo_level: str, buffers: list[int]) -> DataFrame | None:
    """Average per-building visible tree counts up to geo_level (`visible_trees_<buffer>m`, from any
    floor, and `visible_trees_ground_<buffer>m`, from the ground floor), and `share_visible_<buffer>m`:
    the % of the unit's building-tree pairs within the buffer where the tree is in view."""
    return _aggregate_building_buffers(
        sedona, geo_level, buffers, "visibility",
        [("visible_trees", "visible_trees"), ("visible_trees_ground", "visible_trees_ground")], "visibility_agg",
        ratios=[("visible_trees", "candidate_trees", "share_visible")],
    )


# 3-30-300 thresholds (Konijnendijk 2023)
RULE_MIN_TREES = 3
RULE_MIN_CANOPY = 30.0
RULE_MAX_DISTANCE = 300.0


def evaluate_rule(
    buildings_df: pd.DataFrame, tree_col: str | None, canopy_col: str | None, distance_col: str,
    tree_cols_by_metric: dict[str, str | None] | None = None,
) -> pd.DataFrame:
    """Add per-building 3-30-300 pass/fail columns (nullable booleans).

    meets_3: tree count >= 3; meets_30: canopy cover >= 30 %; meets_300:
    park distance <= 300 m. A missing input (or tree_col/canopy_col None)
    leaves that criterion null; meets_3_30_300 is null unless all three are known.
    tree_cols_by_metric (e.g. {"proximity": "tree_count_50m", "visibility":
    "visible_trees_50m"}) adds a meets_3_<metric> flag per tree count, and the
    combined meets_3_30_300_<metric> it gives, so the metrics not chosen for
    meets_3 stay comparable.
    """
    df = buildings_df.copy()

    def _flag(col: str | None, ok) -> pd.Series:
        if col is None:
            return pd.Series(pd.NA, index=df.index, dtype="boolean")
        values = pd.to_numeric(df[col], errors="coerce")
        return ok(values).astype("boolean").mask(values.isna())

    df["meets_3"] = _flag(tree_col, lambda v: v >= RULE_MIN_TREES)
    for metric, col in (tree_cols_by_metric or {}).items():
        df[f"meets_3_{metric}"] = _flag(col, lambda v: v >= RULE_MIN_TREES)
    df["meets_30"] = _flag(canopy_col, lambda v: v >= RULE_MIN_CANOPY)
    df["meets_300"] = _flag(distance_col, lambda v: v <= RULE_MAX_DISTANCE)

    def _all_known(cols: list[str]) -> pd.Series:
        # Kleene AND would make "False & NA" False; the combined rule needs all three known
        parts = df[cols]
        return parts.all(axis=1).astype("boolean").mask(parts.isna().any(axis=1))

    df["meets_3_30_300"] = _all_known(["meets_3", "meets_30", "meets_300"])
    for metric in tree_cols_by_metric or {}:
        df[f"meets_3_30_300_{metric}"] = _all_known([f"meets_3_{metric}", "meets_30", "meets_300"])
    return df


def summarise_rule(evaluated_df: pd.DataFrame, geo_level: str) -> pd.DataFrame:
    """Per geo_level unit, % of buildings meeting each criterion (over buildings where it is known)."""
    flags = ["meets_3", "meets_30", "meets_300", "meets_3_30_300"]
    # per-metric flags: meets_3_<metric> and meets_3_30_300_<metric>
    flags += [c for c in evaluated_df.columns if c.startswith("meets_3_") and c != "meets_3_30_300"]
    numeric = evaluated_df[[geo_level]].copy()
    for f in flags:
        numeric[f] = evaluated_df[f].astype("Float64")
    out = numeric.groupby(geo_level)[flags].mean().mul(100).round(2)
    out.columns = [f"pct_{f}" for f in flags]
    return out.astype(float).reset_index()


def compute_compliance(
    sedona: SparkSession,
    cfg: GreenPyConfig,
    geo_level: str,
    sub_geo_level: str,
    t3_buffer_lst: list[int],
    t30_buildings_buffers: list[int],
    rule_t3_buffer: int = 50,
    rule_t30_buffer: int | None = None,
    rule_distance: str = "euclidean",
    out_name: str = "T3_30_300_buildings",
    rule_t3_metric: str = "proximity",
    visibility_buffers: list[int] | None = None,
) -> DataFrame:
    """Evaluate the 3-30-300 rule per building and aggregate the pass rates to geo_level.

    3: trees within rule_t3_buffer metres — counted by T3 (rule_t3_metric
    "proximity") or seen from some floor by Visibility ("visibility"); both
    flags are kept as meets_3_proximity / meets_3_visibility when their
    outputs exist. 30: canopy cover of the
    building's sub_geo_level unit (the neighbourhood reading of the rule), or
    T30_buildings canopy within rule_t30_buffer metres when given. 300: park
    distance, straight-line (`euclidean`, the WHO guideline) or road
    `network`. Writes database/<out_name>.parquet and registers the
    `compliance_agg` view with pct_meets_* columns.
    """
    if rule_distance not in ("euclidean", "network"):
        raise ValueError(f"rule_distance must be 'euclidean' or 'network', got {rule_distance!r}")
    distance_col = "distance_euclidean" if rule_distance == "euclidean" else "distance_manhattan"

    if rule_t3_metric not in ("proximity", "visibility"):
        raise ValueError(f"rule_t3_metric must be 'proximity' or 'visibility', got {rule_t3_metric!r}")
    tree_sel, prox_col, vis_col, vis_join = "", None, None, ""
    if rule_t3_buffer in t3_buffer_lst:
        prox_col = f"tree_count_{rule_t3_buffer}m"
        tree_sel = f", t.{prox_col}"
    elif rule_t3_metric == "proximity":
        logger.warning(f"No T3 output for --rule_t3_buffer {rule_t3_buffer}m in {t3_buffer_lst} — meets_3 left null")
    if rule_t3_buffer in (visibility_buffers or []):
        vis_col = f"visible_trees_{rule_t3_buffer}m"
        tree_sel += f", v.visible_trees AS {vis_col}, v.visible_trees_ground AS visible_trees_ground_{rule_t3_buffer}m"
        vis_join = f"LEFT JOIN visibility_{rule_t3_buffer}m v ON t.building_id = v.building_id"
    elif rule_t3_metric == "visibility":
        raise FileNotFoundError(
            f"--rule_t3_metric visibility needs Visibility output at --rule_t3_buffer {rule_t3_buffer}m "
            f"(found: {visibility_buffers or []})"
        )
    tree_col = vis_col if rule_t3_metric == "visibility" else prox_col
    by_metric = {m: c for m, c in (("proximity", prox_col), ("visibility", vis_col)) if c is not None}

    if rule_t30_buffer is None:
        canopy_col = "canopy_cover"
        canopy_sel = ", c.canopy_cover"
        canopy_join = f"LEFT JOIN t30 c ON o.{sub_geo_level} = c.{sub_geo_level}"
    else:
        if rule_t30_buffer not in t30_buildings_buffers:
            raise FileNotFoundError(
                f"--rule_t30_buffer {rule_t30_buffer} needs T30_buildings output at that buffer "
                f"(found: {t30_buildings_buffers})"
            )
        canopy_col = f"building_canopy_cover_{rule_t30_buffer}m"
        canopy_sel = f", c.canopy_cover AS {canopy_col}"
        canopy_join = f"LEFT JOIN t30_buildings_{rule_t30_buffer}m c ON t.building_id = c.building_id"

    levels = list(dict.fromkeys([geo_level, sub_geo_level]))
    level_sel = ", ".join(f"o.{c}" for c in levels)
    buildings_df = sedona.sql(f"""
        SELECT DISTINCT t.building_id, {level_sel}{tree_sel}, t.{distance_col}{canopy_sel}
        FROM t3_300 t
        JOIN boundaries_buildings_overlay o ON t.building_id = o.building_id
        {canopy_join}
        {vis_join}
    """).toPandas()

    evaluated = evaluate_rule(buildings_df, tree_col, canopy_col, distance_col, by_metric)
    db_dir = Path(cfg.output.base_dir) / "database"
    write_rule_parquet(evaluated, db_dir / f"{out_name}.parquet", {
        "t3_metric": rule_t3_metric, "t3_col": tree_col, "t30_col": canopy_col, "distance_col": distance_col,
    })

    summary = summarise_rule(evaluated, geo_level)
    compliance_sdf = sedona.createDataFrame(summary)
    compliance_sdf.createOrReplaceTempView("compliance_agg")
    logger.info(
        f"3-30-300 rule: 3 = {'trees visible within' if rule_t3_metric == 'visibility' else 'T3 trees within'} "
        f"{rule_t3_buffer}m, 30 = "
        f"{'sub-geo unit canopy' if rule_t30_buffer is None else f'T30_buildings {rule_t30_buffer}m'}, "
        f"300 = {rule_distance} distance; {int(evaluated['meets_3_30_300'].sum())}/{len(evaluated)} buildings pass"
    )
    return compliance_sdf


def write_rule_parquet(evaluated: pd.DataFrame, path: Path, rule: dict) -> None:
    """Per-building rule table with the columns behind each criterion in the `greenpy_rule` metadata."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.Table.from_pandas(evaluated, preserve_index=False)
    meta = {**(table.schema.metadata or {}), b"greenpy_rule": json.dumps(rule).encode()}
    pq.write_table(table.replace_schema_metadata(meta), path)


def read_rule_metadata(path: Path) -> dict | None:
    """The `greenpy_rule` metadata of a per-building rule table (None for tables written before it existed)."""
    import json

    import pyarrow.parquet as pq

    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(b"greenpy_rule")
    return json.loads(raw) if raw else None


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


def merge_all(
    sedona: SparkSession, geo_level: str,
    t30_buildings_buffers: list[int] | None = None,
    visibility_buffers: list[int] | None = None,
) -> DataFrame:
    """Join the building-level aggregates with canopy cover, spectral indices, tree totals and rule pass rates.

    One row per geo_level unit that has buildings, canopy cover or trees: a unit
    without buildings (a forest, a park) still reports its canopy and tree
    total, with the building metrics left null. `has_buildings` marks units
    with building aggregates.
    """
    ts_cols = [c for c in sedona.table("t30_spectral").columns if c != geo_level]
    sel = "".join(f", ts.{c}" for c in ts_cols)
    agg_cols = [c for c in sedona.table("t3_300_agg").columns if c != geo_level]
    agg_sel = "".join(f", t3_300_agg.{c}" for c in agg_cols)
    extra_sel, extra_join = "", ""
    if t30_buildings_buffers:
        extra_sel += "".join(f", tb.building_canopy_cover_{b}m" for b in t30_buildings_buffers)
        extra_join += f"LEFT JOIN t30_buildings_agg tb ON u.{geo_level} = tb.{geo_level}\n"
    if visibility_buffers:
        va_cols = [c for c in sedona.table("visibility_agg").columns if c != geo_level]
        extra_sel += "".join(f", va.{c}" for c in va_cols)
        extra_join += f"LEFT JOIN visibility_agg va ON u.{geo_level} = va.{geo_level}\n"
    comp_cols = [c for c in sedona.table("compliance_agg").columns if c != geo_level]
    comp_sel = "".join(f", ca.{c}" for c in comp_cols)
    result = sedona.sql(f"""
    SELECT u.{geo_level}{agg_sel}{sel}{extra_sel}, tca.total_trees{comp_sel},
           t3_300_agg.{geo_level} IS NOT NULL AS has_buildings
    FROM (SELECT DISTINCT {geo_level} FROM boundaries WHERE {geo_level} IS NOT NULL) u
    LEFT JOIN t3_300_agg ON u.{geo_level} = t3_300_agg.{geo_level}
    LEFT JOIN t30_spectral ts ON u.{geo_level} = ts.{geo_level}
    LEFT JOIN tree_count_agg tca ON u.{geo_level} = tca.{geo_level}
    LEFT JOIN compliance_agg ca ON u.{geo_level} = ca.{geo_level}
    {extra_join}
    WHERE t3_300_agg.{geo_level} IS NOT NULL OR ts.{geo_level} IS NOT NULL OR tca.{geo_level} IS NOT NULL
    """)
    result.createOrReplaceTempView("t3_30_300_spectral")
    return result


def fill_unit_counts(result_df: pd.DataFrame, t3_buffer_lst: list[int]) -> pd.DataFrame:
    """Zero-fill counts that are missing because they are zero, and drop merge_all's has_buildings marker.

    Units with no tree at all are absent from Tree_count's CSVs (total_trees = 0);
    the mean trees per building is 0 where the unit has buildings and stays
    null where it has none.
    """
    result_df = result_df.copy()
    result_df["total_trees"] = result_df["total_trees"].fillna(0)
    has_buildings = result_df.pop("has_buildings").astype(bool)
    tree_cols = [c for c in (f"tree_count_{b}m" for b in t3_buffer_lst) if c in result_df.columns]
    result_df.loc[has_buildings, tree_cols] = result_df.loc[has_buildings, tree_cols].fillna(0)
    return result_df


def process_data(
    sedona: SparkSession,
    cfg: GreenPyConfig,
    geo_level: str,
    sub_geo_level: str,
    t3_buffer_lst: list[int] = None,
    dggs: str | None = None,
    dggs_resolution: int | None = None,
    rule_t3_buffer: int = 50,
    rule_t30_buffer: int | None = None,
    rule_distance: str = "euclidean",
    rule_t3_metric: str = "proximity",
) -> pd.DataFrame:
    """Run the full merge pipeline and write database/T3_30_300_spectral.parquet.

    With a DGGS the tables get the grid suffix (T3_30_300_spectral_h3_9.parquet,
    T3_30_300_buildings_h3_9.parquet) so they sit beside the census-unit run.

    Produces one row per geo_level unit with total trees, mean T3 tree counts
    per buffer, canopy cover, park distances, the % of buildings meeting each
    3-30-300 criterion and all three (pct_meets_*), and (when available)
    per-building canopy, visible trees, water distance and spectral indices.
    The per-building rule evaluation is written to
    database/T3_30_300_buildings.parquet.
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
    aggregate_visibility(sedona, geo_level, tables["visibility_buffers"])
    compute_compliance(
        sedona, cfg, geo_level, sub_geo_level, t3_buffer_lst, tables["t30_buildings_buffers"],
        rule_t3_buffer=rule_t3_buffer, rule_t30_buffer=rule_t30_buffer, rule_distance=rule_distance,
        out_name=unit_output_name("T3_30_300_buildings", dggs, dggs_resolution),
        rule_t3_metric=rule_t3_metric, visibility_buffers=tables["visibility_buffers"],
    )
    result_sdf = merge_all(sedona, geo_level, tables["t30_buildings_buffers"], tables["visibility_buffers"])

    result_df = fill_unit_counts(result_sdf.toPandas(), t3_buffer_lst)
    tree_cols = [f"tree_count_{b}m" for b in t3_buffer_lst]

    leading = [geo_level, "total_trees"] + tree_cols + ["canopy_cover"] + [
        f"building_canopy_cover_{b}m" for b in tables["t30_buildings_buffers"]
    ] + ["park_distance_manhattan", "park_distance_euclidean", "water_distance",
         "pct_meets_3", "pct_meets_30", "pct_meets_300", "pct_meets_3_30_300",
         "pct_meets_3_proximity", "pct_meets_3_visibility", "pct_meets_3_30_300_proximity",
         "pct_meets_3_30_300_visibility"]
    ordered = [c for c in leading if c in result_df.columns]
    ordered += [c for c in result_df.columns if c not in ordered]
    result_df = result_df[ordered]

    db_dir = Path(cfg.output.base_dir) / "database"
    result_df.to_parquet(db_dir / f"{unit_output_name('T3_30_300_spectral', dggs, dggs_resolution)}.parquet", index=False)

    logger.info("Merge pipeline completed")
    return result_df
