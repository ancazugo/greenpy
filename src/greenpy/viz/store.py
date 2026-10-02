"""Build and open `<base_dir>/database/viz.duckdb`, the dashboard's tile store.

Geometry is reprojected to Web Mercator once, rows are Hilbert-sorted so the
bbox filters in tile queries prune whole row groups, and per-metric stats
(quantiles, histograms) are precomputed so the UI never scans a full table.
The store is rebuilt whenever an input file changes.
"""

import hashlib
import json
import math
import os
import re
from dataclasses import asdict
from pathlib import Path

import duckdb
from loguru import logger

from .catalog import BUILDING_KEY, CRITERIA_MET, RULE_FLAGS, Catalog, Source, UnitLayer, describe

STORE_VERSION = 5
# building columns holding a census unit's name, e.g. "name:ADM3_code"
NAME_PREFIX = "name:"
N_CLASSES = 7
N_BINS = 40
# (max zoom, simplification tolerance in Web Mercator metres) for unit geometry columns
UNIT_SIMPLIFY = (("geom_z9", 9, 300.0), ("geom_z12", 12, 40.0))


def connect(path: Path | str, read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """DuckDB connection with the spatial extension loaded.

    GREENPY_DUCKDB_EXTENSIONS overrides where DuckDB installs extensions
    (default ~/.duckdb), for machines with a full or read-only home.
    """
    con = duckdb.connect(str(path), read_only=read_only)
    ext_dir = os.environ.get("GREENPY_DUCKDB_EXTENSIONS")
    if ext_dir:
        con.execute(f"SET extension_directory = '{ext_dir}'")
    con.execute("INSTALL spatial; LOAD spatial;")
    return con


def unit_table(layer_name: str) -> str:
    return "units_" + re.sub(r"\W", "_", layer_name.lower())


def fingerprint(catalog: Catalog, include_trees: bool) -> str:
    h = hashlib.sha256(f"{STORE_VERSION}|{include_trees}|{catalog.crs}".encode())
    for p in catalog.fingerprint_paths():
        if not include_trees and catalog.trees and p in catalog.trees.paths:
            continue
        st = p.stat()
        h.update(f"{p}|{st.st_size}|{st.st_mtime_ns}".encode())
    return h.hexdigest()


def _sql_str(s: str | Path) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def _ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _reader(src: Source) -> str:
    if src.fmt == "parquet":
        return f"read_parquet({_sql_str(src.path)})"
    return f"read_csv({_sql_str(src.path)}, header = true, union_by_name = true)"


def _source_crs(con: duckdb.DuckDBPyConnection, relation: str, geom_col: str, fallback: str) -> str:
    """CRS stored on the geometry type (GeoParquet / GDAL sources), else the fallback."""
    for name, dtype, *_ in con.execute(f"DESCRIBE SELECT {_ident(geom_col)} FROM {relation}").fetchall():
        if m := re.fullmatch(r"GEOMETRY\('(.+)'\)", dtype):
            return m.group(1)
    return fallback


def _to_mercator(expr: str, crs: str) -> str:
    return f"ST_Transform({expr}, {_sql_str(crs)}, 'EPSG:3857', always_xy := true)"


def _hilbert_order(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Rewrite a table sorted along a Hilbert curve over its extent."""
    minx, miny, maxx, maxy = con.execute(f"SELECT min(minx), min(miny), max(maxx), max(maxy) FROM {table}").fetchone()
    if minx is None:
        return
    box = f"{{'min_x': {minx}, 'min_y': {miny}, 'max_x': {maxx}, 'max_y': {maxy}}}::BOX_2D"
    con.execute(f"CREATE OR REPLACE TABLE {table} AS SELECT * FROM {table} ORDER BY ST_Hilbert(geom, {box})")


_BBOX = "ST_XMin(geom) AS minx, ST_YMin(geom) AS miny, ST_XMax(geom) AS maxx, ST_YMax(geom) AS maxy"


def _metric_views(con: duckdb.DuckDBPyConnection, sources: list[Source], prefix: str) -> list[tuple[str, list[str]]]:
    """One temp table per source, deduplicated on its key (cast to VARCHAR)."""
    views = []
    for i, src in enumerate(sources):
        name = f"{prefix}_{i}"
        cols = ", ".join(f"any_value({_ident(c)}) AS {_ident(m)}" for c, m in src.columns.items())
        con.execute(f"""
            CREATE TEMP TABLE {name} AS
            SELECT CAST({_ident(src.key)} AS VARCHAR) AS k, {cols}
            FROM {_reader(src)} WHERE {_ident(src.key)} IS NOT NULL GROUP BY 1
        """)
        views.append((name, list(src.columns.values())))
    return views


def _zero_fill(con: duckdb.DuckDBPyConnection, table: str, sources: list[Source], ids_sql: str, levels: list[str]) -> None:
    """Set missing metrics to 0 for features inside the geo codes a module ran over.

    ids_sql selects `id` and the census level columns for every feature.
    """
    for src in sources:
        if not src.zero_fill or not levels:
            continue
        codes = ", ".join(_sql_str(c) for c in src.zero_fill)
        inside = " OR ".join(f"CAST({_ident(lv)} AS VARCHAR) IN ({codes})" for lv in levels)
        for m in src.columns.values():
            con.execute(f"""
                UPDATE {table} SET {_ident(m)} = 0
                WHERE {_ident(m)} IS NULL AND id IN (SELECT id FROM ({ids_sql}) WHERE {inside})
            """)


def _join_metrics(views: list[tuple[str, list[str]]]) -> tuple[str, str]:
    """Select every metric once; a metric several sources provide takes the first non-null, in source order."""
    by_metric: dict[str, list[str]] = {}
    for v, cols in views:
        for m in cols:
            by_metric.setdefault(m, []).append(f"{v}.{_ident(m)}")
    sel = "".join(
        f", {refs[0]} AS {_ident(m)}" if len(refs) == 1 else f", COALESCE({', '.join(refs)}) AS {_ident(m)}"
        for m, refs in by_metric.items()
    )
    joins = "".join(f" LEFT JOIN {v} ON {v}.k = g.id" for v, _ in views)
    return sel, joins


def _build_buildings(con: duckdb.DuckDBPyConnection, catalog: Catalog) -> list[str]:
    rel = f"read_parquet({_sql_str(catalog.buildings)})"
    crs = _source_crs(con, rel, "geometry", catalog.crs)
    views = _metric_views(con, catalog.building_sources, "bm")
    sel, joins = _join_metrics(views)
    # the census units each building belongs to, shown in its details
    overlay = catalog.base_dir / "database" / "census_buildings_overlay.parquet"
    if overlay.exists():
        levels = [u.name for u in catalog.unit_layers if u.overlay == overlay]
        sel += "".join(f", CAST(o.{_ident(c)} AS VARCHAR) AS {_ident(c)}" for c in levels)
        joins += f" LEFT JOIN read_parquet({_sql_str(overlay)}) o ON CAST(o.{BUILDING_KEY} AS VARCHAR) = g.id"
        # and their names, when columns.geo_level_names says where they are
        for i, u in enumerate(u for u in catalog.unit_layers if u.overlay == overlay and u.name_col):
            names = f"(SELECT DISTINCT CAST({_ident(u.name)} AS VARCHAR) AS c, CAST({_ident(u.name_col)} AS VARCHAR) AS n FROM read_parquet({_sql_str(u.boundaries)}))"
            sel += f", nm{i}.n AS {_ident(NAME_PREFIX + u.name)}"
            joins += f" LEFT JOIN {names} nm{i} ON nm{i}.c = CAST(o.{_ident(u.name)} AS VARCHAR)"
    con.execute(f"""
        CREATE TABLE buildings AS
        SELECT g.id, g.geom, {_BBOX.replace('geom', 'g.geom')}{sel}
        FROM (SELECT CAST({BUILDING_KEY} AS VARCHAR) AS id, {_to_mercator('geometry', crs)} AS geom FROM {rel}) g
        {joins}
    """)
    if overlay.exists():
        _zero_fill(
            con, "buildings", catalog.building_sources,
            f"SELECT CAST({BUILDING_KEY} AS VARCHAR) AS id, * FROM read_parquet({_sql_str(overlay)})", levels,
        )
    metrics = [m for _, cols in views for m in cols]
    criteria = [f for f in RULE_FLAGS[:3] if f in metrics]
    if len(criteria) == 3:
        # how many of 3 / 30 / 300 a building meets, so the combined rule can be shown as a gradient
        con.execute(f"ALTER TABLE buildings ADD COLUMN {CRITERIA_MET} INTEGER")
        con.execute(f"UPDATE buildings SET {CRITERIA_MET} = " + " + ".join(f"CAST({_ident(f)} AS INTEGER)" for f in criteria))
        metrics.append(CRITERIA_MET)
    _hilbert_order(con, "buildings")
    con.execute("CREATE INDEX buildings_id ON buildings (id)")
    return metrics


def _build_parks(con: duckdb.DuckDBPyConnection, catalog: Catalog) -> dict:
    """Park polygons in Web Mercator, flagged `used` when T300 counts them (park_min_area_ha)."""
    parks = catalog.parks
    rel = f"read_parquet({_sql_str(parks.path)})"
    crs = _source_crs(con, rel, "geometry", catalog.crs)
    cols = {c.lower(): c for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").columns}
    name = f"CAST({_ident(cols['name'])} AS VARCHAR)" if "name" in cols else "CAST(NULL AS VARCHAR)"
    pid = f"CAST({_ident(cols['park_id'])} AS VARCHAR)" if "park_id" in cols else "CAST(row_number() OVER () AS VARCHAR)"
    # areas in the study-area CRS (metres), as T300's filter measures them
    area = f"ST_Area(ST_Transform(geometry, {_sql_str(crs)}, {_sql_str(catalog.crs)}, always_xy := true)) / 10000"
    min_ha = parks.min_area_ha or 0
    con.execute(f"""
        CREATE TABLE parks AS
        SELECT id, name, round(area_ha, 2) AS area_ha, area_ha >= {min_ha} AS used, geom, {_BBOX}
        FROM (
            SELECT {pid} AS id, {name} AS name, {area} AS area_ha, {_to_mercator('geometry', crs)} AS geom
            FROM {rel} WHERE geometry IS NOT NULL AND ST_Dimension(geometry) = 2
        )
    """)
    _hilbert_order(con, "parks")
    n, used = con.execute("SELECT count(*), count(*) FILTER (used) FROM parks").fetchone()
    return {"count": n, "used": used, "min_area_ha": parks.min_area_ha}


def _build_units(con: duckdb.DuckDBPyConnection, catalog: Catalog, layer: UnitLayer, building_metrics: list[str]) -> list[str]:
    table = unit_table(layer.name)
    rel = f"read_parquet({_sql_str(layer.boundaries)})"
    crs = _source_crs(con, rel, "geometry", catalog.crs)
    code = _ident(layer.name)
    n_rows, n_codes = con.execute(f"SELECT count(*), count(DISTINCT {code}) FROM {rel}").fetchone()
    # coarser census levels are dissolved from the finest-level polygons
    geom = "ST_Union_Agg(geometry)" if n_rows != n_codes else "any_value(geometry)"
    boundary_cols = con.sql(f"SELECT * FROM {rel} LIMIT 0").columns
    name = f", any_value(CAST({_ident(layer.name_col)} AS VARCHAR)) AS name" if layer.name_col in boundary_cols else ""
    con.execute(f"""
        CREATE TEMP TABLE {table}_geom AS
        SELECT CAST({code} AS VARCHAR) AS id, {_to_mercator(geom, crs)} AS geom{name}
        FROM {rel} WHERE {code} IS NOT NULL GROUP BY {code}
    """)

    views = _metric_views(con, layer.sources, f"{table}_m")
    metrics = list(dict.fromkeys(m for _, cols in views for m in cols))
    if layer.overlay is not None and not layer.merged and building_metrics:
        # averages of the per-building metrics; Merge's own table already has its means
        aggs = ["count(*) AS n_buildings"]
        for m in building_metrics:
            if describe(m).kind == "boolean":
                aggs.append(f"round(100 * avg(CAST(b.{_ident(m)} AS INTEGER)), 1) AS {_ident('pct_' + m)}")
            else:
                aggs.append(f"round(avg(b.{_ident(m)}), 2) AS {_ident('mean_' + m)}")
        con.execute(f"""
            CREATE TEMP TABLE {table}_bagg AS
            SELECT CAST(o.{code} AS VARCHAR) AS k, {', '.join(aggs)}
            FROM buildings b JOIN read_parquet({_sql_str(layer.overlay)}) o ON b.id = CAST(o.{BUILDING_KEY} AS VARCHAR)
            WHERE o.{code} IS NOT NULL GROUP BY 1
        """)
        bagg_cols = [c for c in con.table(f"{table}_bagg").columns if c != "k"]
        views.append((f"{table}_bagg", bagg_cols))
        metrics += bagg_cols

    sel, joins = _join_metrics(views)
    simplified = "".join(
        f", ST_SimplifyPreserveTopology(g.geom, {tol}) AS {col}" for col, _, tol in UNIT_SIMPLIFY
    )
    con.execute(f"""
        CREATE TABLE {table} AS
        SELECT g.id{', g.name' if name else ''}, g.geom{simplified}, {_BBOX.replace('geom', 'g.geom')}{sel}
        FROM {table}_geom g {joins}
    """)
    levels = [c for c in catalog.census_levels if c in boundary_cols]
    _zero_fill(con, table, layer.sources, f"SELECT CAST({code} AS VARCHAR) AS id, * FROM {rel}", levels)
    _hilbert_order(con, table)
    con.execute(f"CREATE INDEX {table}_id ON {table} (id)")
    return metrics


def _tree_crs(con: duckdb.DuckDBPyConnection, path: Path, fallback: str) -> str:
    rel = f"read_parquet({_sql_str(path)})" if path.suffix.lower() in (".parquet", ".geoparquet") else f"ST_Read({_sql_str(path)})"
    geom_col = "geometry" if rel.startswith("read_parquet") else "geom"
    crs = _source_crs(con, rel, geom_col, "")
    if not crs and not rel.startswith("read_parquet"):
        import pyogrio

        crs = pyogrio.read_info(path).get("crs") or ""
    return crs or fallback


def _build_trees(con: duckdb.DuckDBPyConnection, catalog: Catalog) -> dict:
    """Tree points in Web Mercator with a crown radius (m) and height when known."""
    trees = catalog.trees
    parts = []
    for path in trees.paths:
        is_parquet = path.suffix.lower() in (".parquet", ".geoparquet")
        rel = f"read_parquet({_sql_str(path)})" if is_parquet else f"ST_Read({_sql_str(path)})"
        geom_col = "geometry" if is_parquet else "geom"
        cols = {c.lower(): c for c in con.sql(f"SELECT * FROM {rel} LIMIT 0").columns}
        src_crs = _tree_crs(con, path, catalog.crs)
        # areas are measured in the study-area CRS (metres) even when the file is geographic
        local = f"ST_Transform({_ident(geom_col)}, {_sql_str(src_crs)}, {_sql_str(catalog.crs)}, always_xy := true)"
        area_col = cols.get(trees.area_col.lower())
        area = (
            f"CAST({_ident(area_col)} AS DOUBLE)" if area_col
            else "CASE WHEN ST_Dimension(g) = 2 THEN ST_Area(g) END"
        )
        height_col = cols.get(trees.height_col.lower())
        height = f"CAST({_ident(height_col)} AS DOUBLE)" if height_col else "CAST(NULL AS DOUBLE)"
        parts.append(f"""
            SELECT {_to_mercator('ST_Centroid(g)', catalog.crs)} AS geom,
                   round(sqrt(CASE WHEN a > 0 THEN a END / pi()), 2) AS radius_m, {height} AS height
            FROM (SELECT *, {area} AS a FROM (SELECT {local} AS g, * EXCLUDE ({_ident(geom_col)}) FROM {rel})) WHERE g IS NOT NULL
        """)
    con.execute(f"CREATE TABLE trees AS SELECT geom, ST_X(geom) AS minx, ST_Y(geom) AS miny, ST_X(geom) AS maxx, ST_Y(geom) AS maxy, radius_m, height FROM ({' UNION ALL '.join(parts)})")
    _hilbert_order(con, "trees")
    has_radius, has_height = con.execute("SELECT count(radius_m) > 0, count(height) > 0 FROM trees").fetchone()
    return {"count": con.execute("SELECT count(*) FROM trees").fetchone()[0], "sized": bool(has_radius), "has_height": bool(has_height)}


def metric_stats(con: duckdb.DuckDBPyConnection, table: str, metric: str) -> dict:
    """Summary used by the legend: counts, class breaks and a histogram.

    The histogram and equal-interval breaks span the 0.5-99.5 percentile range
    so a few extreme values don't flatten the ramp; outliers fall in the end bins.
    """
    col = _ident(metric)
    if describe(metric).kind == "boolean":
        n, t, f = con.execute(f"SELECT count(*), count(*) FILTER ({col}), count(*) FILTER (NOT {col}) FROM {table}").fetchone()
        return {"kind": "boolean", "n": n, "true": t, "false": f, "null": n - t - f}

    finite = f"{col} IS NOT NULL AND isfinite(CAST({col} AS DOUBLE))"
    n, n_valid, vmin, vmax, lo, hi, qs = con.execute(f"""
        SELECT count(*), count(*) FILTER ({finite}),
               min({col}) FILTER ({finite}), max({col}) FILTER ({finite}),
               quantile_cont({col}, 0.005) FILTER ({finite}), quantile_cont({col}, 0.995) FILTER ({finite}),
               quantile_cont({col}, {[i / N_CLASSES for i in range(1, N_CLASSES)]}) FILTER ({finite})
        FROM {table}
    """).fetchone()
    stats = {"kind": "numeric", "n": n, "null": n - n_valid, "min": vmin, "max": vmax}
    if not n_valid:
        return stats | {"quantile": [], "equal": [], "hist": [], "domain": None}
    integer = "INT" in str(con.sql(f"SELECT {col} FROM {table} LIMIT 0").types[0])
    if hi <= lo:
        lo, hi = vmin, vmax if vmax > vmin else vmin + 1
    if integer:
        # whole-number bins; interpolated percentiles would label the axis "0.2 ... 9.5"
        lo, hi = math.floor(lo), math.ceil(hi)
        width = max(1, math.ceil((hi - lo + 1) / N_BINS))
        n_bins = math.ceil((hi - lo + 1) / width)
        hi = lo + n_bins * width
    else:
        width, n_bins = (hi - lo) / N_BINS, N_BINS
    rows = con.execute(f"""
        SELECT least(greatest(CAST(floor((CAST({col} AS DOUBLE) - {lo}) / {width}) AS INTEGER), 0), {n_bins - 1}) AS bin, count(*)
        FROM {table} WHERE {finite} GROUP BY 1
    """).fetchall()
    hist = [0] * n_bins
    for b, c in rows:
        hist[b] = c
    quantiles = sorted(set(round(q, 4) for q in qs))
    if sum(q > vmin for q in quantiles) < 3:
        # one value (typically 0) is so common that the quantile breaks collapse onto
        # it: it keeps its own class and the remaining classes split the values above
        # it, starting at the smallest of them so they never share the tied value's colour
        rest_min, rest_qs = con.execute(f"""
            SELECT min({col}), quantile_cont({col}, {[i / (N_CLASSES - 2) for i in range(1, N_CLASSES - 2)]})
            FROM {table} WHERE {finite} AND {col} > {vmin}
        """).fetchone()
        if rest_min is not None:
            quantiles = sorted(set(quantiles) | {round(q, 4) for q in [rest_min, *rest_qs]})
    stats |= {
        "integer": integer,
        "domain": [lo, hi],
        "quantile": quantiles,
        "equal": [round(lo + (hi - lo) * i / N_CLASSES, 4) for i in range(1, N_CLASSES)],
        "hist": hist,
    }
    return stats


def _lonlat_bounds(con: duckdb.DuckDBPyConnection) -> list[float]:
    minx, miny, maxx, maxy = con.execute("SELECT min(minx), min(miny), max(maxx), max(maxy) FROM buildings").fetchone()
    r = 6378137.0

    def lon(x):
        return math.degrees(x / r)

    def lat(y):
        return math.degrees(2 * math.atan(math.exp(y / r)) - math.pi / 2)

    return [lon(minx), lat(miny), lon(maxx), lat(maxy)]


def build_store(catalog: Catalog, path: Path, include_trees: bool = True) -> None:
    """Write the store to a temp file, then move it into place (a crashed build leaves no half store)."""
    tmp = path.with_suffix(".building.duckdb")
    tmp.unlink(missing_ok=True)
    con = connect(tmp)
    try:
        logger.info("viz: loading buildings and building metrics")
        building_metrics = _build_buildings(con, catalog)
        layers = {"buildings": {"label": "Buildings", "table": "buildings", "metrics": building_metrics}}
        for layer in catalog.unit_layers:
            logger.info(f"viz: loading unit layer {layer.name}")
            metrics = _build_units(con, catalog, layer, building_metrics)
            layers[layer.name] = {"label": layer.label, "table": unit_table(layer.name), "metrics": metrics}

        parks = None
        if catalog.parks:
            logger.info("viz: loading parks")
            parks = _build_parks(con, catalog)

        trees = None
        if include_trees and catalog.trees:
            logger.info(f"viz: loading trees from {len(catalog.trees.paths)} file(s)")
            trees = _build_trees(con, catalog)

        logger.info("viz: computing metric statistics")
        con.execute("CREATE TABLE metric_stats (layer VARCHAR, metric VARCHAR, stats JSON)")
        for name, info in layers.items():
            for m in info["metrics"]:
                con.execute("INSERT INTO metric_stats VALUES (?, ?, ?)", [name, m, json.dumps(metric_stats(con, info["table"], m))])

        # per layer: Merge's unit table reuses building metric names (tree_count_50m, ...)
        module_of = {"buildings": {m: s.module for s in catalog.building_sources for m in s.columns.values()}}
        for layer in catalog.unit_layers:
            module_of[layer.name] = {}
            for s in layer.sources:  # the first source of a metric (Merge's table) names its module
                for m in s.columns.values():
                    module_of[layer.name].setdefault(m, s.module)
        meta = {
            "fingerprint": fingerprint(catalog, include_trees),
            "study_area_name": catalog.study_area_name,
            "bounds": _lonlat_bounds(con),
            "layers": {
                name: info | {"metrics": [_metric_meta(m, module_of[name].get(m, ""), catalog, name, info["metrics"]) for m in info["metrics"]]}
                for name, info in layers.items()
            },
            "trees": trees,
            "parks": parks,
        }
        con.execute("CREATE TABLE meta (key VARCHAR PRIMARY KEY, value JSON)")
        con.executemany("INSERT INTO meta VALUES (?, ?)", [[k, json.dumps(v)] for k, v in meta.items()])
        con.execute("CHECKPOINT")
    finally:
        con.close()
    os.replace(tmp, path)
    tmp.with_suffix(".duckdb.wal").unlink(missing_ok=True)


def _metric_meta(metric: str, module: str, catalog: Catalog, layer: str, layer_metrics: list[str]) -> dict:
    meta = asdict(describe(metric, module))
    if layer == "buildings" and catalog.rule_gradients.get(metric) in layer_metrics:
        meta["gradient"] = catalog.rule_gradients[metric]
    return meta


def read_meta(con: duckdb.DuckDBPyConnection) -> dict:
    return {k: json.loads(v) for k, v in con.execute("SELECT key, value FROM meta").fetchall()}


def ensure_store(catalog: Catalog, rebuild: bool = False, include_trees: bool = True) -> Path:
    """Path to an up-to-date viz.duckdb, building it when missing, stale or forced."""
    path = catalog.base_dir / "database" / "viz.duckdb"
    if path.exists() and not rebuild:
        try:
            con = connect(path, read_only=True)
            try:
                current = read_meta(con).get("fingerprint")
            finally:
                con.close()
            if current == fingerprint(catalog, include_trees):
                logger.info(f"viz: using cached store {path}")
                return path
            logger.info("viz: outputs changed since the store was built — rebuilding")
        except (duckdb.Error, KeyError) as e:
            logger.warning(f"viz: unreadable store ({e}) — rebuilding")
    build_store(catalog, path, include_trees)
    logger.info(f"viz: store written to {path}")
    return path
