"""Discover which greenpy outputs exist and what the dashboard can show from them.

Pure filesystem logic: Merge parquets in `<base_dir>/database/` are preferred,
and each module falls back to its per-geo CSVs so the map works before Merge
has run. Nothing here touches DuckDB.
"""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config.schema import GreenPyConfig
from ..dggs import SYSTEM_NAMES
from ..merge import RULE_MAX_DISTANCE, RULE_MIN_CANOPY, RULE_MIN_TREES, _buffers_in

BUILDING_KEY = "building_id"
RULE_FLAGS = ("meets_3", "meets_30", "meets_300", "meets_3_30_300")
# columns that are bookkeeping, not metrics
_SKIP_COLUMNS = {"total_pixels", "tree_pixels", "area", "geometry", "closest_park_access_id", "closest_park_site_id"}


@dataclass(frozen=True)
class Metric:
    """One mappable column. `better` says which end of the ramp is green."""

    name: str
    label: str
    module: str
    kind: str  # count, percent, distance, boolean, index, value
    threshold: float | None = None
    better: str | None = None  # "high", "low" or None


@dataclass
class Source:
    """A table of metrics keyed on one column: a parquet file or a CSV glob."""

    path: str  # file path or glob
    fmt: str  # "csv" or "parquet"
    key: str  # join column in the file
    columns: dict[str, str]  # file column -> metric name
    module: str
    # geo codes the module ran over; rows missing within them mean 0, not unknown
    # (T3's rdd path omits tree-less buildings, Tree_count omits tree-less units)
    zero_fill: list[str] = field(default_factory=list)


@dataclass
class UnitLayer:
    """An aggregation geography: a census level or a DGGS grid."""

    name: str  # code column, e.g. LSOA21CD or h3_9
    label: str
    boundaries: Path
    overlay: Path | None  # building_id -> code lookup, for building averages
    sources: list[Source] = field(default_factory=list)
    merged: bool = False  # metrics come from Merge, which already averages buildings


@dataclass
class TreeSource:
    paths: list[Path]
    height_col: str
    area_col: str


@dataclass
class Catalog:
    study_area_name: str
    crs: str
    base_dir: Path
    buildings: Path
    building_sources: list[Source]
    unit_layers: list[UnitLayer]
    trees: TreeSource | None
    census_levels: list[str] = field(default_factory=list)

    def fingerprint_paths(self) -> list[Path]:
        """Every input file the viz store is built from, for staleness checks."""
        paths = [self.buildings]
        for src in self.building_sources + [s for u in self.unit_layers for s in u.sources]:
            paths += _expand(src.path)
        for u in self.unit_layers:
            paths += [u.boundaries] + ([u.overlay] if u.overlay else [])
        if self.trees:
            paths += self.trees.paths
        return sorted(set(paths))


def describe(name: str, module: str = "") -> Metric:
    """Metric metadata inferred from its column name (rule thresholds, which end is good).

    `mean_<metric>` (a building metric averaged per unit) inherits its metric's
    kind and threshold.
    """
    if name.startswith("mean_"):
        m = describe(name[len("mean_"):], module)
        return replace(m, name=name, label="Mean " + m.label[0].lower() + m.label[1:])
    base = name
    label = name.replace("_", " ")
    if base in RULE_FLAGS:
        return Metric(name, _rule_label(base), module or "Rule", "boolean", better="high")
    if m := re.fullmatch(r"pct_(meets_\w+)", base):
        criterion = _rule_label(m.group(1)).removeprefix("Meets ")
        return Metric(name, f"% of buildings meeting {criterion}", module or "Rule", "percent", better="high")
    if m := re.fullmatch(r"tree_count_(\d+)m", base):
        return Metric(name, f"Trees within {m.group(1)} m", module or "T3", "count", RULE_MIN_TREES, "high")
    if base in ("distance_euclidean", "park_distance_euclidean"):
        return Metric(name, "Park distance, straight line (m)", module or "T300", "distance", RULE_MAX_DISTANCE, "low")
    if base in ("distance_manhattan", "park_distance_manhattan"):
        return Metric(name, "Park distance, road network (m)", module or "T300", "distance", RULE_MAX_DISTANCE, "low")
    if base in ("distance_water", "water_distance"):
        return Metric(name, "Water distance (m)", module or "T300", "distance", better="low")
    if base == "canopy_cover":
        return Metric(name, "Canopy cover (%)", module or "T30", "percent", RULE_MIN_CANOPY, "high")
    if m := re.fullmatch(r"building_canopy_cover_(\d+)m", base):
        return Metric(name, f"Canopy cover within {m.group(1)} m (%)", module or "T30_buildings", "percent", RULE_MIN_CANOPY, "high")
    if m := re.fullmatch(r"visible_trees_(\d+)m", base):
        return Metric(name, f"Visible trees within {m.group(1)} m", module or "Visibility", "count", better="high")
    if base == "n_buildings":
        return Metric(name, "Buildings in unit", module or "Other", "count")
    if base in ("total_trees", "tree_count"):
        return Metric(name, "Trees in unit", module or "Tree_count", "count", better="high")
    return Metric(name, label, module or "Other", "index" if module == "Spectral" else "value")


def _rule_label(flag: str) -> str:
    return {
        "meets_3": "Meets 3 (trees)",
        "meets_30": "Meets 30 (canopy)",
        "meets_300": "Meets 300 (park)",
        "meets_3_30_300": "Meets 3-30-300",
    }[flag]


def _expand(path: str) -> list[Path]:
    p = Path(path)
    return sorted(p.parent.glob(p.name)) if any(c in path for c in "*?[") else [p]


def _source(db: Path, module_dir: Path, parquet: str, csv_glob: str, key: str, columns: dict[str, str], module: str) -> Source | None:
    """Prefer the Merge parquet, else the module CSVs; None when neither exists."""
    if (db / parquet).exists():
        return Source(str(db / parquet), "parquet", key, columns, module)
    if list(module_dir.glob(csv_glob)):
        return Source(str(module_dir / csv_glob), "csv", key, columns, module)
    return None


def _codes(directory: Path, prefix: str, suffix: str) -> list[str]:
    """Geo codes in `<prefix><code><suffix>` output filenames."""
    return sorted(f.name[len(prefix):-len(suffix)] for f in directory.glob(f"{prefix}*{suffix}"))


def _csv_header(path: str) -> list[str]:
    first = _expand(path)[0]
    with open(first) as f:
        return f.readline().strip().split(",")


def _parquet_columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def _columns(src_path: str, fmt: str) -> list[str]:
    return _parquet_columns(Path(src_path)) if fmt == "parquet" else _csv_header(src_path)


def _building_sources(base: Path, db: Path) -> list[Source]:
    sources: list[Source | None] = []

    buffers = sorted(set(_buffers_in(db, "T3", "parquet")) | set(_buffers_in(base / "T3", "T3", "csv")))
    for b in buffers:
        col = f"tree_count_{b}m"
        src = _source(db, base / "T3", f"T3_{b}m.parquet", f"*_{b}m.csv", BUILDING_KEY, {col: col}, "T3")
        if src:
            src.zero_fill = _codes(base / "T3", "T3_", f"_{b}m.csv")
        sources.append(src)

    t300 = _source(db, base / "T300", "T300.parquet", "*.csv", BUILDING_KEY, {}, "T300")
    if t300:
        cols = _columns(t300.path, t300.fmt)
        t300.columns = {c: c for c in ("distance_euclidean", "distance_manhattan") if c in cols}
        sources.append(t300)

    buffers = sorted(set(_buffers_in(db, "T30_buildings", "parquet")) | set(_buffers_in(base / "T30_buildings", "T30_buildings", "csv")))
    for b in buffers:
        sources.append(_source(
            db, base / "T30_buildings", f"T30_buildings_{b}m.parquet", f"*_{b}m.csv",
            BUILDING_KEY, {"canopy_cover": f"building_canopy_cover_{b}m"}, "T30_buildings",
        ))

    buffers = sorted(set(_buffers_in(db, "Visibility", "parquet")) | set(_buffers_in(base / "Visibility", "Visibility", "csv")))
    for b in buffers:
        sources.append(_source(
            db, base / "Visibility", f"Visibility_{b}m.parquet", f"*_{b}m.csv",
            BUILDING_KEY, {"visible_trees": f"visible_trees_{b}m"}, "Visibility",
        ))

    rule = db / "T3_30_300_buildings.parquet"
    if rule.exists():
        cols = _parquet_columns(rule)
        sources.append(Source(str(rule), "parquet", BUILDING_KEY, {c: c for c in RULE_FLAGS if c in cols}, "Rule"))

    buildings_cols = _parquet_columns(db / "buildings.parquet")
    if "distance_water" in buildings_cols:
        sources.append(Source(str(db / "buildings.parquet"), "parquet", BUILDING_KEY, {"distance_water": "distance_water"}, "T300"))

    return [s for s in sources if s and s.columns]


def _unit_sources(base: Path, db: Path) -> list[Source]:
    """Native unit-level outputs; each keys on its first column (the sub-geo level it ran at)."""
    sources = []
    merged = db / "T3_30_300_spectral.parquet"
    if merged.exists():
        cols = _parquet_columns(merged)
        sources.append(Source(str(merged), "parquet", cols[0], {c: c for c in cols[1:] if c not in _SKIP_COLUMNS}, "Merge"))
    for module in ("T30", "Tree_count", "Spectral"):
        src = _source(db, base / module, f"{module}.parquet", "*.csv", "", {}, module)
        if not src:
            continue
        cols = _columns(src.path, src.fmt)
        src.key = cols[0]
        src.columns = {c: c for c in cols[1:] if c not in _SKIP_COLUMNS}
        if module == "Tree_count":
            src.zero_fill = _codes(base / module, "Tree_count_", ".csv")
        sources.append(src)
    return [s for s in sources if s.columns]


def _unit_layers(cfg: GreenPyConfig, db: Path, unit_sources: list[Source]) -> list[UnitLayer]:
    layers = []
    overlay = db / "census_buildings_overlay.parquet"
    for level in cfg.columns.geo_levels:
        layers.append(UnitLayer(level, level, db / "census_boundaries.parquet", overlay if overlay.exists() else None))
    for path in sorted(db.glob("*_boundaries_res*.parquet")):
        m = re.fullmatch(r"(\w+?)_boundaries_res(\d+)\.parquet", path.name)
        if not m or m.group(1) not in SYSTEM_NAMES:
            continue
        system, res = m.group(1), int(m.group(2))
        grid_overlay = db / f"{system}_buildings_overlay_res{res}.parquet"
        layers.append(UnitLayer(
            f"{system}_{res}", f"{system.upper()} res {res}", path, grid_overlay if grid_overlay.exists() else None,
        ))

    by_name = {u.name: u for u in layers}
    for src in unit_sources:
        layer = by_name.get(src.key)
        if layer is None:
            continue
        if src.module == "Merge":
            layer.merged = True
        else:
            # Merge's table already carries these columns at its level
            taken = {m for s in layer.sources for m in s.columns.values()}
            src.columns = {c: m for c, m in src.columns.items() if m not in taken}
        if src.columns:
            layer.sources.append(src)
    return layers


def _tree_source(cfg: GreenPyConfig) -> TreeSource | None:
    if not cfg.data.trees_dir:
        return None
    p = Path(cfg.data.trees_dir)
    paths = [p] if p.is_file() else sorted(p.glob("*.gpkg"))
    if not paths:
        return None
    return TreeSource(paths, cfg.columns.tree_height_col, cfg.columns.tree_area_col)


def build_catalog(cfg: GreenPyConfig) -> Catalog:
    """Inspect `<output.base_dir>` and list every layer and metric that can be mapped."""
    base = Path(cfg.output.base_dir)
    db = base / "database"
    buildings = db / "buildings.parquet"
    if not buildings.exists():
        raise FileNotFoundError(
            f"{buildings} not found — run at least one greenpy module (e.g. `greenpy run -p T3`) first"
        )
    unit_sources = _unit_sources(base, db)
    return Catalog(
        study_area_name=cfg.study_area_name,
        crs=cfg.crs,
        base_dir=base,
        buildings=buildings,
        building_sources=_building_sources(base, db),
        unit_layers=_unit_layers(cfg, db, unit_sources),
        trees=_tree_source(cfg),
        census_levels=list(cfg.columns.geo_levels),
    )
