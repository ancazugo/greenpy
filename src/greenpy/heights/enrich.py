"""The Heights process: coalesce the heights chain into one height per building.

Each source's heights are cached per source (database/heights/<label>_<key>.parquet)
so reordering the chain, or evaluating one source on its own, never repeats a
download. The coalesced table is database/building_heights_<key>.parquet,
keyed by the chain and the buildings cache it was built from, so changing
either builds a new one instead of reusing a stale file.
"""

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from loguru import logger

from ..config.schema import GreenPyConfig
from . import get_source
from .base import HeightContext

OUTPUT_COLUMNS = ["building_id", "building_height", "height_source", "height_quality", "height_res_m"]


def _db_dir(cfg: GreenPyConfig) -> Path:
    return Path(cfg.output.base_dir) / "database"


def buildings_fingerprint(cfg: GreenPyConfig) -> str:
    """Identity of the buildings cache: size and modification time of database/buildings.parquet."""
    st = (_db_dir(cfg) / "buildings.parquet").stat()
    return f"{st.st_size}-{st.st_mtime_ns}"


def _key(payload) -> str:
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def source_cache_path(cfg: GreenPyConfig, source) -> Path:
    return _db_dir(cfg) / "heights" / f"{source.label}_{_key([source.cache_key(), buildings_fingerprint(cfg)])}.parquet"


def heights_cache_path(cfg: GreenPyConfig) -> Path:
    h = cfg.heights
    payload = {
        "sources": [[s.source, s.label, s.options] for s in h.sources],
        "default_height": h.default_height, "storey_height": h.storey_height, "min_overlap": h.min_overlap,
        "min_height": h.min_height, "max_height": h.max_height, "buildings": buildings_fingerprint(cfg),
    }
    return _db_dir(cfg) / f"building_heights_{_key(payload)}.parquet"


def raw_cache_dir(cfg: GreenPyConfig, source_name: str) -> Path:
    """Shared download cache for a remote source: <chm cache dir>/heights/<source>."""
    from ..optional.tree_segmentation import cache_dir_for
    return cache_dir_for(cfg) / "heights" / source_name


def source_heights(cfg: GreenPyConfig, spec, buildings: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame,
                   overwrite: bool = False) -> pd.DataFrame:
    """One source's heights for every building, from its cache or computed (and cached)."""
    source = get_source(spec, cfg)
    path = source_cache_path(cfg, source)
    if path.exists() and not overwrite:
        logger.info(f"Heights: {source.label} — cached {path.name}")
        return pd.read_parquet(path)
    ctx = HeightContext(cfg=cfg, cache_dir=raw_cache_dir(cfg, spec.source), boundaries=boundaries)
    ctx.cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Heights: computing {source.label} ({source.kind}) for {len(buildings)} buildings")
    df = source.heights(buildings[["building_id", *_extra_cols(buildings), "geometry"]], ctx)
    df = df.assign(building_id=df["building_id"].astype(str))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _extra_cols(buildings: gpd.GeoDataFrame) -> list[str]:
    """Footprint attributes the native source reads."""
    return [c for c in buildings.columns if c not in ("building_id", "geometry")]


def coalesce(building_ids: pd.Series, results: list[tuple[str, pd.DataFrame]], cfg: GreenPyConfig) -> pd.DataFrame:
    """First valid height per building in chain order; the rest get heights.default_height.

    A height is valid when it lies within [min_height, max_height]. Returns
    OUTPUT_COLUMNS with height_source naming the source (or its row label),
    or "default".
    """
    h = cfg.heights
    ids = building_ids.astype(str).reset_index(drop=True)
    n = len(ids)
    height = np.full(n, np.nan)
    source = np.full(n, None, dtype=object)
    quality = np.zeros(n)
    res = np.full(n, np.nan)
    for label, df in results:
        df = df.drop_duplicates("building_id").set_index("building_id").reindex(ids)
        v = df["height"].to_numpy(dtype=float)
        take = np.isnan(height) & np.isfinite(v) & (v >= h.min_height) & (v <= h.max_height)
        height[take] = v[take]
        row_label = df["label"].to_numpy(dtype=object) if "label" in df.columns else np.full(n, None, dtype=object)
        source[take] = np.where(pd.isna(row_label[take]), label, row_label[take])
        quality[take] = df["quality"].to_numpy(dtype=float)[take]
        res[take] = df["res_m"].to_numpy(dtype=float)[take]
    missing = np.isnan(height)
    height[missing] = h.default_height
    source[missing] = "default"
    return pd.DataFrame({
        "building_id": ids,
        "building_height": height.astype(np.float32),
        "height_source": source.astype(str),
        "height_quality": quality.astype(np.float32),
        "height_res_m": res.astype(np.float32),
    })


def coverage_table(heights: pd.DataFrame) -> pd.DataFrame:
    """Buildings, share and median height per height_source."""
    g = heights.groupby("height_source")["building_height"]
    t = pd.DataFrame({"buildings": g.size(), "median_height": g.median().round(1)})
    t["share_pct"] = (100 * t["buildings"] / len(heights)).round(2)
    return t.sort_values("buildings", ascending=False)


def build_building_heights(cfg: GreenPyConfig, overwrite: bool = False) -> Path:
    """Build (or reuse) database/building_heights_<key>.parquet for the configured chain."""
    db_dir = _db_dir(cfg)
    out = heights_cache_path(cfg)
    if out.exists() and not overwrite:
        logger.info(f"Heights: using {out.name}")
        return out

    buildings = gpd.read_parquet(db_dir / "buildings.parquet")
    buildings["building_id"] = buildings["building_id"].astype(str)
    boundaries = gpd.read_parquet(db_dir / "census_boundaries.parquet")
    results = [
        (spec.label, source_heights(cfg, spec, buildings, boundaries, overwrite=overwrite))
        for spec in cfg.heights.sources
    ]
    heights = coalesce(buildings["building_id"], results, cfg)

    meta = {
        "sources": [{"source": s.source, "label": s.label, "options": s.options} for s in cfg.heights.sources],
        "default_height": cfg.heights.default_height, "storey_height": cfg.heights.storey_height,
        "min_height": cfg.heights.min_height, "max_height": cfg.heights.max_height,
    }
    table = pa.Table.from_pandas(heights, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), b"greenpy_heights": json.dumps(meta).encode()})
    tmp = out.with_suffix(".parquet.tmp")
    pq.write_table(table, tmp)
    tmp.replace(out)

    table_txt = coverage_table(heights).to_string()
    logger.info(f"Heights: {len(heights)} buildings -> {out.name}\n{table_txt}")
    n_default = int((heights["height_source"] == "default").sum())
    if n_default:
        logger.warning(
            f"Heights: {n_default}/{len(heights)} buildings ({n_default / len(heights):.1%}) have no height from "
            f"any source and use the default {cfg.heights.default_height} m"
        )
    return out


def load_building_heights(cfg: GreenPyConfig, build: bool = True) -> pd.DataFrame:
    """The coalesced heights table for the configured chain, built first when missing and build is True."""
    path = heights_cache_path(cfg)
    if not path.exists():
        if not build:
            raise FileNotFoundError(f"{path} not found — run `greenpy run -p Heights` first")
        build_building_heights(cfg)
    return pd.read_parquet(path)
