#!/usr/bin/env python3
"""Compare building-height sources against reference heights (e.g. Verisk in Cambridge).

Reads the per-source caches written by `greenpy run -p Heights`
(<base_dir>/database/heights/<label>_<key>.parquet, the newest per label) and
the reference column of database/buildings.parquet, and reports per source:
coverage (share of reference buildings the source gives a valid height),
MAE, bias (source - reference), RMSE and Spearman correlation — overall, by
footprint area class and, with --levels_col, by floor count. A floor count x
storey height estimate is evaluated too when --levels_col is given.

    python scripts/evaluate_heights.py -c config.yaml --reference_col building_height \
        --levels_col premise_floor_count --out heights_eval.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from greenpy.config.loader import load_config

AREA_BINS = [0, 50, 150, 500, np.inf]
AREA_LABELS = ["<50 m2", "50-150 m2", "150-500 m2", ">500 m2"]


def latest_source_files(heights_dir: Path) -> dict[str, Path]:
    """Newest per-source cache per label."""
    files: dict[str, Path] = {}
    for p in sorted(heights_dir.glob("*.parquet"), key=lambda p: p.stat().st_mtime):
        files[p.stem.rsplit("_", 1)[0]] = p
    return files


def metrics(ref: pd.Series, est: pd.Series) -> dict:
    ok = ref.notna() & est.notna()
    d = est[ok] - ref[ok]
    return {
        "n_ref": int(ref.notna().sum()),
        "coverage_pct": round(100 * ok.sum() / max(1, ref.notna().sum()), 2),
        "mae": round(float(d.abs().mean()), 2) if ok.any() else np.nan,
        "bias": round(float(d.mean()), 2) if ok.any() else np.nan,
        "rmse": round(float(np.sqrt((d ** 2).mean())), 2) if ok.any() else np.nan,
        "spearman": round(float(ref[ok].corr(est[ok], method="spearman")), 3) if ok.sum() > 2 else np.nan,
        "median_ref": round(float(ref[ok].median()), 2) if ok.any() else np.nan,
        "median_est": round(float(est[ok].median()), 2) if ok.any() else np.nan,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--reference_col", default="building_height", help="Reference height column in buildings.parquet")
    ap.add_argument("--levels_col", default=None, help="Floor-count column in buildings.parquet (optional)")
    ap.add_argument("--exclude", nargs="*", default=["native"], help="Source labels to skip (e.g. the reference itself)")
    ap.add_argument("--out", default=None, help="Write the table as CSV")
    args = ap.parse_args()

    cfg = load_config(args.config)
    db = Path(cfg.output.base_dir) / "database"
    h = cfg.heights
    b = gpd.read_parquet(db / "buildings.parquet")
    b["building_id"] = b["building_id"].astype(str)
    ref = pd.to_numeric(b[args.reference_col], errors="coerce")
    ref = ref.where((ref >= h.min_height) & (ref <= h.max_height))
    base = pd.DataFrame({"building_id": b["building_id"], "ref": ref.values,
                         "area_class": pd.cut(b.geometry.area, AREA_BINS, labels=AREA_LABELS).values})
    if args.levels_col:
        levels = pd.to_numeric(b[args.levels_col], errors="coerce")
        base["floors"] = pd.cut(levels, [0, 1, 2, 3, 5, np.inf], labels=["1", "2", "3", "4-5", "6+"]).values
        base["est_levels_x_storey"] = (levels * h.storey_height).where(levels > 0).values

    estimates = {}
    for label, path in latest_source_files(db / "heights").items():
        if label in args.exclude:
            continue
        df = pd.read_parquet(path).drop_duplicates("building_id").set_index("building_id")
        v = df["height"].reindex(base["building_id"]).to_numpy(dtype=float)
        estimates[label] = np.where((v >= h.min_height) & (v <= h.max_height), v, np.nan)
    if args.levels_col:
        estimates[f"{args.levels_col} x {h.storey_height:g} m"] = base["est_levels_x_storey"].to_numpy()

    rows = []
    for label, est in estimates.items():
        est = pd.Series(est)
        rows.append({"source": label, "group": "all", **metrics(base["ref"], est)})
        for col in ["area_class"] + (["floors"] if args.levels_col else []):
            for g, idx in base.groupby(col, observed=True).groups.items():
                rows.append({"source": label, "group": f"{col}={g}", **metrics(base["ref"].loc[idx], est.loc[idx])})
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_rows", 500):
        print(table.to_string(index=False))
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
