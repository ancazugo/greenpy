"""Evaluate and tune tree segmentation parameters against independent reference trees.

Datasets (cached under --cache):
- london: 1 km chips over the London borough tree inventory (council trees,
  2021), CHM from Defra VOM tiles ("vom": latest survey wins; "vommax":
  per-pixel maximum over survey years) or the Meta/WRI global CHM ("meta").
  Recall is scored on surveyed trees visible in the CHM (canopy >= 3 m within
  3 m of the trunk — leaf-off LiDAR misses many street trees), with recall on
  all surveyed trees reported too. The inventory omits private trees, so
  precision is measured locally: inside a surveyed tree's crown disc there is
  exactly one real tree, and every further detection there is a split crown.
  Boroughs are split into tune/test sets.
- neon: NeonTreeEvaluation plots (40 x 40 m, 1 m LiDAR CHM, hand-drawn crown
  boxes): a treetop inside a box of >= 10 m2 (the size T3 counts) is a hit;
  detections inside smaller boxes are ignored, others are false positives.

Only detections T3 would count (crown area > 10 m2, height > 3 m) are scored,
so parameters that split crowns into slivers lose recall instead of gaining it.

Usage:
    python scripts/tune_tree_segmentation.py prepare --source vom
    python scripts/tune_tree_segmentation.py baseline --source vom
    python scripts/tune_tree_segmentation.py fit-window
    python scripts/tune_tree_segmentation.py search --source vom --trials 300
"""

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, replace
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.windows import from_bounds
from scipy import ndimage
from shapely.geometry import box

from greenpy.optional.chm_sources import chm_mosaic
from greenpy.optional.tree_eval import match_points
from greenpy.optional.tree_segmentation import SegmentationParams, get_params, ground_scale, segment_array, window_cap

LONDON_CSV = Path("/maps/acz25/phd-thesis-data/input/tree_surveys/London/Borough_tree_list_2021July.csv")
VOM_DIR = "/maps/acz25/phd-thesis-data/input/Defra/VOM/unzipped_tiles"
VOM_PATTERN = "VOM_[A-Z][A-Z][0-9]*.tif"
NEON_DIR = Path("/scratch/acz25/greenpy_cache/eval/neon/weecology-NeonTreeEvaluation-f5d92da")
BNG = "EPSG:27700"
CHIP = 1000.0       # chip side (m)
PAD = 120.0         # CHM context around each chip (m)
MIN_RADIUS = 2.0    # match radius floor (m); otherwise half the surveyed spread
# Only detections T3 would count are scored (its defaults: area and height strictly above these)
T3_AREA, T3_HEIGHT = 10.0, 3.0


# --------------------------------------------------------------------------- #
# London data
# --------------------------------------------------------------------------- #

def load_london_refs() -> gpd.GeoDataFrame:
    df = pd.read_csv(LONDON_CSV, usecols=["borough", "height_m", "spread_m", "longitude", "latitude"], dtype=str)
    for c in ["height_m", "spread_m", "longitude", "latitude"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["longitude", "latitude"]).drop_duplicates(subset=["longitude", "latitude"])
    return gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs=4326).to_crs(BNG)


def select_chips(refs: gpd.GeoDataFrame, min_trees: int = 150, per_borough: int = 4) -> gpd.GeoDataFrame:
    """Densest 1 km cells of trees with height >= 3 m and a spread, up to per_borough per borough."""
    good = refs[(refs.height_m >= 3) & (refs.spread_m > 0)]
    cells = good.assign(ix=(good.geometry.x // CHIP).astype(int), iy=(good.geometry.y // CHIP).astype(int))
    counts = cells.groupby(["borough", "ix", "iy"]).size().rename("n").reset_index()
    counts = counts[(counts.n >= min_trees) & (counts.borough != "Out")]
    counts = counts.sort_values("n", ascending=False).groupby("borough").head(per_borough)
    # one borough per cell (keep the one with most trees there)
    counts = counts.sort_values("n", ascending=False).drop_duplicates(["ix", "iy"])
    boroughs = sorted(counts.borough.unique())
    split = {b: ("tune" if i % 2 == 0 else "test") for i, b in enumerate(boroughs)}
    counts["split"] = counts.borough.map(split)
    geoms = [box(x * CHIP, y * CHIP, (x + 1) * CHIP, (y + 1) * CHIP) for x, y in zip(counts.ix, counts.iy)]
    chips = gpd.GeoDataFrame(counts.reset_index(drop=True), geometry=geoms, crs=BNG)
    chips["chip_id"] = [f"{b.replace(' ', '')}_{x}_{y}" for b, x, y in zip(chips.borough, chips.ix, chips.iy)]
    return chips


def cut_chip(layers: list, chip_geom, out: Path) -> None:
    """Save the CHM window covering chip + PAD (in the raster CRS) with its georeferencing."""
    z = None
    for path in layers:
        with rasterio.open(path) as src:
            area = gpd.GeoSeries([chip_geom.buffer(PAD, join_style="mitre")], crs=BNG).to_crs(src.crs)
            win = from_bounds(*area.total_bounds, transform=src.transform).round_offsets().round_lengths()
            layer = src.read(1, window=win, boundless=True, fill_value=np.nan, masked=True).filled(np.nan).astype(np.float32)
            z = layer if z is None else np.fmax(z, layer)
            t, crs = src.window_transform(win), src.crs
    cx, cy = area.geometry.iloc[0].centroid.coords[0]
    np.savez_compressed(out, z=z, transform=np.array(t)[:6], crs=crs.to_string(), scale=ground_scale(crs, cx, cy))


def prepare_london(source: str, cache: Path) -> None:
    out_dir = cache / f"london_{source}"
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = load_london_refs()
    chips = select_chips(refs)
    chips.to_parquet(cache / "london_chips.parquet")
    in_chips = gpd.sjoin(refs, chips[["chip_id", "geometry"]], predicate="within")
    in_chips.drop(columns="index_right").to_parquet(cache / "london_refs.parquet")
    print(f"{len(chips)} chips ({(chips.split == 'tune').sum()} tune), {len(in_chips)} reference trees")
    for row in chips.itertuples():
        out = out_dir / f"{row.chip_id}.npz"
        if out.exists():
            continue
        context = gpd.GeoDataFrame(geometry=[row.geometry.buffer(PAD + 80)], crs=BNG)
        layers = chm_mosaic(context, "meta" if source == "meta" else "chm_tiles", cache / "chm", row.chip_id,
                            chm_tiles_dir=VOM_DIR, chm_pattern=VOM_PATTERN,
                            overlap="max" if source == "vommax" else "latest")
        cut_chip(layers, row.geometry, out)
        print("cut", row.chip_id, flush=True)


# --------------------------------------------------------------------------- #
# NEON data
# --------------------------------------------------------------------------- #

def prepare_neon(cache: Path) -> None:
    rows = []
    for xml in sorted((NEON_DIR / "annotations").glob("*.xml")):
        stem = xml.stem
        rgb, chm = NEON_DIR / "evaluation" / "RGB" / f"{stem}.tif", NEON_DIR / "evaluation" / "CHM" / f"{stem}_CHM.tif"
        if not (rgb.exists() and chm.exists()):
            continue
        with rasterio.open(rgb) as src:
            t = src.transform
        for obj in ET.parse(xml).getroot().iter("object"):
            b = obj.find("bndbox")
            x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            (mx0, my0), (mx1, my1) = t * (x0, y1), t * (x1, y0)
            rows.append({"plot": stem, "xmin": mx0, "ymin": my0, "xmax": mx1, "ymax": my1})
    pd.DataFrame(rows).to_parquet(cache / "neon_boxes.parquet")
    print(f"NEON: {len(set(r['plot'] for r in rows))} plots, {len(rows)} boxes")


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

_DATA: dict = {}


def _load(cache: Path, source: str) -> None:
    """Load chips/plots once per process (inherited by forked workers)."""
    if _DATA.get("key") == (str(cache), source):
        return
    data = {"key": (str(cache), source), "london": [], "neon": []}
    if source in ("vom", "vommax", "meta"):
        refs = gpd.read_parquet(cache / "london_refs.parquet")
        chips = gpd.read_parquet(cache / "london_chips.parquet")
        for row in chips.itertuples():
            f = cache / f"london_{source}" / f"{row.chip_id}.npz"
            if not f.exists():
                continue
            npz = np.load(f)
            r = refs[refs.chip_id == row.chip_id]
            r = r[(r.height_m >= 3) & (r.spread_m > 0)]
            t = rasterio.Affine(*npz["transform"])
            z, scale = npz["z"], float(npz["scale"])
            # visible: canopy >= 3 m within 3 m of the surveyed position
            px = 3.0 / (abs(t.a) * scale)
            zmax = ndimage.maximum_filter(np.nan_to_num(z, nan=0.0), size=2 * int(math.ceil(px)) + 1)
            rx, ry = r.geometry.x.values, r.geometry.y.values
            if str(npz["crs"]) != BNG:
                rx, ry = Transformer.from_crs(BNG, str(npz["crs"]), always_xy=True).transform(rx, ry)
            rows, cols = rasterio.transform.rowcol(t, rx, ry)
            rows = np.clip(np.asarray(rows), 0, z.shape[0] - 1)
            cols = np.clip(np.asarray(cols), 0, z.shape[1] - 1)
            visible = zmax[rows, cols] >= 3
            data["london"].append({
                "visible": visible,
                "chip_id": row.chip_id, "split": row.split, "borough": row.borough,
                "z": npz["z"], "transform": rasterio.Affine(*npz["transform"]), "crs": str(npz["crs"]),
                "scale": float(npz["scale"]), "core": row.geometry.bounds,
                "ref_xy": np.column_stack([r.geometry.x, r.geometry.y]),
                "ref_h": r.height_m.values, "ref_spread": r.spread_m.values,
            })
    if source in ("vom", "vommax", "neon"):
        boxes = pd.read_parquet(cache / "neon_boxes.parquet")
        for plot, g in boxes.groupby("plot"):
            with rasterio.open(NEON_DIR / "evaluation" / "CHM" / f"{plot}_CHM.tif") as src:
                z = src.read(1, masked=True).filled(np.nan).astype(np.float32)
                data["neon"].append({"plot": plot, "z": z, "transform": src.transform,
                                     "gt": g[["xmin", "ymin", "xmax", "ymax"]].values})
    _DATA.clear()
    _DATA.update(data)


def _segment(z, gx, gy, p: SegmentationParams):
    z = np.where(z < -1, np.nan, z)  # VOM/NEON nodata sentinels
    cap = window_cap(np.where(np.isfinite(z), z, np.nan), p) if p.ws_max is None else p.ws_max
    return segment_array(z, gx, gy, p, ws_cap=cap)


def eval_london_chip(i: int, p: SegmentationParams) -> dict:
    c = _DATA["london"][i]
    t = c["transform"]
    gx, gy = abs(t.a) * c["scale"], abs(t.e) * c["scale"]
    seg = _segment(c["z"], gx, gy, p)
    xs, ys = rasterio.transform.xy(t, seg.rows, seg.cols)
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if c["crs"] != BNG and xs.size:
        xs, ys = Transformer.from_crs(c["crs"], BNG, always_xy=True).transform(xs, ys)
    area = np.bincount(seg.labels.ravel(), minlength=seg.rows.size + 1)[1:] * gx * gy
    x0, y0, x1, y1 = c["core"]
    near = (xs >= x0 - 10) & (xs < x1 + 10) & (ys >= y0 - 10) & (ys < y1 + 10)
    near &= (area > T3_AREA) & (seg.heights > T3_HEIGHT)
    det_xy = np.column_stack([xs[near], ys[near]])
    det_h, det_area = seg.heights[near], area[near]

    radius = np.maximum(MIN_RADIUS, c["ref_spread"] / 2)
    di, ri, _ = match_points(det_xy, c["ref_xy"], radius)
    vis = c["visible"]
    # detections inside a visible surveyed tree's crown disc: each disc holds one real tree
    from scipy.spatial import cKDTree
    in_disc = np.zeros(len(det_xy), bool)
    if len(det_xy) and vis.any():
        for h in cKDTree(det_xy).query_ball_point(c["ref_xy"][vis], radius[vis]):
            in_disc[h] = True
    matched_vis = int(vis[ri].sum())
    core = (det_xy[:, 0] >= x0) & (det_xy[:, 0] < x1) & (det_xy[:, 1] >= y0) & (det_xy[:, 1] < y1) if len(det_xy) else np.zeros(0, bool)
    h_err = det_h[di] - c["ref_h"][ri]
    d_err = 2 * np.sqrt(det_area[di] / np.pi) - c["ref_spread"][ri]
    return {"chip_id": c["chip_id"], "split": c["split"], "n_ref": len(c["ref_xy"]), "matched": len(di),
            "n_vis": int(vis.sum()), "matched_vis": matched_vis,
            "n_local": int(in_disc.sum()), "n_det_core": int(core.sum()),
            "h_err_sum": float(h_err.sum()), "h_err_sq": float((h_err**2).sum()),
            "d_err_sum": float(d_err.sum())}


def eval_neon_plot(i: int, p: SegmentationParams) -> dict:
    c = _DATA["neon"][i]
    t = c["transform"]
    seg = _segment(c["z"], abs(t.a), abs(t.e), p)
    area = np.bincount(seg.labels.ravel(), minlength=seg.rows.size + 1)[1:] * abs(t.a * t.e)
    keep = (area > T3_AREA) & (seg.heights > T3_HEIGHT)
    xs, ys = rasterio.transform.xy(t, seg.rows[keep], seg.cols[keep])
    xs, ys = np.asarray(xs, float).reshape(-1), np.asarray(ys, float).reshape(-1)
    gt = c["gt"]
    big = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1]) >= 10
    inside = (xs[:, None] >= gt[None, :, 0]) & (xs[:, None] <= gt[None, :, 2]) & (ys[:, None] >= gt[None, :, 1]) & (ys[:, None] <= gt[None, :, 3])
    # one-to-one: each big box takes the inside treetop nearest its centre
    cx, cy = (gt[:, 0] + gt[:, 2]) / 2, (gt[:, 1] + gt[:, 3]) / 2
    d = np.hypot(xs[:, None] - cx[None, :], ys[:, None] - cy[None, :])
    d = np.where(inside & big[None, :], d, np.inf)
    tp, used_d, used_g = 0, set(), set()
    for k in np.argsort(d, axis=None):
        di, gi = np.unravel_index(k, d.shape)
        if not np.isfinite(d[di, gi]):
            break
        if di in used_d or gi in used_g:
            continue
        used_d.add(di); used_g.add(gi); tp += 1
    unmatched = np.setdiff1d(np.arange(len(xs)), list(used_d))
    ignored = (inside[unmatched][:, ~big].any(axis=1) & ~inside[unmatched][:, big].any(axis=1)).sum() if len(unmatched) else 0
    return {"plot": c["plot"], "n_pred": int(len(xs) - ignored), "n_gt": int(big.sum()), "tp": tp}


def _task(args):
    kind, i, p = args
    return kind, (eval_london_chip(i, p) if kind == "london" else eval_neon_plot(i, p))


def evaluate(p: SegmentationParams, pool=None) -> dict:
    """Micro-averaged scores per dataset/split for one parameter set."""
    tasks = [("london", i, p) for i in range(len(_DATA["london"]))] + [("neon", i, p) for i in range(len(_DATA["neon"]))]
    results = pool.map(_task, tasks, chunksize=1) if pool is not None else map(_task, tasks)
    lon, neon = [], []
    for kind, r in results:
        (lon if kind == "london" else neon).append(r)
    out = {}
    if lon:
        df = pd.DataFrame(lon)
        for split, g in list(df.groupby("split")) + [("all", df)]:
            recall = g.matched_vis.sum() / max(1, g.n_vis.sum())
            lprec = g.matched_vis.sum() / max(1, g.n_local.sum())
            out[f"london_{split}_visible_frac"] = g.n_vis.sum() / max(1, g.n_ref.sum())
            out[f"london_{split}_recall_all"] = g.matched.sum() / max(1, g.n_ref.sum())
            out[f"london_{split}_recall"] = recall
            out[f"london_{split}_local_precision"] = lprec
            out[f"london_{split}_f1"] = 2 * recall * lprec / (recall + lprec) if recall + lprec else 0.0
            out[f"london_{split}_height_bias"] = g.h_err_sum.sum() / max(1, g.matched.sum())
            out[f"london_{split}_height_rmse"] = math.sqrt(g.h_err_sq.sum() / max(1, g.matched.sum()))
            out[f"london_{split}_crown_diam_bias"] = g.d_err_sum.sum() / max(1, g.matched.sum())
            out[f"london_{split}_det_per_km2"] = g.n_det_core.sum() / len(g)
    if neon:
        df = pd.DataFrame(neon)
        prec, rec = df.tp.sum() / max(1, df.n_pred.sum()), df.tp.sum() / max(1, df.n_gt.sum())
        out.update({"neon_precision": prec, "neon_recall": rec,
                    "neon_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0})
    return out


def objective(scores: dict, split: str = "tune") -> float:
    """London F1 on the tuning boroughs, blended with NEON F1 when available (LiDAR only)."""
    f = scores.get(f"london_{split}_f1", np.nan)
    if "neon_f1" in scores and np.isfinite(f):
        return 0.7 * f + 0.3 * scores["neon_f1"]
    return f if np.isfinite(f) else scores.get("neon_f1", np.nan)


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #

def sample_params(rng: np.random.Generator, base: SegmentationParams) -> SegmentationParams:
    smoothing = rng.choice(["none", "median", "gaussian"], p=[0.2, 0.4, 0.4])
    smooth_size = {"none": 1.0, "median": float(rng.choice([3.0, 5.0])), "gaussian": float(rng.uniform(0.4, 2.0))}[smoothing]
    window = rng.choice(["gaussian", "linear"])
    ws_min = float(rng.uniform(2, 8))
    kw = dict(
        smoothing=str(smoothing), smooth_size=smooth_size,
        hmin=float(rng.uniform(2, 5)),
        window=str(window), ws_round=False, ws_min=ws_min,
        ws_max=float(rng.uniform(max(ws_min, 8), 30)),
        th_tree=float(rng.uniform(1, 3)), th_seed=float(rng.uniform(0.3, 0.7)),
        th_cr=float(rng.uniform(0.4, 0.8)), max_cr=float(rng.uniform(5, 15)),
    )
    if window == "gaussian":
        kw.update(ws_base=float(rng.uniform(2, 8)), ws_amp=float(rng.uniform(4, 20)),
                  ws_mu=float(rng.uniform(8, 25)), ws_sigma=float(rng.uniform(3, 12)))
    else:
        kw.update(ws_intercept=float(rng.uniform(0, 6)), ws_slope=float(rng.uniform(0.1, 0.9)))
    return replace(base, **kw)


def perturb(rng: np.random.Generator, p: SegmentationParams, scale: float = 0.15) -> SegmentationParams:
    """Local move around a good parameter set (categoricals fixed)."""
    bounds = {"hmin": (1.5, 6), "ws_min": (1.5, 10), "ws_max": (6, 35), "th_tree": (0.5, 4), "th_seed": (0.2, 0.8),
              "th_cr": (0.3, 0.9), "max_cr": (3, 20), "ws_base": (1, 10), "ws_amp": (2, 24), "ws_mu": (5, 30),
              "ws_sigma": (2, 15), "ws_intercept": (0, 8), "ws_slope": (0.05, 1.2)}
    if p.smoothing == "gaussian":
        bounds["smooth_size"] = (0.3, 3)
    kw = {}
    for k, (lo, hi) in bounds.items():
        v = getattr(p, k)
        if v is None:
            continue
        kw[k] = float(np.clip(v + rng.normal(0, scale * (hi - lo)), lo, hi))
    if kw.get("ws_max", 0) < kw.get("ws_min", 0):
        kw["ws_max"] = kw["ws_min"]
    return replace(p, **kw)


def fit_window(cache: Path) -> None:
    """Crown spread vs height on London trees: median spread per 1 m height bin and a linear fit."""
    refs = load_london_refs()
    r = refs[(refs.height_m >= 2) & (refs.height_m <= 35) & (refs.spread_m > 0)]
    bins = r.groupby(r.height_m.round()).spread_m.agg(["median", "count", lambda s: s.quantile(0.25), lambda s: s.quantile(0.75)])
    bins.columns = ["median", "count", "q25", "q75"]
    print(bins[bins["count"] >= 50].round(2).to_string())
    # median (L1) regression of spread on height
    from scipy.optimize import minimize
    x, y = r.height_m.values, r.spread_m.values
    res = minimize(lambda b: np.abs(y - b[0] - b[1] * x).mean(), x0=[1.0, 0.5], method="Nelder-Mead")
    a, b = res.x
    print(f"median crown spread ~ {a:.2f} + {b:.3f} * height  (n = {len(r)})")
    (cache / "window_fit.json").write_text(json.dumps({"intercept": a, "slope": b}))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _pool(n):
    return concurrent.futures.ProcessPoolExecutor(max_workers=n, mp_context=multiprocessing.get_context("fork"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["prepare", "baseline", "fit-window", "search", "score"])
    ap.add_argument("--source", choices=["vom", "vommax", "meta", "neon"], default="vom")
    ap.add_argument("--cache", type=Path, default=Path("/scratch/acz25/greenpy_cache/eval"))
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--refine", type=int, default=100, help="Local perturbation trials around the best sets")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", type=str, help="JSON of SegmentationParams overrides (score)")
    args = ap.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)

    if args.command == "prepare":
        if args.source == "neon":
            prepare_neon(args.cache)
        else:
            prepare_london(args.source, args.cache)
        return
    if args.command == "fit-window":
        fit_window(args.cache)
        return

    _load(args.cache, args.source)
    print(f"Loaded {len(_DATA['london'])} London chips, {len(_DATA['neon'])} NEON plots", flush=True)
    with _pool(args.workers) as pool:
        if args.command in ("baseline", "score"):
            p = get_params("legacy_vom", **json.loads(args.params)) if args.params else get_params("legacy_vom")
            t = time.time()
            scores = evaluate(p, pool)
            print(json.dumps({k: round(v, 4) for k, v in scores.items()}, indent=1))
            print(f"objective(tune) = {objective(scores):.4f}, objective(test) = {objective(scores, 'test'):.4f}  [{time.time() - t:.1f}s]")
            return

        rng = np.random.default_rng(args.seed)
        base = get_params("legacy_vom")
        log = args.cache / f"search_{args.source}_seed{args.seed}.csv"
        rows = []

        def run(p, phase):
            s = evaluate(p, pool)
            row = {"phase": phase, "objective": objective(s), **s, "params": json.dumps(asdict(p))}
            rows.append(row)
            pd.DataFrame(rows).to_csv(log, index=False)
            return row

        best = run(base, "legacy")
        print(f"legacy objective {best['objective']:.4f}", flush=True)
        for i in range(args.trials):
            row = run(sample_params(rng, base), "random")
            if row["objective"] > best["objective"]:
                best = row
                print(f"[{i}] random best {best['objective']:.4f}", flush=True)
        top = sorted(rows, key=lambda r: -r["objective"])[:5]
        for i in range(args.refine):
            parent = top[i % len(top)]
            p = SegmentationParams(**json.loads(parent["params"]))
            row = run(perturb(rng, p, scale=0.1 if i > args.refine // 2 else 0.2), "refine")
            if row["objective"] > best["objective"]:
                best = row
                print(f"[{i}] refine best {best['objective']:.4f}", flush=True)
            top = sorted(rows, key=lambda r: -r["objective"])[:5]
        print("BEST", json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in best.items()}, indent=1))
        print(f"Log: {log}")


if __name__ == "__main__":
    main()
