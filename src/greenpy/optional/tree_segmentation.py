"""
Individual tree detection and crown segmentation from a canopy height model (optional module).

A numpy/scipy port of the lidR pipeline behind the Defra VOM trees (median
smoothing -> variable-window local maximum filter -> Dalponte 2016 region
growing), restructured so the cost follows canopy pixels rather than window
area: treetops are only tested at 3x3 local-maximum candidates, and crowns
grow from an active frontier instead of full-image sweeps.

Large rasters are processed in blocks with a halo. A tree belongs to the block
whose core holds its treetop, so blocks run in parallel without seams or
duplicates. Lengths in SegmentationParams are ground metres, converted to
pixels per block (pixels of a Mercator CRS such as EPSG:3857 are scaled by
cos(latitude)).
"""

import math
import concurrent.futures
import multiprocessing
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import shapely
from rasterio.features import shapes
from rasterio.windows import Window, from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import CRS, Transformer
from loguru import logger

_EPS = 1e-8


@dataclass(frozen=True)
class SegmentationParams:
    # Smoothing before detection: "median" (kernel width), "gaussian" (sigma) or "none"; size in metres
    smoothing: str = "median"
    smooth_size: float = 5.0
    # Treetops lower than this (m) are ignored
    hmin: float = 2.0
    # Local-maximum window diameter (m) as a function of height: "gaussian"
    # (base + amp * exp(-(h - mu)^2 / 2 sigma^2), the legacy lidR function) or
    # "linear" (intercept + slope * h); heights below ws_floor_height count as it
    window: str = "gaussian"
    ws_floor_height: float = 3.0
    ws_base: float = 6.0
    ws_amp: float = 18.0
    ws_mu: float = 18.0
    ws_sigma: float = 7.0
    ws_intercept: float = 3.0
    ws_slope: float = 0.3
    ws_round: bool = True
    ws_min: float = 7.0
    # Fixed cap on the window; None caps at ws_max_frac * p95 height of the processed area
    ws_max: float | None = None
    ws_max_frac: float = 0.7
    # Dalponte 2016 crown growing; max_cr bounds the crown's half-extent from its seed (m)
    th_tree: float = 2.0
    th_seed: float = 0.45
    th_cr: float = 0.55
    max_cr: float = 10.0
    # Detect on the unsmoothed CHM when smoothing leaves no treetops
    retry_unsmoothed: bool = True


# legacy_vom reproduces chm_processing.R (lidR lmf + dalponte2016 on Defra VOM tiles).
# vom was tuned (scripts/tune_tree_segmentation.py) on 1 m LiDAR CHMs against the
# London borough tree inventory and NeonTreeEvaluation, scoring only trees T3
# counts: on held-out boroughs F1 0.48 -> 0.58 and NEON F1 0.32 -> 0.60. A
# fixed 4.6 m window matched every height-dependent window the search found.
# meta was tuned the same way on the Meta/WRI global CHM over London (no NEON):
# held-out F1 0.41 -> 0.48. Meta still splits about one surveyed crown in two
# (local precision ~0.54), so its T3 counts run higher than LiDAR's.
PRESETS: dict[str, SegmentationParams] = {
    "legacy_vom": SegmentationParams(),
    "vom": SegmentationParams(
        smoothing="gaussian", smooth_size=0.6, hmin=3.5,
        window="linear", ws_intercept=4.6, ws_slope=0.0, ws_round=False, ws_min=4.6, ws_max=4.6,
        th_tree=1.2, th_seed=0.2, th_cr=0.4, max_cr=11.0,
    ),
    "meta": SegmentationParams(
        smoothing="none", hmin=3.5,
        window="linear", ws_intercept=0.0, ws_slope=0.47, ws_round=False, ws_min=4.1, ws_max=35.0,
        th_tree=0.5, th_seed=0.33, th_cr=0.35, max_cr=12.5,
    ),
}


def get_params(preset: str = "legacy_vom", **overrides) -> SegmentationParams:
    """Preset parameters with explicit overrides applied (unknown keys raise TypeError)."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown segmentation preset '{preset}', expected one of {list(PRESETS)}")
    return replace(PRESETS[preset], **overrides)


# --------------------------------------------------------------------------- #
# Array algorithms
# --------------------------------------------------------------------------- #

def _odd_pixels(size_m: float, res: float) -> int:
    k = max(1, int(round(size_m / res)))
    return k if k % 2 else k + 1


def smooth_chm(z: np.ndarray, gx: float, gy: float, p: SegmentationParams) -> np.ndarray:
    """Smooth the CHM; nodata (NaN) is treated as ground while filtering and stays NaN."""
    if p.smoothing == "none":
        return z
    nan = ~np.isfinite(z)
    zf = np.where(nan, 0.0, z).astype(np.float32, copy=False)
    if p.smoothing == "median":
        out = ndimage.median_filter(zf, size=(_odd_pixels(p.smooth_size, gy), _odd_pixels(p.smooth_size, gx)), mode="nearest")
    elif p.smoothing == "gaussian":
        out = ndimage.gaussian_filter(zf, sigma=(p.smooth_size / gy, p.smooth_size / gx), mode="nearest")
    else:
        raise ValueError(f"Unknown smoothing '{p.smoothing}'")
    if nan.any():
        out[nan] = np.nan
    return out


def window_cap(z: np.ndarray, p: SegmentationParams) -> float:
    """Upper bound of the window diameter: ws_max, or ws_max_frac * p95 of the (floored) valid heights."""
    if p.ws_max is not None:
        return p.ws_max
    valid = z[np.isfinite(z)]
    if valid.size == 0:
        return p.ws_min
    return p.ws_max_frac * float(np.quantile(np.maximum(valid, p.ws_floor_height), 0.95))


def window_size(h: np.ndarray, p: SegmentationParams, cap: float) -> np.ndarray:
    """Window diameter (m) per treetop height, clamped to [ws_min, cap] (cap wins, as in lidR)."""
    h = np.maximum(np.asarray(h, dtype=float), p.ws_floor_height)
    if p.window == "gaussian":
        ws = p.ws_base + p.ws_amp * np.exp(-((h - p.ws_mu) ** 2) / (2 * p.ws_sigma**2))
    elif p.window == "linear":
        ws = p.ws_intercept + p.ws_slope * h
    else:
        raise ValueError(f"Unknown window function '{p.window}'")
    if p.ws_round:
        ws = np.round(ws)  # half-to-even, like R's round()
    return np.minimum(np.maximum(ws, p.ws_min), cap)


def _disc_offsets(radius: float, gx: float, gy: float) -> tuple[np.ndarray, np.ndarray]:
    """Row/col offsets of cells whose centres lie within radius (inclusive) of the centre cell."""
    kr, kc = int(radius // gy), int(radius // gx)
    di, dj = np.mgrid[-kr:kr + 1, -kc:kc + 1]
    inside = (di * gy) ** 2 + (dj * gx) ** 2 <= radius**2 + _EPS
    return di[inside], dj[inside]


def detect_treetops(
    zs: np.ndarray, gx: float, gy: float, p: SegmentationParams, cap: float
) -> tuple[np.ndarray, np.ndarray]:
    """Variable-window local maxima (lidR lmf, circular). Returns treetop rows, cols in raster order.

    A cell >= hmin is a treetop when no cell within half its window diameter
    is strictly higher. Among equal-height maxima within each other's window
    the first in raster order wins.
    """
    nrow, ncol = zs.shape
    zf = np.where(np.isfinite(zs), zs, -np.inf).astype(np.float32, copy=False)

    # Every treetop is also a maximum over the part of the 3x3 neighbourhood its smallest window covers
    rmin = p.ws_min / 2 if p.ws_min <= cap else cap / 2
    fi, fj = _disc_offsets(min(rmin, math.hypot(gx, gy)), gx, gy)
    footprint = np.zeros((3, 3), bool)
    footprint[fi + 1, fj + 1] = True
    local_max = ndimage.maximum_filter(zf, footprint=footprint, mode="constant", cval=-np.inf)
    cand = np.flatnonzero((zf >= p.hmin) & (zf >= local_max))
    if cand.size == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)

    zc = zf.ravel()[cand]
    radius = window_size(zc, p, cap) / 2

    pad = int(math.ceil(radius.max() / min(gx, gy))) + 1
    zp = np.pad(zf, pad, constant_values=-np.inf)
    width = zp.shape[1]
    rows, cols = np.divmod(cand, ncol)
    pidx = (rows + pad) * width + (cols + pad)
    zp_flat = zp.ravel()

    # Radii that select the same set of cells share one disc
    di_all, dj_all = _disc_offsets(radius.max(), gx, gy)
    d2 = np.unique((di_all * gy) ** 2 + (dj_all * gx) ** 2)
    disc_key = np.searchsorted(d2, radius**2 + _EPS, side="right")

    is_max = np.zeros(cand.size, bool)
    for key in np.unique(disc_key):
        sel = np.flatnonzero(disc_key == key)
        di, dj = _disc_offsets(math.sqrt(d2[key - 1]), gx, gy)
        off = di * width + dj
        step = max(1, 4_000_000 // off.size)
        for s in range(0, sel.size, step):
            chunk = sel[s:s + step]
            is_max[chunk] = zc[chunk] >= zp_flat[pidx[chunk, None] + off[None, :]].max(axis=1)

    keep = np.flatnonzero(is_max)
    keep = keep[_suppress_ties(rows[keep], cols[keep], zc[keep], radius[keep], gx, gy)]
    return rows[keep], cols[keep]


def _suppress_ties(rows, cols, z, radius, gx, gy) -> np.ndarray:
    """Mask of maxima to keep: drop one equal in height to an earlier kept maximum inside its window."""
    keep = np.ones(rows.size, bool)
    if rows.size < 2:
        return keep
    xy = np.column_stack([cols * gx, rows * gy])
    pairs = cKDTree(xy).query_pairs(float(radius.max()) + _EPS, output_type="ndarray")
    if pairs.size == 0:
        return keep
    a, b = np.sort(pairs, axis=1).T  # inputs are in raster order, so a precedes b
    d2 = ((xy[a] - xy[b]) ** 2).sum(axis=1)
    tie = (z[a] == z[b]) & (d2 <= radius[b] ** 2 + _EPS)
    a, b = a[tie], b[tie]
    if a.size == 0:
        return keep
    order = np.lexsort((a, b))
    a, b = a[order], b[order]
    starts = np.flatnonzero(np.r_[True, b[1:] != b[:-1]])
    ends = np.r_[starts[1:], b.size]
    for s, e in zip(starts, ends):
        if keep[a[s:e]].any():
            keep[b[s]] = False
    return keep


def dalponte2016(
    zs: np.ndarray, rows: np.ndarray, cols: np.ndarray, gx: float, gy: float, p: SegmentationParams
) -> np.ndarray:
    """Seeded region growing (Dalponte & Coomes 2016, lidR semantics). Returns int32 labels, 0 = none.

    Seeds are labelled 1..n in input order. Each sweep, a labelled interior
    pixel claims an unlabelled 4-neighbour when it exceeds th_tree,
    th_seed * seed height and th_cr * crown mean height, is at most 1.05 *
    seed height, and lies within max_cr of the seed along both axes. As in
    lidR, every successful claim updates the crown mean and, when several
    crowns claim a pixel in one sweep, the claimer last in raster order wins.
    Only the active frontier is visited, so cost follows canopy size.
    """
    nrow, ncol = zs.shape
    img = np.where(np.isfinite(zs), zs, -np.inf).astype(np.float64).ravel()
    region = np.zeros(nrow * ncol, np.int32)
    n = rows.size
    if n == 0:
        return region.reshape(nrow, ncol)

    seed_flat = rows * ncol + cols
    region[seed_flat] = np.arange(1, n + 1, dtype=np.int32)
    seed_r = np.r_[0, rows]
    seed_c = np.r_[0, cols]
    h_seed = np.r_[0.0, img[seed_flat]]
    sum_h = h_seed.copy()
    npix = np.r_[1.0, np.ones(n)]
    max_r, max_c = p.max_cr / gy, p.max_cr / gx
    offsets = np.array([-ncol, -1, 1, ncol])

    active = seed_flat
    while active.size:
        r, c = np.divmod(active, ncol)
        claimers = active[(r > 0) & (r < nrow - 1) & (c > 0) & (c < ncol - 1)]
        if claimers.size == 0:
            break
        ids = np.broadcast_to(region[claimers][:, None], (claimers.size, 4))
        target = claimers[:, None] + offsets[None, :]
        zt = img[target]
        tr, tc = np.divmod(target, ncol)
        hs = h_seed[ids]
        # every condition but the crown-mean one is fixed for a (crown, pixel) pair
        static_ok = (
            (zt > p.th_tree)
            & (zt > hs * p.th_seed)
            & (zt <= hs * 1.05)
            & (np.abs(seed_r[ids] - tr) < max_r)
            & (np.abs(seed_c[ids] - tc) < max_c)
        )
        ok = static_ok & (zt > (sum_h / npix)[ids] * p.th_cr) & (region[target] == 0)
        if not ok.any():
            break
        t_ok, id_ok = target[ok], ids[ok]
        c_ok = np.broadcast_to(claimers[:, None], target.shape)[ok]
        npix += np.bincount(id_ok, minlength=n + 1)
        sum_h += np.bincount(id_ok, weights=img[t_ok], minlength=n + 1)

        order = np.lexsort((c_ok, t_ok))
        t_sorted = t_ok[order]
        last = np.r_[t_sorted[1:] != t_sorted[:-1], True]
        won = t_sorted[last]
        region[won] = id_ok[order][last]

        # claimers stay active while an unlabelled neighbour could still pass (only the mean test can change)
        still_open = ((region[target] == 0) & static_ok).any(axis=1)
        active = np.concatenate([claimers[still_open], won])
    return region.reshape(nrow, ncol)


@dataclass
class Segmentation:
    labels: np.ndarray       # int32 crown labels (0 = none); label i is treetop i-1
    rows: np.ndarray         # treetop rows
    cols: np.ndarray         # treetop cols
    heights: np.ndarray      # treetop heights (m) on the detection surface


def segment_array(
    z: np.ndarray, gx: float, gy: float, p: SegmentationParams, ws_cap: float | None = None
) -> Segmentation:
    """Smooth, detect treetops and grow crowns on one in-memory CHM array (ground pixel size gx, gy)."""
    zs = smooth_chm(z, gx, gy, p)
    cap = ws_cap if ws_cap is not None else window_cap(zs, p)
    rows, cols = detect_treetops(zs, gx, gy, p, cap)
    detect_surface = zs
    if rows.size == 0 and p.retry_unsmoothed and p.smoothing != "none":
        raw = np.where(np.isfinite(z), z, np.nan).astype(np.float32, copy=False)
        cap = ws_cap if ws_cap is not None else window_cap(raw, p)
        rows, cols = detect_treetops(raw, gx, gy, p, cap)
        detect_surface = raw
    labels = dalponte2016(zs, rows, cols, gx, gy, p)
    return Segmentation(labels, rows, cols, detect_surface[rows, cols].astype(float))


# --------------------------------------------------------------------------- #
# Raster driver
# --------------------------------------------------------------------------- #

def ground_scale(crs: CRS, x: float, y: float) -> float:
    """Ground metres per CRS unit at (x, y): cos(lat) for Mercator, 1 for other metric CRSs."""
    crs = CRS.from_user_input(crs)
    if crs.is_geographic:
        raise ValueError("Tree segmentation needs a projected CHM (metres); reproject geographic rasters first")
    unit = crs.axis_info[0].unit_name.lower() if crs.axis_info else "metre"
    if unit not in ("metre", "meter", "m"):
        raise ValueError(f"CHM CRS units must be metres, got '{unit}'")
    method = (crs.coordinate_operation.method_name if crs.coordinate_operation else "").lower()
    if "mercator" in method and "transverse" not in method:
        lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(x, y)[1]
        return math.cos(math.radians(lat))
    return 1.0


def halo_metres(p: SegmentationParams, cap: float | None) -> float:
    """Context a block needs around its core so core treetops and their crowns match a whole-raster run.

    Smoothing radius + the largest window radius (treetop test) + two crown
    extents (a core crown and the neighbouring crowns competing with it).
    Without a cap the largest window the function can return is assumed.
    """
    if cap is None:
        cap = p.ws_base + p.ws_amp if p.window == "gaussian" else p.ws_intercept + p.ws_slope * 100
    smooth = 0.0 if p.smoothing == "none" else (p.smooth_size / 2 if p.smoothing == "median" else 4 * p.smooth_size)
    return smooth + cap / 2 + 2 * p.max_cr + 2


def _read_layers(paths: list[str], window: Window, out_shape=None) -> np.ndarray:
    """Read one window from each layer (rasters on a shared grid) and take the per-pixel maximum."""
    z = None
    for path in paths:
        with rasterio.open(path) as src:
            layer = src.read(1, window=window, out_shape=out_shape, masked=True).filled(np.nan).astype(np.float32)
        z = layer if z is None else np.fmax(z, layer)
    return z


def _estimate_cap(paths: list[str], window: Window, p: SegmentationParams, max_side: int = 2048) -> float:
    """window_cap over a decimated read, so every block shares one cap regardless of block size."""
    if p.ws_max is not None:
        return p.ws_max
    scale = max(1.0, max(window.width, window.height) / max_side)
    shape = (max(1, int(window.height / scale)), max(1, int(window.width / scale)))
    return window_cap(_read_layers(paths, window, out_shape=shape), p)


def _block_windows(window: Window, block_size: int) -> list[Window]:
    col0, row0 = int(window.col_off), int(window.row_off)
    w, h = int(window.width), int(window.height)
    return [
        Window(col0 + c, row0 + r, min(block_size, w - c), min(block_size, h - r))
        for r in range(0, h, block_size)
        for c in range(0, w, block_size)
    ]


def _process_block(
    paths: list[str], core: Window, p: SegmentationParams, cap: float, geometry: str
) -> pd.DataFrame | None:
    """Segment one block (core + halo) and return the trees whose treetop lies in the core.

    paths are layers on one grid, combined by per-pixel maximum. Columns:
    height, area, top_x, top_y, geometry (WKB, raster CRS) — crown polygon,
    or the crown centroid when geometry == "point".
    """
    with rasterio.Env(GDAL_NUM_THREADS="1"):
        with rasterio.open(paths[0]) as src:
            cx, cy = src.xy(core.row_off + core.height / 2, core.col_off + core.width / 2)
            scale = ground_scale(src.crs, cx, cy)
            gx, gy = abs(src.transform.a) * scale, abs(src.transform.e) * scale
            halo = int(math.ceil(halo_metres(p, cap) / min(gx, gy)))
            row0 = max(0, int(core.row_off) - halo)
            col0 = max(0, int(core.col_off) - halo)
            row1 = min(src.height, int(core.row_off + core.height) + halo)
            col1 = min(src.width, int(core.col_off + core.width) + halo)
            win = Window(col0, row0, col1 - col0, row1 - row0)
            transform = src.window_transform(win)
        z = _read_layers(paths, win)

    if not np.nanmax(z, initial=-np.inf) >= p.hmin:
        return None
    seg = segment_array(z, gx, gy, p, ws_cap=cap)

    r_core, c_core = int(core.row_off) - row0, int(core.col_off) - col0
    owned = (
        (seg.rows >= r_core) & (seg.rows < r_core + core.height)
        & (seg.cols >= c_core) & (seg.cols < c_core + core.width)
    )
    if not owned.any():
        return None
    ids = np.flatnonzero(owned) + 1  # crown labels of owned trees

    labels = seg.labels
    flat = labels.ravel()
    n_lab = seg.rows.size + 1
    count = np.bincount(flat, minlength=n_lab)
    top_x, top_y = rasterio.transform.xy(transform, seg.rows[owned], seg.cols[owned])
    out = pd.DataFrame({
        "height": seg.heights[owned],
        "area": count[ids] * gx * gy,
        "top_x": np.asarray(top_x, float),
        "top_y": np.asarray(top_y, float),
    })

    if geometry == "point":
        rr, cc = np.divmod(np.arange(flat.size), labels.shape[1])
        mean_r = np.bincount(flat, weights=rr, minlength=n_lab)[ids] / count[ids]
        mean_c = np.bincount(flat, weights=cc, minlength=n_lab)[ids] / count[ids]
        x, y = rasterio.transform.xy(transform, mean_r, mean_c)
        out["geometry"] = shapely.to_wkb(shapely.points(np.asarray(x), np.asarray(y)))
    else:
        keep_label = np.zeros(n_lab, bool)
        keep_label[ids] = True
        owned_mask = keep_label[labels]
        parts: dict[int, list] = {}
        for geom, value in shapes(labels, mask=owned_mask, connectivity=4, transform=transform):
            parts.setdefault(int(value), []).append(shapely.geometry.shape(geom))
        out["geometry"] = shapely.to_wkb([
            ps[0] if len(ps) == 1 else shapely.MultiPolygon(ps) for ps in (parts[i] for i in ids)
        ])
    return out


def segment_raster(
    path: str | Path | list,
    params: SegmentationParams | None = None,
    bounds: tuple[float, float, float, float] | None = None,
    block_size: int = 2048,
    n_workers: int = 1,
    geometry: str = "polygon",
    ws_cap: float | None = None,
    dst_crs: str | None = None,
    region=None,
) -> gpd.GeoDataFrame:
    """Segment trees over a CHM raster (or VRT), optionally limited to bounds in the raster CRS.

    `path` may be a list of layers on one pixel grid (see chm_sources.chm_mosaic
    with overlap="max"); they are combined by per-pixel maximum.

    Trees are owned by the block containing their treetop; crowns may extend
    past bounds. Blocks whose core misses `region` (a shapely geometry in the
    raster CRS) are skipped. Returns treeID, height (m), area (m², ground), top_x/top_y
    (treetop) and crown polygons (or crown centroids for geometry="point"),
    in dst_crs when given, else the raster CRS.
    """
    p = params or SegmentationParams()
    if geometry not in ("polygon", "point"):
        raise ValueError("geometry must be 'polygon' or 'point'")
    paths = [str(x) for x in path] if isinstance(path, (list, tuple)) else [str(path)]
    with rasterio.open(paths[0]) as src:
        crs = CRS.from_user_input(src.crs)
        full = Window(0, 0, src.width, src.height)
        window = full
        if bounds is not None:
            window = from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths()
            window = window.intersection(full)
        blocks = _block_windows(window, block_size)
        if region is not None:
            boxes = shapely.box(*np.array([rasterio.windows.bounds(b, src.transform) for b in blocks]).T)
            blocks = [b for b, hit in zip(blocks, shapely.intersects(boxes, region)) if hit]
        if ws_cap is None and p.ws_max is None and len(blocks) > 1:
            ws_cap = _estimate_cap(paths, window, p)

    if ws_cap is None and p.ws_max is not None:
        ws_cap = p.ws_max
    logger.info(f"Segmenting {paths[0]}{f' (+{len(paths) - 1} layers)' if len(paths) > 1 else ''}: {len(blocks)} blocks, {n_workers} workers")

    if not blocks:
        parts = []
    elif n_workers > 1 and len(blocks) > 1:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            parts = list(ex.map(_process_block, [paths] * len(blocks), blocks, [p] * len(blocks),
                                [ws_cap] * len(blocks), [geometry] * len(blocks)))
    else:
        parts = [_process_block(paths, b, p, ws_cap, geometry) for b in blocks]

    parts = [d for d in parts if d is not None]
    if parts:
        df = pd.concat(parts, ignore_index=True)
        gdf = gpd.GeoDataFrame(df.drop(columns="geometry"), geometry=shapely.from_wkb(df["geometry"].values), crs=crs)
    else:
        gdf = gpd.GeoDataFrame(columns=["height", "area", "top_x", "top_y"], geometry=[], crs=crs)
    gdf.insert(0, "treeID", np.arange(1, len(gdf) + 1))

    if dst_crs is not None and not crs.equals(CRS.from_user_input(dst_crs)):
        tx, ty = Transformer.from_crs(crs, dst_crs, always_xy=True).transform(gdf["top_x"].values, gdf["top_y"].values)
        gdf = gdf.to_crs(dst_crs)
        gdf["top_x"], gdf["top_y"] = tx, ty
    return gdf


# --------------------------------------------------------------------------- #
# Pipeline entry point
# --------------------------------------------------------------------------- #

def cache_dir_for(cfg) -> Path:
    """data.chm_cache_dir, else $GREENPY_CACHE_DIR, else <output.base_dir>/database/chm_cache."""
    import os
    d = cfg.data.chm_cache_dir or os.environ.get("GREENPY_CACHE_DIR")
    return Path(d) if d else Path(cfg.output.base_dir) / "database" / "chm_cache"


def chm_source(cfg) -> str:
    return cfg.tree_segmentation.source or ("chm_tiles" if cfg.data.chm_tiles_dir else "meta")


def process_geo_code(
    geo_level: str,
    geo_code: str,
    cfg,
    boundaries_gdf: gpd.GeoDataFrame,
    n_workers: int = 1,
    overwrite: bool = True,
) -> Path | None:
    """Segment the trees whose treetop lies in one geo_code into data.trees_dir/trees_<geo_code>.parquet.

    The CHM (local tiles or Meta, see chm_source) is mosaicked over the
    boundary plus the segmentation halo, so crowns at the edge see their
    neighbours. Columns: treeID, height, area, top_x, top_y and crown
    polygons (or centroids) in cfg.crs — the column names T3 expects by
    default. Returns the output path, or None on error.
    """
    from .chm_sources import chm_mosaic, boundary_bounds_in

    if not cfg.data.trees_dir:
        raise ValueError("The Trees process writes to data.trees_dir; set it to an output directory")
    out_dir = Path(cfg.data.trees_dir)
    if out_dir.suffix:
        raise ValueError(f"data.trees_dir must be a directory for the Trees process, got file path {out_dir}")
    out_path = out_dir / f"trees_{geo_code}.parquet"
    if out_path.exists() and not overwrite:
        return out_path

    ts = cfg.tree_segmentation
    try:
        p = get_params(ts.preset, **ts.params)
        boundary = boundaries_gdf.loc[boundaries_gdf[geo_level] == geo_code].to_crs(cfg.crs).dissolve()
        if boundary.empty:
            raise ValueError(f"{geo_code} not found in {geo_level}")
        cache = cache_dir_for(cfg)
        halo = halo_metres(p, None)
        context = boundary.buffer(halo).to_frame("geometry")

        layers = chm_mosaic(
            context, chm_source(cfg), cache, f"{geo_level}_{geo_code}",
            chm_tiles_dir=cfg.data.chm_tiles_dir, chm_pattern=cfg.data.chm_pattern, overlap=cfg.data.chm_overlap,
        )
        with rasterio.open(layers[0]) as src:
            region = boundary.to_crs(src.crs).geometry.iloc[0]
        trees = segment_raster(
            layers, p, bounds=boundary_bounds_in(layers[0], boundary), block_size=ts.block_size,
            n_workers=n_workers, geometry=ts.geometry, dst_crs=cfg.crs, region=region,
        )
        # a tree belongs to the geo_code containing its treetop
        inside = shapely.intersects_xy(boundary.geometry.iloc[0], trees["top_x"].values, trees["top_y"].values)
        trees = trees.loc[inside].reset_index(drop=True)
        trees["treeID"] = np.arange(1, len(trees) + 1)

        out_dir.mkdir(parents=True, exist_ok=True)
        trees.to_parquet(out_path, index=False)
        logger.info(f"Trees: {geo_code} — {len(trees)} trees -> {out_path}")
        return out_path
    except Exception:
        logger.exception(f"Trees: error processing {geo_code}")
        return None
