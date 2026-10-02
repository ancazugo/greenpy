"""Score Trees outputs against a point tree survey (e.g. a council inventory).

Only detections T3 would count are scored (crown area > 10 m2, height > 3 m).
Survey trees are matched one-to-one to treetops within --radius metres.
Recall is reported on all surveyed trees and, when --chm is given, on those
visible in the CHM (canopy >= 3 m within 3 m). Inventories usually omit
private trees, so precision is local: detections within --radius of a visible
surveyed tree that are not its match are split crowns.

Usage:
    python scripts/evaluate_trees_survey.py --survey tree-data-2024-11.csv --x Ox --y Oy --crs EPSG:27700 \\
        --query "Tree_type == 'TT'" --boundary boundaries.parquet \\
        --run legacy=trees_legacy.parquet:chm_latest.vrt --run tuned=trees_tuned.parquet:chm_L0.vrt,chm_L1.vrt
"""

import argparse
import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from scipy import ndimage
from scipy.spatial import cKDTree

from greenpy.optional.tree_eval import match_points
from greenpy.optional.tree_segmentation import _read_layers, ground_scale

T3_AREA, T3_HEIGHT = 10.0, 3.0


def visible(points: gpd.GeoDataFrame, layers: list[str], reach: float = 3.0, min_height: float = 3.0) -> np.ndarray:
    """True where the CHM has canopy >= min_height within `reach` metres of the point."""
    with rasterio.open(layers[0]) as src:
        crs, transform = src.crs, src.transform
        pts = points.to_crs(crs)
        minx, miny, maxx, maxy = pts.total_bounds
        pad = 20 * abs(transform.a)
        win = from_bounds(minx - pad, miny - pad, maxx + pad, maxy + pad, transform=transform).round_offsets().round_lengths()
        win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        wt = src.window_transform(win)
        cx, cy = pts.union_all().centroid.coords[0]
        px = reach / (abs(transform.a) * ground_scale(crs, cx, cy))
    z = np.nan_to_num(_read_layers(layers, win), nan=0.0)
    zmax = ndimage.maximum_filter(z, size=2 * int(math.ceil(px)) + 1)
    rows, cols = rasterio.transform.rowcol(wt, pts.geometry.x.values, pts.geometry.y.values)
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < z.shape[0]) & (cols >= 0) & (cols < z.shape[1])
    out = np.zeros(len(pts), bool)
    out[inside] = zmax[rows[inside], cols[inside]] >= min_height
    return out


def score(trees: gpd.GeoDataFrame, ref_xy: np.ndarray, vis: np.ndarray | None, radius: float) -> dict:
    t = trees[(trees["area"] > T3_AREA) & (trees["height"] > T3_HEIGHT)]
    det_xy = np.column_stack([t["top_x"], t["top_y"]])
    di, ri, dist = match_points(det_xy, ref_xy, radius)
    out = {"T3_trees": len(t), "survey": len(ref_xy), "recall_all": len(ri) / len(ref_xy), "median_dist": float(np.median(dist)) if len(dist) else np.nan}
    if vis is not None:
        n_vis = int(vis.sum())
        matched_vis = int(vis[ri].sum())
        in_disc = np.zeros(len(det_xy), bool)
        for h in cKDTree(det_xy).query_ball_point(ref_xy[vis], radius):
            in_disc[h] = True
        recall = matched_vis / max(1, n_vis)
        lprec = matched_vis / max(1, in_disc.sum())
        out.update({"visible_frac": n_vis / len(ref_xy), "recall_visible": recall, "local_precision": lprec,
                    "f1": 2 * recall * lprec / (recall + lprec) if recall + lprec else 0.0})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survey", required=True)
    ap.add_argument("--x", default="x")
    ap.add_argument("--y", default="y")
    ap.add_argument("--crs", default="EPSG:27700")
    ap.add_argument("--query", help="pandas query selecting survey rows, e.g. \"Tree_type == 'TT'\"")
    ap.add_argument("--boundary", help="Vector file; only survey trees inside it are scored")
    ap.add_argument("--radius", type=float, nargs="+", default=[3.0])
    ap.add_argument("--run", action="append", required=True, help="name=trees.parquet[:chm1,chm2...]")
    args = ap.parse_args()

    df = pd.read_csv(args.survey) if args.survey.endswith(".csv") else gpd.read_file(args.survey)
    if args.query:
        df = df.query(args.query)
    survey = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df[args.x], df[args.y]), crs=args.crs)
    rows = []
    for spec in args.run:
        name, rest = spec.split("=", 1)
        trees_path, _, chm = rest.partition(":")
        trees = gpd.read_parquet(trees_path)
        s = survey.to_crs(trees.crs)
        if args.boundary:
            read = gpd.read_parquet if args.boundary.endswith(".parquet") else gpd.read_file
            s = s[s.within(read(args.boundary).to_crs(trees.crs).union_all())]
        ref_xy = np.column_stack([s.geometry.x, s.geometry.y])
        vis = visible(s, chm.split(",")) if chm else None
        for r in args.radius:
            rows.append({"run": name, "radius": r, **score(trees, ref_xy, vis, r)})
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
