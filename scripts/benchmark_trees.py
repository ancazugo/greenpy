"""Benchmark greenpy tree segmentation against the legacy lidR VOM trees.

For each VOM tile with a legacy VOM_trees_<tile>_<year>.gpkg (chm_processing.R
output), runs the legacy_vom preset as one block (so the window cap is the
tile p95, as in lidR) and reports:

- counts: ours vs lidR crowns
- crown match: share of our crowns whose treetop falls in a lidR crown that
  no other of our treetops claims (one-to-one), both ways
- pixel agreement: share of canopy pixels labelled by both whose label pair
  is the dominant pairing of our crown
- wall time, and the blocked/parallel run's agreement with the single block

Usage:
    python scripts/benchmark_trees.py --tiles TL4000 TL4005 --workers 1 8 32
"""

import argparse
import re
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize

from greenpy.optional.tree_segmentation import get_params, segment_raster

VOM_DIR = Path("/maps/acz25/phd-thesis-data/input/Defra/VOM/unzipped_tiles")
LIDR_DIR = Path("/maps/acz25/phd-thesis-data/output/3-30-300/VOM_Trees")


def find_tile(tile: str) -> tuple[Path, Path]:
    for lidr in sorted(LIDR_DIR.glob(f"VOM_trees_{tile}_*.gpkg"), reverse=True):
        year = re.search(r"_(\d{4})\.gpkg$", lidr.name).group(1)
        chm = sorted((VOM_DIR / year).glob(f"VOM_{tile}_*.tif"))
        chm = [c for c in chm if "VOM_HS_" not in c.name]
        if chm:
            return chm[0], lidr
    raise FileNotFoundError(f"No VOM tile + lidR output pair for {tile}")


def compare(ours: gpd.GeoDataFrame, lidr: gpd.GeoDataFrame, chm_path: Path) -> dict:
    with rasterio.open(chm_path) as src:
        shape, transform = (src.height, src.width), src.transform
    lidr = lidr.reset_index(drop=True)
    lab_l = rasterize(((g, i + 1) for i, g in enumerate(lidr.geometry)), out_shape=shape, transform=transform, dtype="int32")
    lab_o = rasterize(((g, i + 1) for i, g in enumerate(ours.geometry)), out_shape=shape, transform=transform, dtype="int32")

    rows, cols = rasterio.transform.rowcol(transform, ours["top_x"].values, ours["top_y"].values)
    hit = lab_l[np.asarray(rows), np.asarray(cols)]
    hits = pd.Series(hit[hit > 0])
    one_to_one = hits.map(hits.value_counts()) == 1

    both = (lab_l > 0) & (lab_o > 0)
    pairs = pd.DataFrame({"o": lab_o[both], "l": lab_l[both]})
    dominant = pairs.groupby("o")["l"].agg(lambda s: s.value_counts().iloc[0]).sum()
    return {
        "n_ours": len(ours),
        "n_lidr": len(lidr),
        "count_diff_pct": round(100 * (len(ours) - len(lidr)) / max(1, len(lidr)), 2),
        "match_ours": round(one_to_one.sum() / max(1, len(ours)), 4),
        "match_lidr": round(one_to_one.sum() / max(1, len(lidr)), 4),
        "pixel_agreement": round(dominant / max(1, both.sum()), 4),
        "canopy_px_ours": int((lab_o > 0).sum()),
        "canopy_px_lidr": int((lab_l > 0).sum()),
    }


def same_trees(a: gpd.GeoDataFrame, b: gpd.GeoDataFrame) -> float:
    """Share of treetops (rounded coordinates + height) present in both runs."""
    key = lambda g: set(zip(g["top_x"].round(2), g["top_y"].round(2), g["height"].round(3), g["area"].round(2)))
    ka, kb = key(a), key(b)
    return len(ka & kb) / max(1, len(ka | kb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=["TL4000"])
    ap.add_argument("--workers", nargs="*", type=int, default=[], help="Also run blocked with these worker counts")
    ap.add_argument("--block_size", type=int, default=1024)
    ap.add_argument("--out", type=Path, default=Path("/scratch/acz25/greenpy_cache/benchmark_trees.csv"))
    args = ap.parse_args()

    p = get_params("legacy_vom")
    rows = []
    for tile in args.tiles:
        chm, lidr_path = find_tile(tile)
        lidr = gpd.read_file(lidr_path)
        t = time.time()
        ours = segment_raster(chm, p, block_size=10**6)
        single_s = time.time() - t
        row = {"tile": tile, "single_block_s": round(single_s, 1), **compare(ours, lidr, chm)}
        with rasterio.open(chm) as src:
            # blocked runs share one cap; pass the tile p95 so they are comparable to the single block
            from greenpy.optional.tree_segmentation import smooth_chm, window_cap
            cap = window_cap(smooth_chm(src.read(1).astype(np.float32), 1, 1, p), p)
        for w in args.workers:
            t = time.time()
            blocked = segment_raster(chm, p, block_size=args.block_size, n_workers=w, ws_cap=cap)
            row[f"blocked_w{w}_s"] = round(time.time() - t, 1)
            row[f"blocked_w{w}_same"] = round(same_trees(ours, blocked), 4)
        print(row, flush=True)
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
