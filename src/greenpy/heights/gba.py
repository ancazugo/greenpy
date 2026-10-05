"""GlobalBuildingAtlas (TUM, Zhu et al. 2025): global building polygons with heights.

2.75 billion polygons (from Google, Microsoft, OSM and TUM footprints) with a
height estimated from PlanetScope imagery, published on GEE as one
FeatureCollection per 5x5 degree tile under the sat-io community catalog.
Licensed CC BY-NC 4.0 — non-commercial use only.
"""

import re

import geopandas as gpd

from .base import HeightContext, VectorHeightSource
from .gee import bounds_4326, fetch_features, list_asset_names

ASSET_ROOT = "projects/sat-io/open-datasets/GLOBAL_BUILDING_ATLAS"
# tile names give the west, north, east and south edges: e.g. w075_n05_w070_n00
_TILE_RE = re.compile(r"^([ew])(\d{3})_([ns])(\d{2})_([ew])(\d{3})_([ns])(\d{2})$")


def parse_tile(name: str) -> tuple[float, float, float, float]:
    """(minx, miny, maxx, maxy) in degrees of a GBA tile name."""
    m = _TILE_RE.match(name)
    if not m:
        raise ValueError(f"Not a GlobalBuildingAtlas tile name: {name!r}")
    w = int(m[2]) * (-1 if m[1] == "w" else 1)
    n = int(m[4]) * (-1 if m[3] == "s" else 1)
    e = int(m[6]) * (-1 if m[5] == "w" else 1)
    s = int(m[8]) * (-1 if m[7] == "s" else 1)
    return (w, s, e, n)


def tiles_for_bounds(names: list[str], bounds: tuple) -> list[tuple[str, tuple]]:
    """(name, bounds ∩ tile) for the tiles overlapping bounds (minx, miny, maxx, maxy)."""
    minx, miny, maxx, maxy = bounds
    out = []
    for name in names:
        if not _TILE_RE.match(name):
            continue
        tx0, ty0, tx1, ty1 = parse_tile(name)
        ix0, iy0, ix1, iy1 = max(minx, tx0), max(miny, ty0), min(maxx, tx1), min(maxy, ty1)
        if ix0 < ix1 and iy0 < iy1:
            out.append((name, (ix0, iy0, ix1, iy1)))
    return out


def list_tiles(ctx: HeightContext, project: str | None) -> list[str]:
    """Tile names under ASSET_ROOT (cached in the source cache dir)."""
    return list_asset_names(ASSET_ROOT, ctx.cache_dir / "tiles.json", project)


class GBAHeights(VectorHeightSource):
    license = "CC BY-NC 4.0 (GlobalBuildingAtlas, TUM) — non-commercial use only"

    def fetch(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> gpd.GeoDataFrame:
        bounds = bounds_4326(buildings)
        tiles = tiles_for_bounds(list_tiles(ctx, self.cfg.gee_project), bounds)
        if not tiles:
            raise ValueError(f"No GlobalBuildingAtlas tile covers the footprints (bounds {bounds})")
        gdf = fetch_features(
            [(f"{ASSET_ROOT}/{name}", b) for name, b in tiles], ctx.cache_dir / "chunks",
            self.cfg.gee_project, properties=["height"],
        )
        return gdf.to_crs(self.cfg.crs)
