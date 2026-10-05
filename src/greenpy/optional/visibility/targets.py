"""Tree targets: the treetop plus points on the crown, and the crown used for self-masking."""

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import shapely

# Radius (m) given to point trees with no usable area
_DEFAULT_RADIUS = 1.0


@dataclass
class Targets:
    """Sightline targets grouped by tree: tree t owns points start[t] : start[t] + count[t]
    (treetop first). ref_x/ref_y is the crown centroid (pair eligibility, as T3); crowns
    are polygons (point trees get a disc of their crown area)."""

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    start: np.ndarray
    count: np.ndarray
    ref_x: np.ndarray
    ref_y: np.ndarray
    height: np.ndarray
    crowns: np.ndarray

    def __len__(self) -> int:
        return len(self.ref_x)


def crown_polygons(trees: gpd.GeoDataFrame) -> np.ndarray:
    """Crown polygons; points become discs of radius sqrt(tree_area / pi)."""
    geoms = trees.geometry.values
    is_point = shapely.get_type_id(geoms) == 0
    if not is_point.any():
        return np.asarray(geoms)
    area = trees["tree_area"].to_numpy(dtype=float) if "tree_area" in trees.columns else np.full(len(trees), np.nan)
    radius = np.where(np.isfinite(area) & (area > 0), np.sqrt(area / np.pi), _DEFAULT_RADIUS)
    out = np.asarray(geoms).copy()
    out[is_point] = shapely.buffer(geoms[is_point], radius[is_point], quad_segs=4)
    return out


def tree_targets(trees: gpd.GeoDataFrame, crown_points: int = 4, crown_frac: float = 2 / 3, inset: float = 0.9) -> Targets:
    """Targets for trees with canonical tree_height (and optionally top_x, top_y, tree_area).

    The treetop is (top_x, top_y) when given (the Trees process), else the
    crown's representative point, at the tree height. crown_points points are
    spread evenly along the crown outline, pulled `inset` of the way out from
    the treetop (so they lie inside the crown), at crown_frac x height.
    """
    n = len(trees)
    crowns = crown_polygons(trees)
    h = trees["tree_height"].to_numpy(dtype=float)
    centroid = shapely.centroid(crowns)
    ref_x, ref_y = shapely.get_x(centroid), shapely.get_y(centroid)

    rep = shapely.point_on_surface(crowns)
    top_x, top_y = shapely.get_x(rep), shapely.get_y(rep)
    if "top_x" in trees.columns and "top_y" in trees.columns:
        tx = trees["top_x"].to_numpy(dtype=float)
        ty = trees["top_y"].to_numpy(dtype=float)
        ok = np.isfinite(tx) & np.isfinite(ty)
        top_x, top_y = np.where(ok, tx, top_x), np.where(ok, ty, top_y)

    k = int(crown_points)
    count = np.full(n, 1 + k, dtype=np.int64)
    start = np.arange(n, dtype=np.int64) * (1 + k)
    x = np.empty(n * (1 + k))
    y = np.empty(n * (1 + k))
    z = np.empty(n * (1 + k))
    x[start], y[start], z[start] = top_x, top_y, h
    if k:
        # the largest part's outline for multipart crowns
        rings = shapely.get_exterior_ring(_largest_part(crowns))
        frac = (np.arange(k) + 0.5) / k
        pts = shapely.line_interpolate_point(np.repeat(rings, k), np.tile(frac, n), normalized=True)
        px, py = shapely.get_x(pts), shapely.get_y(pts)
        tx_rep, ty_rep = np.repeat(top_x, k), np.repeat(top_y, k)
        idx = (start[:, None] + 1 + np.arange(k)[None, :]).ravel()
        x[idx] = tx_rep + inset * (px - tx_rep)
        y[idx] = ty_rep + inset * (py - ty_rep)
        z[idx] = np.repeat(h * crown_frac, k)
    return Targets(x, y, z, start, count, ref_x, ref_y, h, crowns)


def _largest_part(geoms: np.ndarray) -> np.ndarray:
    multi = shapely.get_type_id(geoms) == 6
    if not multi.any():
        return geoms
    out = np.asarray(geoms).copy()
    for i in np.flatnonzero(multi):
        parts = shapely.get_parts(geoms[i])
        out[i] = parts[np.argmax(shapely.area(parts))]
    return out
