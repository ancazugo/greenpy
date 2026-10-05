"""Observer windows: points around each footprint's facade, at the eye height of every floor."""

from dataclasses import dataclass

import numpy as np
import shapely


@dataclass
class FacadePoints:
    """Observer points grouped by building: point i belongs to building b_idx[i]; building b
    owns points start[b] : start[b] + count[b]. (nx, ny) is the wall's outward unit normal."""

    b_idx: np.ndarray
    x: np.ndarray
    y: np.ndarray
    nx: np.ndarray
    ny: np.ndarray
    start: np.ndarray
    count: np.ndarray

    def __len__(self) -> int:
        return len(self.x)


def facade_points(
    geoms: np.ndarray, spacing: float, offset: float, blockers: shapely.STRtree | None = None,
) -> FacadePoints:
    """A point every ~spacing metres along each footprint's exterior wall, offset metres outside it.

    Each exterior ring (one per polygon part; courtyards are ignored) gets
    N = max(4, round(perimeter / spacing)) points at the midpoints of N equal
    arcs. Points falling inside any footprint in `blockers` — party walls of
    terraced houses, walls touching a neighbour — are dropped: those walls
    have no windows.
    """
    n_b = len(geoms)
    parts, part_b = shapely.get_parts(geoms, return_index=True)
    parts = shapely.orient_polygons(parts, exterior_cw=False)  # CCW: the outside is on the right
    coords, ring = shapely.get_coordinates(shapely.get_exterior_ring(parts), return_index=True)

    same = ring[:-1] == ring[1:]
    x0, y0 = coords[:-1][same, 0], coords[:-1][same, 1]
    dx, dy = coords[1:][same, 0] - x0, coords[1:][same, 1] - y0
    seg_ring = ring[:-1][same]
    seg_len = np.hypot(dx, dy)
    keep = seg_len > 0
    x0, y0, dx, dy, seg_ring, seg_len = x0[keep], y0[keep], dx[keep], dy[keep], seg_ring[keep], seg_len[keep]

    n_parts = len(parts)
    perim = np.bincount(seg_ring, weights=seg_len, minlength=n_parts)
    n_pts = np.where(perim > 0, np.maximum(4, np.round(perim / spacing)), 0).astype(np.int64)
    cum = np.cumsum(seg_len)
    ring_base = np.zeros(n_parts)
    first_seg = np.searchsorted(seg_ring, np.arange(n_parts))
    has_seg = first_seg < len(seg_ring)
    ring_base[has_seg] = cum[first_seg[has_seg]] - seg_len[first_seg[has_seg]]

    pt_ring = np.repeat(np.arange(n_parts), n_pts)
    k = np.arange(len(pt_ring)) - np.repeat(np.cumsum(n_pts) - n_pts, n_pts)
    pos = ring_base[pt_ring] + (k + 0.5) * perim[pt_ring] / n_pts[pt_ring]
    seg = np.searchsorted(cum, pos, side="right")
    seg = np.minimum(seg, len(cum) - 1)
    t = (pos - (cum[seg] - seg_len[seg])) / seg_len[seg]
    nx, ny = dy[seg] / seg_len[seg], -dx[seg] / seg_len[seg]
    px = x0[seg] + t * dx[seg] + offset * nx
    py = y0[seg] + t * dy[seg] + offset * ny
    pb = part_b[pt_ring]

    if blockers is not None and len(px):
        hit, _ = blockers.query(shapely.points(px, py), predicate="intersects")
        ok = np.ones(len(px), dtype=bool)
        ok[hit] = False
        px, py, nx, ny, pb = px[ok], py[ok], nx[ok], ny[ok], pb[ok]

    order = np.argsort(pb, kind="stable")
    pb, px, py, nx, ny = pb[order], px[order], py[order], nx[order], ny[order]
    count = np.bincount(pb, minlength=n_b)
    start = np.cumsum(count) - count
    return FacadePoints(pb.astype(np.int64), px, py, nx, ny, start.astype(np.int64), count.astype(np.int64))


def floor_eyes(height: np.ndarray, storey: float, eye: float) -> tuple[np.ndarray, np.ndarray]:
    """(n_floors, z_top): floors = max(1, floor(H / storey)); the top floor's eye is at (n_floors - 1) * storey + eye."""
    n_floors = np.maximum(1, np.floor(np.asarray(height, dtype=float) / storey)).astype(np.int64)
    return n_floors, (n_floors - 1) * storey + eye
