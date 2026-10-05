"""Line-of-sight kernel: the lowest eye height from which a building sees a tree, over a surface raster.

For a sightline from an observer at height z (point O) to a target at height
zt (point T), the ray height at fraction t of the way is z(1-t) + t·zt. A
surface cell of height h crossed over [ta, tb] blocks the ray unless
z > (h - t·zt) / (1 - t) for every t in [ta, tb]; that bound is monotonic in
t, so it is attained at ta or tb. The maximum over the cells crossed is the
ray's required eye height; visibility only improves as the observer rises
(flat ground), so a tree is visible from every floor whose eye is strictly
above the minimum over the pair's rays (grazing counts as blocked).

Cells are visited exactly (Amanatides–Woo grid traversal). The first skip0
and last skip1 metres of each ray are ignored (rasterisation slack at the
facade and the target). Inside the target tree's own crown (crown_id) only
the building surface counts, so a tree never hides itself.
"""

import math

import numpy as np
from numba import njit, prange


@njit(cache=False, inline="always")
def _in_crown(crown_id, j, i, tid):
    for k in range(crown_id.shape[0]):
        if crown_id[k, j, i] == tid + 1:
            return True
    return False


@njit(cache=False, inline="always")
def _ray_zreq(dsm, bldg, crown_id, x0, y0, res, ox, oy, tx, ty, tz, tid, skip0, skip1, best):
    """Required eye height for one ray, stopping early once it reaches `best` (returns >= best then)."""
    dx, dy = tx - ox, ty - oy
    length = math.sqrt(dx * dx + dy * dy)
    if length <= skip0 + skip1:
        return -np.inf
    t_lo = skip0 / length
    t_hi = 1.0 - skip1 / length
    nrows, ncols = dsm.shape

    # grid coordinates: u = column axis, v = row axis (rows grow southwards)
    u0, v0 = (ox - x0) / res, (y0 - oy) / res
    du, dv = dx / res, -dy / res
    i, j = int(math.floor(u0)), int(math.floor(v0))
    i_end, j_end = int(math.floor(u0 + du)), int(math.floor(v0 + dv))
    step_i = 1 if du > 0 else -1
    step_j = 1 if dv > 0 else -1
    if du != 0.0:
        next_u = i + 1.0 if du > 0 else float(i)
        t_max_u = (next_u - u0) / du
        t_delta_u = abs(1.0 / du)
    else:
        t_max_u = np.inf
        t_delta_u = np.inf
    if dv != 0.0:
        next_v = j + 1.0 if dv > 0 else float(j)
        t_max_v = (next_v - v0) / dv
        t_delta_v = abs(1.0 / dv)
    else:
        t_max_v = np.inf
        t_delta_v = np.inf

    zreq = -np.inf
    t_enter = 0.0
    n_steps = abs(i_end - i) + abs(j_end - j) + 1
    for _ in range(n_steps + 1):
        t_exit = min(t_max_u, t_max_v, 1.0)
        ta = max(t_enter, t_lo)
        tb = min(t_exit, t_hi)
        if ta <= tb and 0 <= j < nrows and 0 <= i < ncols:
            h = bldg[j, i] if _in_crown(crown_id, j, i, tid) else dsm[j, i]
            if h > 0.0:
                za = (h - ta * tz) / (1.0 - ta)
                zb = (h - tb * tz) / (1.0 - tb)
                z = za if za > zb else zb
                if z > zreq:
                    zreq = z
                    if zreq >= best:
                        return zreq
        if t_exit >= t_hi or t_exit >= 1.0:
            break
        t_enter = t_exit
        if t_max_u < t_max_v:
            i += step_i
            t_max_u += t_delta_u
        else:
            j += step_j
            t_max_v += t_delta_v
    return zreq


@njit(parallel=True, cache=False)
def pair_min_eye_height(
    dsm, bldg, crown_id, x0, y0, res,
    ox, oy, onx, ony, ob_start, ob_count,
    tx, ty, tz, tt_start, tt_count,
    pair_b, pair_t, z_cap, z_stop, skip0, skip1,
):
    """Per pair, the lowest eye height (over the building's facade points and the tree's
    target points) from which the tree is visible; inf when that is not below z_cap[b].

    Facade points whose wall faces away from the target are skipped. A pair
    stops searching once some ray needs less than z_stop (e.g. the ground-floor
    eye height), so results below z_stop are only known to be < z_stop. Pass
    z_cap = inf and z_stop = -inf for exact values.
    """
    n = pair_b.shape[0]
    out = np.empty(n, dtype=np.float64)
    for p in prange(n):
        b = pair_b[p]
        t = pair_t[p]
        cap = z_cap[b]
        best = cap
        for kt in range(tt_start[t], tt_start[t] + tt_count[t]):
            for ko in range(ob_start[b], ob_start[b] + ob_count[b]):
                if onx[ko] * (tx[kt] - ox[ko]) + ony[ko] * (ty[kt] - oy[ko]) <= 0.0:
                    continue
                z = _ray_zreq(dsm, bldg, crown_id, x0, y0, res, ox[ko], oy[ko], tx[kt], ty[kt], tz[kt],
                              t, skip0, skip1, best)
                if z < best:
                    best = z
                    if best < z_stop:
                        break
            if best < z_stop:
                break
        out[p] = best if best < cap else np.inf
    return out


def ray_zreq_reference(dsm, bldg, crown_id, x0, y0, res, ox, oy, tx, ty, tz, tid, skip0, skip1) -> float:
    """Pure-Python exact reference for one ray: intersects the sightline with every cell box (shapely)."""
    import shapely
    from shapely.geometry import LineString, box

    length = math.hypot(tx - ox, ty - oy)
    if length <= skip0 + skip1:
        return -math.inf
    t_lo, t_hi = skip0 / length, 1.0 - skip1 / length
    line = LineString([(ox, oy), (tx, ty)])
    zreq = -math.inf
    nrows, ncols = dsm.shape
    for j in range(nrows):
        for i in range(ncols):
            h = bldg[j, i] if (crown_id[:, j, i] == tid + 1).any() else dsm[j, i]
            if h <= 0:
                continue
            cell = box(x0 + i * res, y0 - (j + 1) * res, x0 + (i + 1) * res, y0 - j * res)
            inter = line.intersection(cell)
            if inter.is_empty:
                continue
            ts = [math.hypot(x - ox, y - oy) / length for x, y in shapely.get_coordinates(inter)]
            ta, tb = max(min(ts), t_lo), min(max(ts), t_hi)
            if ta > tb:
                continue
            zreq = max(zreq, (h - ta * tz) / (1 - ta), (h - tb * tz) / (1 - tb))
    return zreq
