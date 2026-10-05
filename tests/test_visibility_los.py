"""Line-of-sight kernel against analytic cases and an exact shapely reference."""

import math

import numpy as np
import pytest

from greenpy.optional.visibility.los import pair_min_eye_height, ray_zreq_reference

RES = 0.5


def _grid(w, h):
    return np.zeros((h, w), np.float32), np.zeros((h, w), np.float32), np.zeros((h, w), np.int32)


def _burn(arr, x0, y0, xmin, xmax, ymin, ymax, value, res=RES):
    """Set cells whose centre lies in [xmin, xmax] x [ymin, ymax]."""
    rows, cols = arr.shape
    for j in range(rows):
        for i in range(cols):
            cx, cy = x0 + (i + 0.5) * res, y0 - (j + 0.5) * res
            if xmin <= cx <= xmax and ymin <= cy <= ymax:
                arr[j, i] = value


def _single_pair(dsm, bldg, cid, x0, y0, res, obs, normal, targets, z_cap=np.inf, z_stop=-np.inf, skip=0.5):
    ox, oy = np.array([obs[0]]), np.array([obs[1]])
    onx, ony = np.array([normal[0]]), np.array([normal[1]])
    tx, ty, tz = (np.array(v, dtype=float) for v in zip(*targets))
    return pair_min_eye_height(
        dsm, bldg, cid, x0, y0, res, ox, oy, onx, ony, np.array([0]), np.array([1]),
        tx, ty, tz, np.array([0]), np.array([len(targets)]),
        np.array([0]), np.array([0]), np.array([z_cap]), z_stop, skip, skip,
    )[0]


def _wall_scene():
    # T1111 of the synthetic city, shifted: facade at x = 0.5 (observer), 8 m wall over x 10..14,
    # 10 m tree at x = 30, everything along y = 0
    x0, y0 = -5.0, 25.0
    dsm, bldg, cid = _grid(int(45 / RES), int(50 / RES))
    _burn(dsm, x0, y0, 10, 14, -20, 20, 8.0)
    _burn(bldg, x0, y0, 10, 14, -20, 20, 8.0)
    return dsm, bldg, cid, x0, y0


def test_open_ground_visible_from_anywhere():
    dsm, bldg, cid = _grid(80, 80)
    z = _single_pair(dsm, bldg, cid, 0.0, 40.0, RES, (1.0, 20.0), (1, 0), [(30.0, 20.0, 10.0)])
    assert z == -np.inf


def test_wall_analytic_required_height():
    dsm, bldg, cid, x0, y0 = _wall_scene()
    z = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), [(30.0, 0.0, 10.0)])
    ta, tb = 9.5 / 29.5, 13.5 / 29.5
    expected = max((8 - ta * 10) / (1 - ta), (8 - tb * 10) / (1 - tb))
    assert z == pytest.approx(expected, abs=1e-6)  # ~7.05 m: seen from floor 2 (eye 7.5) up, not 1.5/4.5


def test_grazing_counts_as_blocked_and_z_cap():
    dsm, bldg, cid, x0, y0 = _wall_scene()
    z_exact = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), [(30.0, 0.0, 10.0)])
    # a building whose top floor eye is exactly z_req does not see it
    assert _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), [(30.0, 0.0, 10.0)], z_cap=z_exact) == np.inf
    assert _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), [(30.0, 0.0, 10.0)], z_cap=z_exact + 1e-6) == pytest.approx(z_exact)


def test_back_facing_observer_is_skipped():
    dsm, bldg, cid, x0, y0 = _wall_scene()
    z = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (-1, 0), [(30.0, 0.0, 10.0)])
    assert z == np.inf


def test_target_crown_masked_but_other_crown_blocks():
    x0, y0 = 0.0, 20.0
    dsm, bldg, cid = _grid(80, 40)
    # target crown (tree 0) around x = 30 m, 10 m tall; another 12 m crown (tree 1) over x 18..22
    _burn(dsm, x0, y0, 27, 33, 7, 13, 10.0)
    _burn(cid, x0, y0, 27, 33, 7, 13, 1)
    obs, target = (1.0, 10.0), [(30.0, 10.0, 10.0)]
    assert _single_pair(dsm, bldg, cid, x0, y0, RES, obs, (1, 0), target) == -np.inf
    _burn(dsm, x0, y0, 18, 22, 7, 13, 12.0)
    _burn(cid, x0, y0, 18, 22, 7, 13, 2)
    assert _single_pair(dsm, bldg, cid, x0, y0, RES, obs, (1, 0), target) > 7.0


def test_own_building_blocks_rays_passing_back_through_it():
    # L-shaped 20 m building; the observer sits outside the inner corner and the tree is behind the other wing
    x0, y0 = 0.0, 40.0
    dsm, bldg, cid = _grid(80, 80)
    for arr in (dsm, bldg):
        _burn(arr, x0, y0, 0, 10, 0, 30, 20.0)
        _burn(arr, x0, y0, 0, 30, 0, 10, 20.0)
    z = _single_pair(dsm, bldg, cid, x0, y0, RES, (10.5, 20.0), (1, 0), [(35.0, 5.0, 10.0)])
    assert z > 19.0


def test_multiple_targets_take_the_best():
    dsm, bldg, cid, x0, y0 = _wall_scene()
    _burn(dsm, x0, y0, 10, 14, 3, 20, 30.0)  # make the wall much taller north of y = 3
    z = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), [(30.0, 8.0, 10.0), (30.0, -1.0, 6.0)])
    assert z < 10.0


@pytest.mark.parametrize("seed", range(6))
def test_kernel_matches_exact_reference(seed):
    rng = np.random.default_rng(seed)
    x0, y0, res = 100.0, 260.0, 1.0
    dsm = np.zeros((40, 40), np.float32)
    for _ in range(12):  # random boxes of random heights
        j, i = rng.integers(0, 36, 2)
        dsm[j:j + rng.integers(1, 5), i:i + rng.integers(1, 5)] = rng.uniform(1, 25)
    bldg = dsm.copy()
    cid = np.zeros_like(dsm, dtype=np.int32)
    for _ in range(25):
        ox, oy, tx, ty = rng.uniform(100, 140), rng.uniform(220, 260), rng.uniform(100, 140), rng.uniform(220, 260)
        tz = rng.uniform(3, 20)
        ref = ray_zreq_reference(dsm, bldg, cid, x0, y0, res, ox, oy, tx, ty, tz, 0, 0.5, 0.5)
        got = _single_pair(dsm, bldg, cid, x0, y0, res, (ox, oy), (tx - ox, ty - oy), [(tx, ty, tz)])
        if math.isinf(ref):
            assert got == ref
        else:
            assert got == pytest.approx(ref, rel=1e-6, abs=1e-6)


def test_z_stop_prunes_but_keeps_threshold_answers():
    dsm, bldg, cid, x0, y0 = _wall_scene()
    target = [(30.0, 0.0, 10.0)]
    exact = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), target)
    pruned = _single_pair(dsm, bldg, cid, x0, y0, RES, (0.5, 0.0), (1, 0), target, z_cap=16.5, z_stop=1.5)
    assert (pruned < 16.5) == (exact < 16.5) and (pruned < 1.5) == (exact < 1.5)
