"""Tree detection and crown segmentation on synthetic canopy height models."""

import math

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from greenpy.optional.tree_segmentation import (
    PRESETS,
    SegmentationParams,
    dalponte2016,
    detect_treetops,
    get_params,
    ground_scale,
    segment_array,
    segment_raster,
    window_size,
)

RAW = SegmentationParams(smoothing="none", ws_max=24)


def cones(shape, trees, res=1.0):
    """CHM as the max of cones; trees = [(row, col, height, radius_m)]."""
    rr, cc = np.mgrid[: shape[0], : shape[1]]
    z = np.zeros(shape, np.float32)
    for r, c, h, rad in trees:
        d = np.hypot(rr - r, cc - c) * res
        z = np.maximum(z, h * np.clip(1 - d / rad, 0, None))
    return z


def write_tif(path, z, crs="EPSG:27700", origin=(500000, 200000), res=1.0):
    with rasterio.open(
        path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
        crs=crs, transform=from_origin(origin[0], origin[1], res, res),
    ) as dst:
        dst.write(z, 1)
    return path


def test_legacy_window_function():
    p = get_params("legacy_vom")
    # floor at 3 m: round(6 + 18 exp(-225/98)) = 8; peak at mu = 24, capped
    np.testing.assert_array_equal(window_size([0, 3, 18, 18], p, cap=30), [8, 8, 24, 24])
    np.testing.assert_array_equal(window_size([18], p, cap=11.2), [11.2])
    # the cap wins over ws_min, as in lidR
    np.testing.assert_array_equal(window_size([18], p, cap=5), [5])


def test_separate_trees_found_at_their_peaks():
    trees = [(20, 20, 15, 6), (20, 60, 10, 5), (60, 40, 20, 7)]
    z = cones((80, 80), trees)
    rows, cols = detect_treetops(z, 1, 1, RAW, cap=24)
    assert sorted(zip(rows, cols)) == sorted((r, c) for r, c, _, _ in trees)


def test_low_vegetation_ignored():
    z = cones((40, 40), [(20, 20, 1.5, 6)])
    seg = segment_array(z, 1, 1, RAW)
    assert seg.rows.size == 0 and not seg.labels.any()


def test_flat_top_gives_one_seed():
    z = cones((40, 40), [(20, 20, 12, 8)])
    z[19:22, 19:23] = 12.0  # plateau
    rows, cols = detect_treetops(z, 1, 1, RAW, cap=24)
    assert rows.size == 1
    assert (rows[0], cols[0]) == (19, 19)  # first in raster order


def test_touching_crowns_are_split():
    z = cones((40, 60), [(20, 18, 12, 9), (20, 38, 12, 9)])
    seg = segment_array(z, 1, 1, RAW)
    assert seg.rows.size == 2
    a, b = (seg.labels == 1), (seg.labels == 2)
    # th_seed = 0.45 keeps pixels above 5.4 m: about pi * 4.95^2 = 77 px per cone
    assert 50 < a.sum() <= 80 and 50 < b.sum() <= 80
    assert seg.labels[20, 18] == 1 and seg.labels[20, 38] == 2
    # crown 1 stays left of the valley, crown 2 right of it
    assert np.nonzero(a)[1].max() <= 29 and np.nonzero(b)[1].min() >= 27


def test_crown_extent_bounded_by_max_cr():
    z = np.full((60, 60), 10.0, np.float32)
    z[30, 30] = 10.4  # one seed on a flat canopy
    p = SegmentationParams(smoothing="none", max_cr=6)
    labels = dalponte2016(z, np.array([30]), np.array([30]), 1, 1, p)
    r, c = np.nonzero(labels)
    assert np.abs(r - 30).max() < 6 and np.abs(c - 30).max() < 6


def test_crown_respects_height_ratios():
    z = cones((40, 40), [(20, 20, 20, 10)])
    p = SegmentationParams(smoothing="none", th_seed=0.6)
    labels = dalponte2016(z, np.array([20]), np.array([20]), 1, 1, p)
    assert labels.any()
    assert z[labels == 1].min() > 0.6 * 20


def _random_forest(seed=0, shape=(300, 300), n=150):
    rng = np.random.default_rng(seed)
    trees = [
        (rng.uniform(0, shape[0]), rng.uniform(0, shape[1]), rng.uniform(4, 25), rng.uniform(3, 9))
        for _ in range(n)
    ]
    z = cones(shape, trees)
    return (z + rng.normal(0, 0.15, shape)).clip(0).astype(np.float32)


def _key(gdf):
    return sorted(zip(gdf.top_x.round(3), gdf.top_y.round(3), gdf.height.round(4), gdf["area"].round(3)))


@pytest.mark.parametrize("block_size,workers", [(64, 1), (100, 1), (128, 2)])
def test_blocked_matches_single_block(tmp_path, block_size, workers):
    path = write_tif(tmp_path / "chm.tif", _random_forest())
    p = get_params("legacy_vom")
    cap = 14.0  # blocked runs need one shared cap
    whole = segment_raster(path, p, block_size=10**6, ws_cap=cap)
    blocked = segment_raster(path, p, block_size=block_size, n_workers=workers, ws_cap=cap)
    assert len(whole) > 50
    assert _key(blocked) == _key(whole)
    assert blocked.treeID.tolist() == list(range(1, len(blocked) + 1))


def test_polygons_and_points_agree(tmp_path):
    path = write_tif(tmp_path / "chm.tif", _random_forest(seed=1))
    p = get_params("legacy_vom")
    poly = segment_raster(path, p, block_size=128, ws_cap=14.0)
    pts = segment_raster(path, p, block_size=128, ws_cap=14.0, geometry="point")
    np.testing.assert_allclose(poly.geometry.area.values, poly["area"].values)
    assert (pts.geom_type == "Point").all()
    np.testing.assert_allclose(pts.geometry.x, poly.geometry.centroid.x, atol=1e-6)


def test_bounds_limit_owned_trees(tmp_path):
    path = write_tif(tmp_path / "chm.tif", _random_forest(seed=2))
    p = get_params("legacy_vom")
    bounds = (500050, 199850, 500200, 199950)
    sub = segment_raster(path, p, bounds=bounds, block_size=64, ws_cap=14.0)
    whole = segment_raster(path, p, block_size=10**6, ws_cap=14.0)
    inside = whole[whole.top_x.between(bounds[0], bounds[2]) & whole.top_y.between(bounds[1], bounds[3])]
    assert _key(sub) == _key(inside)


def test_mercator_pixels_scaled_to_ground(tmp_path):
    assert ground_scale("EPSG:27700", 500000, 200000) == 1.0
    lat = 52.2
    y = 6378137 * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    assert ground_scale("EPSG:3857", 13358, y) == pytest.approx(math.cos(math.radians(lat)), rel=1e-6)
    with pytest.raises(ValueError):
        ground_scale("EPSG:4326", 0, 52)

    # one 3857 pixel of 1.194 m is ~0.73 m on the ground: crown areas shrink by cos^2(lat)
    z = cones((60, 60), [(30, 30, 12, 15)])
    path = write_tif(tmp_path / "m.tif", z, crs="EPSG:3857", origin=(13358, y), res=1.194)
    gdf = segment_raster(path, SegmentationParams(smoothing="none", ws_max=24))
    assert len(gdf) == 1
    n_px = gdf.geometry.area.iloc[0] / 1.194**2
    assert gdf["area"].iloc[0] == pytest.approx(n_px * (1.194 * math.cos(math.radians(lat))) ** 2, rel=1e-3)


@pytest.mark.parametrize("preset", sorted(PRESETS))
def test_presets_segment_separate_trees(preset):
    trees = [(30, 30, 15, 6), (30, 80, 12, 5), (80, 55, 20, 7)]
    # legacy_vom caps windows at 0.7 * the tile's p95 height, which on this mostly
    # empty tile would be ~2 m; give it the cap of a typical VOM tile (TL4000)
    cap = 11.2 if get_params(preset).ws_max is None else None
    seg = segment_array(cones((110, 110), trees), 1, 1, get_params(preset), ws_cap=cap)
    assert sorted(zip(seg.rows, seg.cols)) == sorted((r, c) for r, c, _, _ in trees)
