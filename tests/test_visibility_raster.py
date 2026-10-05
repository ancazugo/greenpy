"""Raster visibility engine: the T1111 scene, CHM surfaces, and parity with exact prism geometry."""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import shapely
from rasterio.transform import from_origin
from shapely.geometry import LineString, Point, box

from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, OutputPaths
from greenpy.optional.visibility.dsm import build_dsm, read_chm
from greenpy.optional.visibility.inputs import VisibilityInputs
from greenpy.optional.visibility.los import pair_min_eye_height
from greenpy.optional.visibility.observers import facade_points, floor_eyes
from greenpy.optional.visibility.pairs import eligible_pairs
from greenpy.optional.visibility.params import VisibilityParams
from greenpy.optional.visibility.raster_engine import visibility_for_buildings
from greenpy.optional.visibility.targets import tree_targets

CRS = "EPSG:32632"


def _cfg():
    return GreenPyConfig(study_area_name="t", crs=CRS, data=DataPaths("b", "p", "a", "r", "c"),
                         columns=ColumnMapping(geo_levels=["x"]), output=OutputPaths("/nonexistent"))


def _inputs(ids, geoms, heights):
    geoms = np.asarray(geoms)
    return VisibilityInputs(
        building_id=np.asarray(ids), geoms=geoms, height=np.asarray(heights, float),
        height_source=np.array(["native"] * len(ids)), tree=shapely.STRtree(geoms),
        overlay=pd.DataFrame({"building_id": ids, "x": "A"}),
    )


def _trees(geoms, h, area=None):
    area = area if area is not None else [g.area for g in geoms]
    return gpd.GeoDataFrame({"tree_height": h, "tree_area": area}, geometry=list(geoms), crs=CRS)


def _run(inputs, trees, params, idx=None):
    idx = np.arange(len(inputs.building_id)) if idx is None else idx
    n_floors, z_top = floor_eyes(inputs.height[idx], params.storey_height, params.eye_height)
    counts, _, _ = visibility_for_buildings(idx, inputs, trees, params, _cfg(), None, n_floors, z_top)
    return counts.set_index("building_id")


def test_t1111_scene():
    inputs = _inputs(["B_OBS", "B_WALL"], [box(640, 695, 650, 705), box(660, 680, 664, 720)], [20.0, 8.0])
    trees = _trees([Point(680, 700).buffer(2.0)], [10.0])
    counts = _run(inputs, trees, VisibilityParams(buffer=50, tree_area=10, tree_height=3))
    assert counts.loc["B_OBS", ["visible_trees", "visible_trees_ground", "candidate_trees"]].tolist() == [1, 0, 1]
    assert counts.loc["B_WALL", ["visible_trees", "visible_trees_ground", "candidate_trees"]].tolist() == [1, 1, 1]
    assert counts.loc["B_OBS", "n_floors"] == 6


def test_small_trees_still_obstruct_but_do_not_count():
    # a 2 m^2, 9 m shrub-tree (below tree_area) stands between a bungalow and the target
    inputs = _inputs(["H"], [box(0, 0, 10, 10)], [3.0])
    trees = _trees([Point(20, 5).buffer(2.0), Point(14, 5).buffer(0.8)], [10.0, 9.0], area=[12.6, 2.0])
    counts = _run(inputs, trees, VisibilityParams(buffer=50, tree_area=10, tree_height=3, crown_points=0))
    assert counts.loc["H", "candidate_trees"] == 1
    assert counts.loc["H", "visible_trees"] == 1  # other facade points see past the shrub


def test_read_chm_warps_and_masks_roofs(tmp_path):
    path = tmp_path / "chm.tif"
    arr = np.zeros((40, 40), np.float32)
    arr[10:20, 10:20] = 15.0
    arr[0, 0] = -9999
    with rasterio.open(path, "w", driver="GTiff", width=40, height=40, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(0, 40, 1, 1), nodata=-9999) as dst:
        dst.write(arr, 1)
    from affine import Affine
    veg = read_chm([path], CRS, Affine(0.5, 0, 0, 0, -0.5, 40), (80, 80), 0.5)
    assert veg.max() == 15.0 and veg[0, 0] == 0.0 and (veg > 0).sum() == 400
    d = build_dsm((0, 0, 40, 40), 1.0, CRS, np.array([box(10, 25, 15, 30)]), np.array([9.0]), np.array([]),
                  chm_layers=[path], mask_chm_buildings=True)
    assert d.dsm[10:15, 10:15].max() == 9.0  # roof pixels (y 25..30) keep the building, not the CHM


# --------------------------------------------------------------------------- parity with exact prisms

def _exact_zreq(obs, tgt, obstacles, skip):
    """Exact required eye height for one ray over prism obstacles [(polygon, height)]."""
    (ox, oy), (tx, ty, tz) = obs, tgt
    length = math.hypot(tx - ox, ty - oy)
    if length <= 2 * skip:
        return -math.inf
    line = LineString([(ox, oy), (tx, ty)])
    t_lo, t_hi = skip / length, 1 - skip / length
    z = -math.inf
    for poly, h in obstacles:
        inter = line.intersection(poly)
        for part in getattr(inter, "geoms", [inter]):
            if part.is_empty:
                continue
            ts = [math.hypot(x - ox, y - oy) / length for x, y in shapely.get_coordinates(part)]
            ta, tb = max(min(ts), t_lo), min(max(ts), t_hi)
            if ta <= tb:
                z = max(z, (h - ta * tz) / (1 - ta), (h - tb * tz) / (1 - tb))
    return z


@pytest.mark.parametrize("seed", range(3))
def test_raster_matches_exact_prisms_away_from_thresholds(seed):
    rng = np.random.default_rng(seed)
    blds = [box(x, y, x + rng.uniform(6, 15), y + rng.uniform(6, 15))
            for x, y in rng.uniform(0, 100, (12, 2))]
    blds = [b for i, b in enumerate(blds) if not any(b.intersects(o) for o in blds[:i])]
    bh = rng.uniform(3, 25, len(blds))
    crowns = [Point(x, y).buffer(rng.uniform(1.5, 4)) for x, y in rng.uniform(0, 110, (25, 2))]
    crowns = [c for c in crowns if not any(c.intersects(b) for b in blds)]  # crowns may overlap each other
    th = rng.uniform(4, 20, len(crowns))
    params = VisibilityParams(buffer=40, tree_area=0, tree_height=0, resolution=0.25, crown_points=2)

    geoms = np.asarray(blds)
    fp = facade_points(geoms, params.facade_spacing, params.facade_offset, blockers=shapely.STRtree(geoms))
    trees = _trees(crowns, th)
    tg = tree_targets(trees, params.crown_points, params.crown_point_height)
    pb, pt = eligible_pairs(geoms, tg.ref_x, tg.ref_y, params.buffer)
    d = build_dsm((-50, -50, 160, 160), params.resolution, CRS, geoms, bh, tg.crowns, veg_geoms=tg.crowns, veg_h=th)
    z_raster = pair_min_eye_height(
        d.dsm, d.bldg, d.crown_id, d.x0, d.y0, d.res, fp.x, fp.y, fp.nx, fp.ny, fp.start, fp.count,
        tg.x, tg.y, tg.z, tg.start, tg.count, pb, pt, np.full(len(blds), np.inf), -np.inf, params.skip, params.skip,
    )
    eye_levels = [1.5 + 3 * k for k in range(9)]
    checked = 0
    for p, (b, t) in enumerate(zip(pb, pt)):
        # inside the target's crown only buildings block
        obstacles = [(g, h) for g, h in zip(blds, bh)] + [
            (c.difference(crowns[t]), h) for i, (c, h) in enumerate(zip(crowns, th)) if i != t]
        z_exact = math.inf
        for ko in range(fp.start[b], fp.start[b] + fp.count[b]):
            for kt in range(tg.start[t], tg.start[t] + tg.count[t]):
                if fp.nx[ko] * (tg.x[kt] - fp.x[ko]) + fp.ny[ko] * (tg.y[kt] - fp.y[ko]) <= 0:
                    continue
                z_exact = min(z_exact, _exact_zreq((fp.x[ko], fp.y[ko]), (tg.x[kt], tg.y[kt], tg.z[kt]), obstacles, params.skip))
        for eye in eye_levels:
            # rasterisation shifts edges by <= res; skip pairs whose answer hinges on that
            if math.isfinite(z_exact) and abs(z_exact - eye) < 1.0:
                continue
            assert (z_raster[p] < eye) == (z_exact < eye), (seed, b, t, eye, z_raster[p], z_exact)
            checked += 1
    assert checked > 50
