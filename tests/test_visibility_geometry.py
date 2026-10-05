"""Visibility geometry: facade observer points, floors, tree targets, building–tree pairs, counts."""

import geopandas as gpd
import numpy as np
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, box

from greenpy.optional.visibility.observers import facade_points, floor_eyes
from greenpy.optional.visibility.pairs import building_counts, eligible_pairs
from greenpy.optional.visibility.targets import tree_targets


@pytest.mark.parametrize("geom", [box(0, 0, 10, 10), shapely.reverse(box(0, 0, 10, 10))])
def test_square_facade_points_either_orientation(geom):
    fp = facade_points(np.array([geom]), spacing=5.0, offset=0.5)
    assert len(fp) == 8 and fp.count.tolist() == [8] and fp.start.tolist() == [0]
    # every point 0.5 m outside the wall, normal pointing away from the centre
    d = shapely.distance(shapely.points(fp.x, fp.y), geom)
    assert d == pytest.approx(np.full(8, 0.5))
    assert ((fp.x - 5) * fp.nx + (fp.y - 5) * fp.ny > 0).all()
    assert np.hypot(fp.nx, fp.ny) == pytest.approx(np.ones(8))


def test_tiny_building_gets_four_points_and_multipolygon_parts():
    geoms = np.array([box(0, 0, 1, 1), MultiPolygon([box(10, 0, 20, 10), box(30, 0, 40, 10)])])
    fp = facade_points(geoms, spacing=5.0, offset=0.5)
    assert fp.count.tolist() == [4, 16]
    assert (fp.b_idx[fp.start[1]:] == 1).all()


def test_party_wall_points_dropped():
    a, b = box(0, 0, 10, 10), box(10, 0, 20, 10)  # terraced pair sharing x = 10
    geoms = np.array([a, b])
    fp = facade_points(geoms, spacing=5.0, offset=0.5, blockers=shapely.STRtree(geoms))
    pts = shapely.points(fp.x, fp.y)
    assert not shapely.intersects(pts, a).any() and not shapely.intersects(pts, b).any()
    assert fp.count.tolist() == [6, 6]  # the 2 points on each shared wall are gone


def test_floor_eyes():
    n, z_top = floor_eyes(np.array([20.0, 2.0, 6.0, 8.9]), storey=3.0, eye=1.5)
    assert n.tolist() == [6, 1, 2, 2]
    assert z_top.tolist() == [16.5, 1.5, 4.5, 4.5]


def _trees(geoms, h, **cols):
    return gpd.GeoDataFrame({"tree_height": h, **cols}, geometry=geoms, crs="EPSG:32632")


def test_targets_use_treetop_columns_and_crown_points():
    crown = Point(0, 0).buffer(4)
    t = tree_targets(_trees([crown], [12.0], top_x=[1.0], top_y=[-1.0]), crown_points=4, crown_frac=2 / 3)
    assert t.count.tolist() == [5] and t.start.tolist() == [0]
    assert (t.x[0], t.y[0], t.z[0]) == (1.0, -1.0, 12.0)
    assert t.z[1:] == pytest.approx(np.full(4, 8.0))
    assert shapely.contains(crown, shapely.points(t.x[1:], t.y[1:])).all()
    assert (t.ref_x[0], t.ref_y[0]) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_targets_without_treetop_and_point_trees():
    t = tree_targets(_trees([box(0, 0, 4, 4), Point(10, 10)], [9.0, 6.0], tree_area=[16.0, np.pi * 4]), crown_points=0)
    assert t.count.tolist() == [1, 1]
    assert shapely.contains(box(0, 0, 4, 4), Point(t.x[0], t.y[0]))
    assert shapely.area(t.crowns[1]) == pytest.approx(np.pi * 4, rel=0.1)  # disc of radius 2
    assert (t.x[1], t.y[1]) == pytest.approx((10.0, 10.0), abs=0.6)


def test_pairs_match_t3_rule_and_counts():
    fps = np.array([box(0, 0, 10, 10), box(100, 0, 110, 10)])
    ref_x, ref_y = np.array([15.0, 25.0, 105.0]), np.array([5.0, 5.0, 60.1])
    b, t = eligible_pairs(fps, ref_x, ref_y, buffer=10)
    assert list(zip(b, t)) == [(0, 0)]  # 25 is 15 m away; (105, 60.1) is 50.1 m away
    b, t = eligible_pairs(fps, ref_x, ref_y, buffer=51)
    assert list(zip(b, t)) == [(0, 0), (0, 1), (1, 2)]

    z_req = np.array([-np.inf, 7.0, np.inf])
    counts = building_counts(2, b, z_req, z_top=np.array([16.5, 4.5]), eye=1.5)
    assert counts["visible_trees"].tolist() == [2, 0]
    assert counts["visible_trees_ground"].tolist() == [1, 0]
    assert counts["candidate_trees"].tolist() == [2, 1]
