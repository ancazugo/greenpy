from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from greenpy.dggs.geohash import GeohashSystem, cell_size, geohash_encode


def test_encode_known_vectors():
    # Reference values from the original geohash.org implementation
    assert geohash_encode(57.64911, 10.40744, 11) == "u4pruydqqvj"
    assert geohash_encode(42.6, -5.6, 5) == "ezs42"
    assert geohash_encode(0.0, 0.0, 1) == "s"
    assert geohash_encode(-25.382708, -49.265506, 8) == "6gkzwgjz"


def test_cell_size_halves_per_bit():
    w5, h5 = cell_size(5)
    assert w5 == 360.0 / 2**13
    assert h5 == 180.0 / 2**12


def test_cells_covering_polygon():
    poly = Polygon([(0.10, 52.20), (0.16, 52.20), (0.16, 52.24), (0.10, 52.24)])
    cells = list(GeohashSystem().cells_covering(poly, 6))
    assert cells
    # ids match their box centres and boxes jointly cover the polygon
    for cell_id, box_poly in cells:
        centre = box_poly.centroid
        assert geohash_encode(centre.y, centre.x, 6) == cell_id
    assert unary_union([b for _, b in cells]).contains(poly)


def test_cells_covering_point_like():
    tiny = Point(0.12, 52.21).buffer(1e-6)
    cells = list(GeohashSystem().cells_covering(tiny, 9))
    assert len(cells) in (1, 2, 4)  # bbox may straddle cell edges
