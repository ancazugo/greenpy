"""Canopy-cover arithmetic of the T30 vector and raster paths."""

import geopandas as gpd
import numpy as np
import pytest
import rioxarray  # noqa: F401 — registers the .rio accessor
import xarray as xr
from shapely.geometry import Point, box

from greenpy.t30 import get_canopy_cover_raster, get_canopy_cover_vector

CRS = "EPSG:27700"


def _units():
    # two adjacent 100 x 100 m units
    return gpd.GeoDataFrame({"u": ["A", "B"]}, geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=CRS)


def _cover(result):
    return dict(zip(result["u"], result["canopy_cover"]))


def test_straddling_crown_is_split_between_units():
    # 20 x 20 m crown, half in each unit; its tree_area attribute must not override the clip
    tree = gpd.GeoDataFrame({"tree_area": [400.0]}, geometry=[box(90, 40, 110, 60)], crs=CRS)
    assert _cover(get_canopy_cover_vector(_units(), tree)) == {"A": 2.0, "B": 2.0}


def test_overlapping_crowns_are_not_double_counted():
    trees = gpd.GeoDataFrame(geometry=[box(10, 10, 20, 20), box(10, 10, 20, 20), box(15, 10, 25, 20)], crs=CRS)
    # union = 15 x 10 = 150 m2 of 10 000 m2
    assert _cover(get_canopy_cover_vector(_units(), trees)) == {"A": 1.5, "B": 0.0}


def test_point_trees_use_tree_area_in_containing_unit():
    pts = gpd.GeoDataFrame({"tree_area": [50.0, 100.0]}, geometry=[Point(10, 10), Point(150, 50)], crs=CRS)
    result = get_canopy_cover_vector(_units(), pts)
    assert _cover(result) == {"A": 0.5, "B": 1.0}
    assert list(result["total_pixels"]) == [10000, 10000]


def test_point_trees_without_area_raise():
    pts = gpd.GeoDataFrame(geometry=[Point(10, 10)], crs=CRS)
    with pytest.raises(ValueError, match="tree_area"):
        get_canopy_cover_vector(_units(), pts)


def _raster(values):
    """1 m raster over x 0..200, y 0..100 from a (100, 200) array."""
    return xr.DataArray(
        values[np.newaxis],
        dims=("band", "y", "x"),
        coords={"band": [1], "y": np.arange(99.5, 0, -1), "x": np.arange(0.5, 200, 1)},
    ).rio.write_crs(CRS)


def test_raster_excludes_nodata_from_denominator():
    arr = np.zeros((100, 200))
    arr[:, :50] = 1.0  # half of unit A is canopy
    arr[:, 150:] = np.nan  # half of unit B is unmapped
    result = get_canopy_cover_raster(_units(), _raster(arr))
    assert _cover(result) == {"A": 50.0, "B": 0.0}
    assert list(result["total_pixels"]) == [10000, 5000]


def test_raster_fractional_cover():
    # coarsened GEE masks hold canopy fraction per pixel
    arr = np.full((100, 200), 0.25)
    arr[:, 100:] = 0.6
    assert _cover(get_canopy_cover_raster(_units(), _raster(arr))) == {"A": 25.0, "B": 60.0}


def test_raster_unit_without_valid_pixels_is_nan():
    arr = np.zeros((100, 200))
    arr[:, 100:] = np.nan
    result = _cover(get_canopy_cover_raster(_units(), _raster(arr)))
    assert result["A"] == 0.0 and np.isnan(result["B"])
