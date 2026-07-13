"""Tests for the file-based building use/type filter."""

import geopandas as gpd
import pytest
from shapely.geometry import Point

from greenpy.config.schema import ColumnMapping, GreenPyConfig, DataPaths, OutputPaths
from greenpy.pipeline import _filter_buildings


def _cfg(use_col=None, use_value=None):
    return GreenPyConfig(
        study_area_name="t", crs="EPSG:27700",
        data=DataPaths(buildings="b.parquet", parks_sites="p", parks_access="pa",
                       roads="r", census_boundaries="c"),
        columns=ColumnMapping(geo_levels=["LAD22CD"], building_id="id",
                              building_use_col=use_col, building_use_value=use_value),
        output=OutputPaths(base_dir="/tmp/x"),
    )


def _buildings():
    return gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "map_use": ["Residential", "Retail", "Residential", "Office"],
         "geometry": [Point(i, i) for i in range(4)]},
        crs="EPSG:27700",
    )


def test_no_filter_keeps_all():
    assert len(_filter_buildings(_buildings(), _cfg())) == 4


def test_scalar_filter():
    out = _filter_buildings(_buildings(), _cfg("map_use", "Residential"))
    assert len(out) == 2 and set(out["map_use"]) == {"Residential"}


def test_list_filter():
    out = _filter_buildings(_buildings(), _cfg("map_use", ["Residential", "Office"]))
    assert len(out) == 3


def test_missing_column_raises():
    with pytest.raises(ValueError, match="not found"):
        _filter_buildings(_buildings(), _cfg("premise_type", "Detached"))


def test_no_match_raises():
    with pytest.raises(ValueError, match="No buildings match"):
        _filter_buildings(_buildings(), _cfg("map_use", "Industrial"))
