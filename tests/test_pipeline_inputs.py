"""Input preparation: derived road nodes, edge lengths, park filters and the buildings overlay."""

import geopandas as gpd
import pandas as pd
import pytest
import yaml
from shapely.geometry import LineString, Point, box

from greenpy.config.loader import load_config
from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, OutputPaths
from greenpy.pipeline import (
    _build_overlay, _derive_road_nodes, _ensure_edge_lengths, _filter_park_access, _filter_parks,
)

CRS = "EPSG:27700"


def _cfg(**kwargs):
    return GreenPyConfig(
        study_area_name="t", crs=CRS,
        data=DataPaths(buildings="b", parks_sites="p", parks_access="pa", roads="r", census_boundaries="c"),
        columns=ColumnMapping(geo_levels=["LAD22CD"], building_id="id"),
        output=OutputPaths(base_dir="/tmp/x"),
        **kwargs,
    )


def test_derive_road_nodes_shares_endpoints():
    edges = gpd.GeoDataFrame(
        geometry=[LineString([(0, 0), (100, 0)]), LineString([(100, 0.04), (100, 100)])], crs=CRS
    )
    edges, nodes = _derive_road_nodes(edges)
    # endpoints within a decimetre merge into one node
    assert len(nodes) == 3
    assert edges.loc[0, "road_edge_end"] == edges.loc[1, "road_edge_start"]


def test_edge_lengths_computed_when_column_missing():
    edges = gpd.GeoDataFrame(geometry=[LineString([(0, 0), (30, 40)])], crs=CRS)
    assert _ensure_edge_lengths(edges, "length")["road_edge_length"].tolist() == [50.0]


def test_edge_lengths_filled_only_where_invalid():
    edges = gpd.GeoDataFrame(
        {"road_edge_length": [12.0, None, "bad"]},
        geometry=[LineString([(0, 0), (10, 0)])] * 3, crs=CRS,
    )
    assert _ensure_edge_lengths(edges, "length")["road_edge_length"].tolist() == [12.0, 10.0, 10.0]


def _parks():
    return gpd.GeoDataFrame(
        {"park_id": ["small", "big"]},
        geometry=[box(0, 0, 50, 50), box(0, 0, 100, 100)],  # 0.25 ha, 1 ha
        crs=CRS,
    )


def test_park_min_area_filters_parks_and_linked_access():
    cfg = _cfg(park_min_area_ha=0.5)
    parks = _filter_parks(_parks(), cfg)
    assert list(parks["park_id"]) == ["big"]
    access = gpd.GeoDataFrame(
        {"park_id": ["a0", "a1"], "park_access_ref": ["small", "big"]},
        geometry=[Point(0, 0), Point(1, 1)], crs=CRS,
    )
    assert list(_filter_park_access(access, parks, cfg)["park_access_ref"]) == ["big"]


def test_no_park_min_area_keeps_all():
    assert len(_filter_parks(_parks(), _cfg())) == 2


MINIMAL = {
    "study_area_name": "testville",
    "crs": "EPSG:32632",
    "data": {k: f"{k}.gpkg" for k in ("buildings", "parks_sites", "parks_access", "roads", "census_boundaries")},
    "columns": {"geo_levels": ["district"], "building_id": "id"},
    "output": {"base_dir": "/tmp/testville"},
}


def _load(tmp_path, **overrides):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({**MINIMAL, **overrides}))
    return load_config(path)


def test_config_park_min_area(tmp_path):
    assert _load(tmp_path).park_min_area_ha is None
    assert _load(tmp_path, park_min_area_ha=1).park_min_area_ha == 1.0
    with pytest.raises(ValueError, match="park_min_area_ha"):
        _load(tmp_path, park_min_area_ha=-1)


def test_overlay_assigns_building_on_shared_boundary_once(tmp_path):
    units = gpd.GeoDataFrame(
        {"district": ["D0", "D1"]}, geometry=[box(0, 0, 100, 100), box(100, 0, 200, 100)], crs=CRS
    )
    # b_edge's representative point (100, 50) lies exactly on the shared edge
    buildings = gpd.GeoDataFrame(
        {"building_id": ["b_in", "b_edge", "b_out"]},
        geometry=[box(10, 10, 20, 20), box(95, 45, 105, 55), box(300, 0, 310, 10)], crs=CRS,
    )
    out = tmp_path / "overlay.parquet"
    _build_overlay(buildings, units, ["district"], out)
    overlay = pd.read_parquet(out).set_index("building_id")
    assert overlay.loc["b_in", "district"] == "D0"
    assert overlay.loc["b_edge", "district"] in ("D0", "D1")
    assert "b_out" not in overlay.index and overlay.index.is_unique
