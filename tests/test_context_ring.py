"""Context ring around the study area: ring geometry, context buildings, merged footprints for Heights/Visibility."""

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import box

from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, OutputPaths
from greenpy.pipeline import all_buildings, context_ring, ensure_context_buildings

CRS = "EPSG:32618"


def _cfg(tmp_path, buildings="overture", context_buffer=100.0):
    (tmp_path / "database").mkdir(exist_ok=True)
    return GreenPyConfig(
        study_area_name="t", crs=CRS, data=DataPaths(buildings, "p", "a", "r", "c"),
        columns=ColumnMapping(geo_levels=["unit"]), output=OutputPaths(str(tmp_path)), context_buffer=context_buffer,
    )


def _study(tmp_path):
    (tmp_path / "database").mkdir(exist_ok=True)
    census = gpd.GeoDataFrame({"unit": ["A", "B"]}, geometry=[box(500000, 500000, 501000, 501000),
                                                               box(501000, 500000, 502000, 501000)], crs=CRS)
    census.to_parquet(tmp_path / "database" / "census_boundaries.parquet")
    gpd.GeoDataFrame({"building_id": ["in1", "edge"], "building_height": [6.0, 9.0]},
                     geometry=[box(500100, 500100, 500110, 500110), box(499995, 500500, 500005, 500510)],
                     crs=CRS).to_parquet(tmp_path / "database" / "buildings.parquet")
    return census


def test_context_ring_is_the_band_around_the_union(tmp_path):
    ring = context_ring(_study(tmp_path), 100)
    assert ring.area == pytest.approx(6000 * 100 + 3.14159 * 100 ** 2, rel=0.001)  # perimeter x width + rounded corners
    assert not ring.intersects(box(500001, 500001, 501999, 500999))


def test_context_buildings_fetched_once_and_merged(tmp_path, monkeypatch):
    _study(tmp_path)
    calls = []

    def fake_fetch(polygon_4326, crs):
        calls.append(polygon_4326)
        return gpd.GeoDataFrame({"building_id": ["edge", "out1"], "building_height": [9.0, 12.0]},
                                geometry=[box(499995, 500500, 500005, 500510), box(499950, 500200, 499960, 500210)],
                                crs=CRS)

    monkeypatch.setattr("greenpy.overture.fetch_overture_buildings", fake_fetch)
    cfg = _cfg(tmp_path)
    out = ensure_context_buildings(cfg)
    ctx = gpd.read_parquet(out)
    assert ctx["building_id"].tolist() == ["out1"]  # "edge" is already a study-area building
    assert ensure_context_buildings(cfg) == out and len(calls) == 1  # cached
    merged = all_buildings(cfg)
    assert sorted(merged["building_id"]) == ["edge", "in1", "out1"]
    # a different buffer refetches
    ensure_context_buildings(_cfg(tmp_path, context_buffer=50.0))
    assert len(calls) == 2


def test_no_context_for_files_or_zero_buffer(tmp_path):
    _study(tmp_path)
    assert ensure_context_buildings(_cfg(tmp_path, buildings="/data/b.gpkg")) is None
    assert ensure_context_buildings(_cfg(tmp_path, context_buffer=0)) is None
    assert sorted(all_buildings(_cfg(tmp_path, context_buffer=0))["building_id"]) == ["edge", "in1"]
