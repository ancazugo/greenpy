"""Height sources that need no network: GBA tile selection, local files, the registry."""

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, HeightSourceSpec, OutputPaths
from greenpy.heights import SOURCE_NAMES, get_source
from greenpy.heights.base import HeightContext
from greenpy.heights.gba import parse_tile, tiles_for_bounds
from greenpy.heights.gee import tiles_with_footprints

CRS = "EPSG:32632"


def _cfg():
    return GreenPyConfig(
        study_area_name="t", crs=CRS, data=DataPaths("b", "p", "a", "r", "c"),
        columns=ColumnMapping(geo_levels=["x"]), output=OutputPaths("/nonexistent"),
    )


def _ctx(tmp_path):
    return HeightContext(cfg=_cfg(), cache_dir=tmp_path, boundaries=None)


def _footprints(*geoms):
    return gpd.GeoDataFrame({"building_id": [f"b{i}" for i in range(len(geoms))]}, geometry=list(geoms), crs=CRS)


def test_parse_tile_hemispheres():
    assert parse_tile("w075_n05_w070_n00") == (-75, 0, -70, 5)
    assert parse_tile("e010_s05_e015_s10") == (10, -10, 15, -5)
    assert parse_tile("w005_n55_e000_n50") == (-5, 50, 0, 55)
    with pytest.raises(ValueError):
        parse_tile("bogus")


def test_tiles_for_bounds_bogota_and_cambridge():
    names = ["w075_n05_w070_n00", "w075_n10_w070_n05", "e000_n55_e005_n50", "w005_n55_e000_n50", "README"]
    bog = tiles_for_bounds(names, (-74.23, 4.47, -73.99, 4.84))
    assert [n for n, _ in bog] == ["w075_n05_w070_n00"]
    assert bog[0][1] == (-74.23, 4.47, -73.99, 4.84)
    cam = tiles_for_bounds(names, (-0.01, 52.1, 0.25, 52.3))
    assert sorted(n for n, _ in cam) == ["e000_n55_e005_n50", "w005_n55_e000_n50"]
    assert dict(cam)["w005_n55_e000_n50"] == (-0.01, 52.1, 0, 52.3)


def test_tiles_with_footprints_aligned_grid():
    fp = _footprints(box(10, 10, 20, 20), box(4090, 0, 4100, 5))  # second straddles the x=4096 tile edge
    assert tiles_with_footprints(fp, res=1.0, tile_px=4096) == [(0, 0), (0, 1)]


def test_registry_knows_every_source():
    for name in SOURCE_NAMES:
        opts = {"city": "x"} if name == "utglobus" else {"path": "x.gpkg"} if name == "file" else {}
        src = get_source(HeightSourceSpec(name, opts), _cfg())
        assert src.label == name and src.kind in ("native", "vector", "raster")


def test_file_source_vector(tmp_path):
    path = tmp_path / "h.gpkg"
    gpd.GeoDataFrame({"hgt": [14.0]}, geometry=[box(0, 0, 10, 10)], crs=CRS).to_file(path)
    src = get_source(HeightSourceSpec("file", {"path": str(path), "column": "hgt"}), _cfg())
    assert src.kind == "vector"
    out = src.heights(_footprints(box(1, 1, 9, 9), box(50, 50, 60, 60)), _ctx(tmp_path))
    assert out["height"].iloc[0] == pytest.approx(14.0) and np.isnan(out["height"].iloc[1])


def test_file_source_raster_dir(tmp_path):
    d = tmp_path / "ndsm"
    d.mkdir()
    for name, x0, v in [("a.tif", 0.0, 3.0), ("b.tif", 20.0, 9.0)]:
        with rasterio.open(d / name, "w", driver="GTiff", width=20, height=20, count=1, dtype="float32",
                           crs=CRS, transform=from_origin(x0, 20.0, 1.0, 1.0)) as dst:
            dst.write(np.full((1, 20, 20), v, dtype="float32"))
    src = get_source(HeightSourceSpec("file", {"path": str(d), "stat": "max"}), _cfg())
    assert src.kind == "raster"
    out = src.heights(_footprints(box(15, 5, 25, 15)), _ctx(tmp_path))
    assert out["height"].iloc[0] == pytest.approx(9.0)


def test_file_source_cache_key_tracks_file_changes(tmp_path):
    path = tmp_path / "h.gpkg"
    gpd.GeoDataFrame({"height": [1.0]}, geometry=[box(0, 0, 1, 1)], crs=CRS).to_file(path)
    k1 = get_source(HeightSourceSpec("file", {"path": str(path)}), _cfg()).cache_key()
    gpd.GeoDataFrame({"height": [1.0, 2.0]}, geometry=[box(0, 0, 1, 1), box(2, 2, 3, 3)], crs=CRS).to_file(path)
    assert get_source(HeightSourceSpec("file", {"path": str(path)}), _cfg()).cache_key() != k1
