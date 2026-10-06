"""Meta CHM v1/v2 as a canopy source: quadkeys, footprint masking, binarisation and config."""

import json

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform as shp_transform
from pyproj import Transformer

from greenpy.config.loader import load_config
from greenpy.optional import canopy_meta, chm_sources
from greenpy.optional.chm_sources import (
    META_V2_NODATA, META_VERSIONS, footprint_dates, mask_to_footprints, quadkeys_for_bounds,
)
from greenpy.t30 import get_canopy_cover_raster

# a 200 x 100 px, 1 m Web-Mercator tile near Cambridge
ORIGIN = (13_500.0, 6_840_000.0)
TO_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform
TO_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform


def _tile(path, z, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="uint8",
        crs="EPSG:3857", transform=from_origin(*ORIGIN, 1.0, 1.0), nodata=nodata,
    ) as dst:
        dst.write(z, 1)
    return path


def _footprints(*boxes_3857, dates=("2018-06-29",)):
    feats = [
        {"type": "Feature", "properties": {"acq_date": d}, "geometry": mapping(shp_transform(TO_4326, b))}
        for b, d in zip(boxes_3857, dates)
    ]
    return {"type": "FeatureCollection", "features": feats}


def test_v2_quadkeys_are_zoom_10():
    # Cambridge sits in zoom-10 tile 1202020002 (verified against the AWS bucket)
    assert quadkeys_for_bounds(0.10, 52.18, 0.15, 52.22, zoom=META_VERSIONS["v2"]["zoom"]) == ["1202020002"]
    assert META_VERSIONS["v1"]["zoom"] == 9


def test_mask_to_footprints_sets_nodata_outside(tmp_path):
    z = np.full((100, 200), 7, np.uint8)
    src = _tile(tmp_path / "raw.tif", z)
    x0, y0 = ORIGIN
    imaged = box(x0, y0 - 100, x0 + 120, y0)  # left 120 columns imaged, the rest is "sea"
    mask_to_footprints(src, _footprints(imaged), tmp_path / "masked.tif", block_rows=32)
    with rasterio.open(tmp_path / "masked.tif") as r:
        out = r.read(1)
        assert r.nodata == META_V2_NODATA
    assert (out[:, :120] == 7).all() and (out[:, 120:] == META_V2_NODATA).all()


def test_mask_without_footprints_keeps_every_pixel(tmp_path):
    src = _tile(tmp_path / "raw.tif", np.zeros((100, 200), np.uint8))
    mask_to_footprints(src, {"features": []}, tmp_path / "masked.tif")
    with rasterio.open(tmp_path / "masked.tif") as r:
        assert (r.read(1) == 0).all()


def test_footprint_dates(tmp_path):
    for name, dates in [("a", ("2019-04-18", "2018-10-09")), ("b", ("2011-05-25",))]:
        boxes = [box(0, 0, 1, 1)] * len(dates)
        (tmp_path / f"{name}.geojson").write_text(json.dumps(_footprints(*boxes, dates=dates)))
    paths = [str(tmp_path / "a.tif"), str(tmp_path / "b.tif"), str(tmp_path / "none.tif")]
    assert footprint_dates(paths) == ("2011-05-25", "2019-04-18")


def _cfg(tmp_path, version):
    raw = {
        "study_area_name": "t", "crs": "EPSG:27700",
        "data": {k: f"{k}.gpkg" for k in ("buildings", "parks_sites", "parks_access", "roads", "census_boundaries")}
        | {"meta_chm": version, "chm_cache_dir": str(tmp_path / "cache")},
        "columns": {"geo_levels": ["district"], "building_id": "id"},
        "output": {"base_dir": str(tmp_path / "out")},
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_config_meta_chm(tmp_path):
    assert _cfg(tmp_path, "V2").data.meta_chm == "v2"
    assert _cfg(tmp_path, None).data.meta_chm is None
    with pytest.raises(ValueError, match="meta_chm"):
        _cfg(tmp_path, "v3")


def test_meta_binary_canopy_binarises_and_masks(tmp_path, monkeypatch):
    # heights: left half 10 m (canopy), right half 1 m; last 40 columns unimaged
    z = np.full((100, 200), 1, np.uint8)
    z[:, :100] = 10
    z[:, 160:] = META_V2_NODATA
    tile = _tile(tmp_path / "1202020002.tif", z, nodata=META_V2_NODATA)
    monkeypatch.setattr(canopy_meta, "download_meta_tiles", lambda *a, **k: [str(tile)])

    cfg = _cfg(tmp_path, "v2")
    x0, y0 = ORIGIN
    whole = gpd.GeoDataFrame({"u": ["all"]}, geometry=[box(x0, y0 - 100, x0 + 200, y0)], crs="EPSG:3857")
    units = gpd.GeoDataFrame(
        {"u": ["canopy", "mixed"]},
        geometry=[box(x0 + 10, y0 - 90, x0 + 90, y0 - 10), box(x0 + 110, y0 - 90, x0 + 190, y0 - 10)],
        crs="EPSG:3857",
    ).to_crs(cfg.crs)

    cache = canopy_meta.meta_cache_path(cfg, "D1", 3, 60)
    assert cache.name == "D1_v2_h3-60.tif"
    da = canopy_meta.meta_binary_canopy(whole.to_crs(cfg.crs), cfg, 3, 60, cache_path=cache)
    assert da.rio.crs.to_epsg() == 27700 and cache.exists()
    assert set(np.unique(da.values[~np.isnan(da.values)])) <= {0.0, 1.0}

    cover = get_canopy_cover_raster(units, da).set_index("u")
    assert cover.loc["canopy", "canopy_cover"] == 100.0
    # "mixed" is 1 m heights (no canopy) plus unimaged pixels, which must not count as valid
    assert cover.loc["mixed", "canopy_cover"] == 0.0
    with rasterio.open(cache) as r:
        full_px = units.to_crs(cfg.crs).area.iloc[1] / abs(r.res[0] * r.res[1])
    assert cover.loc["mixed", "total_pixels"] < 0.75 * full_px  # ~5/8 of the unit is imaged


def test_unknown_meta_version_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown Meta CHM version"):
        chm_sources.download_meta_tiles(
            gpd.GeoDataFrame(geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"), tmp_path, version="v9"
        )
