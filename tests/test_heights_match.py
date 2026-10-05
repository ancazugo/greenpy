"""Heights chain: vector overlap matching, raster zonal statistics, native heights, coalescing."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from greenpy.config.schema import GreenPyConfig, HeightSourceSpec, HeightsConfig
from greenpy.heights.enrich import coalesce, coverage_table
from greenpy.heights.match import match_by_overlap, zonal_heights
from greenpy.heights.native import NativeHeights
from greenpy.osm import parse_osm_height

CRS = "EPSG:32632"


def _footprints(*geoms, ids=None):
    ids = ids or [f"b{i}" for i in range(len(geoms))]
    return gpd.GeoDataFrame({"building_id": ids}, geometry=list(geoms), crs=CRS)


def _cfg(**heights):
    from greenpy.config.schema import ColumnMapping, DataPaths, OutputPaths
    return GreenPyConfig(
        study_area_name="t", crs=CRS,
        data=DataPaths("b", "p", "a", "r", "c"), columns=ColumnMapping(geo_levels=["x"]),
        output=OutputPaths("/nonexistent"), heights=HeightsConfig(**heights),
    )


# --------------------------------------------------------------------------- vector

def test_overlap_area_weighted_and_coverage():
    fp = _footprints(box(0, 0, 10, 10), box(100, 100, 110, 110), box(200, 0, 210, 10))
    src = gpd.GeoDataFrame(
        {"height": [10.0, 20.0, 30.0, 5.0]},
        geometry=[box(0, 0, 5, 10), box(5, 0, 10, 10),   # splits b0 in two halves
                  box(100, 100, 110, 101),               # covers 10 % of b1
                  box(195, -5, 215, 15)],                # contains b2
        crs=CRS,
    )
    out = match_by_overlap(fp, src, min_overlap=0.3).set_index("building_id")
    assert out.loc["b0", "height"] == pytest.approx(15.0)
    assert out.loc["b0", "quality"] == pytest.approx(1.0)
    assert np.isnan(out.loc["b1", "height"]) and out.loc["b1", "quality"] == pytest.approx(0.1)
    assert out.loc["b2", "height"] == pytest.approx(5.0) and out.loc["b2", "quality"] == pytest.approx(1.0)


def test_overlap_drops_duplicate_source_polygons_and_null_heights():
    fp = _footprints(box(0, 0, 10, 10))
    src = gpd.GeoDataFrame(
        {"height": [12.0, 12.0, None]}, geometry=[box(0, 0, 10, 5), box(0, 0, 10, 5), box(0, 5, 10, 10)], crs=CRS
    )
    out = match_by_overlap(fp, src, min_overlap=0.3)
    # duplicate counted once (50 % cover), null-height polygon ignored
    assert out["height"].iloc[0] == pytest.approx(12.0) and out["quality"].iloc[0] == pytest.approx(0.5)


def test_overlap_reprojects_source_and_handles_empty():
    fp = _footprints(box(500000, 0, 500010, 10))
    src = gpd.GeoDataFrame({"height": [7.0]}, geometry=[box(500000, 0, 500010, 10)], crs=CRS).to_crs(4326)
    assert match_by_overlap(fp, src, 0.3)["height"].iloc[0] == pytest.approx(7.0, abs=1e-6)
    empty = gpd.GeoDataFrame({"height": []}, geometry=[], crs=CRS)
    assert match_by_overlap(fp, empty, 0.3)["height"].isna().all()


# --------------------------------------------------------------------------- raster

def _raster(path, arr, x0=0.0, y0=100.0, res=1.0, nodata=None):
    with rasterio.open(
        path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype="float32",
        crs=CRS, transform=from_origin(x0, y0, res, res), nodata=nodata,
    ) as dst:
        dst.write(arr.astype("float32"), 1)
    return path


def test_zonal_median_matches_rasterstats(tmp_path):
    rng = np.random.default_rng(0)
    arr = rng.uniform(0, 30, (100, 100))
    path = _raster(tmp_path / "h.tif", arr)
    fp = _footprints(box(10.2, 10.2, 30.7, 25.1), box(50, 50, 80, 90), box(0, 0, 3, 3))
    out = zonal_heights(fp, [path], "median")

    from rasterstats import zonal_stats
    ref = zonal_stats(fp, str(path), stats=["median"])
    assert out["height"].to_numpy() == pytest.approx([r["median"] for r in ref], rel=1e-6)
    assert (out["quality"] == 1.0).all() and (out["res_m"] == 1.0).all()


def test_zonal_nodata_quality_and_subpixel_fallback(tmp_path):
    arr = np.full((10, 10), 9.0)
    arr[:, :5] = -9999  # west half nodata
    path = _raster(tmp_path / "h.tif", arr, y0=10.0, res=1.0, nodata=-9999)
    fp = _footprints(box(2, 2, 8, 8), box(7.1, 7.1, 7.4, 7.4), box(50, 50, 60, 60))
    out = zonal_heights(fp, [path], "mean").set_index("building_id")
    assert out.loc["b0", "height"] == pytest.approx(9.0) and out.loc["b0", "quality"] == pytest.approx(0.5)
    # 0.3 m footprint holds no pixel centre: all_touched fallback
    assert out.loc["b1", "height"] == pytest.approx(9.0)
    # outside the raster
    assert np.isnan(out.loc["b2", "height"]) and out.loc["b2", "quality"] == 0.0


def test_zonal_coarse_raster_resampled_approximates_area_weighting(tmp_path):
    # two 100 m cells (10 m and 30 m); a footprint 70 % in the first -> 16 m area-weighted
    coarse = np.array([[10.0, 30.0]])
    fine = np.repeat(np.repeat(coarse, 10, axis=0), 10, axis=1)  # 10 m grid, nearest
    path = _raster(tmp_path / "h.tif", fine, y0=100.0, res=10.0)
    fp = _footprints(box(30, 20, 130, 80))
    out = zonal_heights(fp, [path], "mean")
    assert out["height"].iloc[0] == pytest.approx(16.0)
    assert out["res_m"].iloc[0] == 10.0


def test_zonal_across_tiles_does_not_split_footprints(tmp_path):
    a = _raster(tmp_path / "a.tif", np.full((100, 50), 4.0), x0=0)
    b = _raster(tmp_path / "b.tif", np.full((100, 50), 8.0), x0=50)
    fp = _footprints(box(40, 40, 60, 60))
    out = zonal_heights(fp, [a, b], "mean", vrt_dir=tmp_path, block=16)
    assert out["height"].iloc[0] == pytest.approx(6.0)


# --------------------------------------------------------------------------- native + chain

def test_native_height_then_levels():
    cfg = _cfg(storey_height=3.0)
    fp = _footprints(box(0, 0, 1, 1), box(2, 2, 3, 3), box(4, 4, 5, 5), box(6, 6, 7, 7))
    fp["building_height"] = [12.0, None, 0.0, None]
    fp["num_floors"] = [None, 4, 2, None]
    out = NativeHeights(HeightSourceSpec("native"), cfg).heights(fp, None)
    assert out["height"].tolist()[:3] == [12.0, 12.0, 6.0] and np.isnan(out["height"].iloc[3])
    assert [None if pd.isna(v) else v for v in out["label"]] == [None, "native_levels", "native_levels", None]
    assert out["quality"].tolist() == [1.0, 0.5, 0.5, 0.0]


def test_native_custom_levels_col_must_exist():
    cfg = _cfg()
    fp = _footprints(box(0, 0, 1, 1))
    with pytest.raises(ValueError, match="levels_col"):
        NativeHeights(HeightSourceSpec("native", {"levels_col": "floors"}), cfg).heights(fp, None)


def test_coalesce_chain_order_range_and_default():
    cfg = _cfg(default_height=6.0, min_height=2.0, max_height=300.0)
    ids = pd.Series(["a", "b", "c", "d"])
    first = pd.DataFrame({"building_id": ["a", "b", "c", "d"], "height": [10.0, 1.0, np.nan, 500.0],
                          "quality": 1.0, "res_m": 0.0, "label": [None, None, None, "x_levels"]})
    second = pd.DataFrame({"building_id": ["b", "c", "a"], "height": [7.0, np.nan, 99.0],
                           "quality": [0.8, 0.0, 1.0], "res_m": 100.0})
    out = coalesce(ids, [("first", first), ("second", second)], cfg).set_index("building_id")
    assert out["building_height"].tolist() == [10.0, 7.0, 6.0, 6.0]
    assert out["height_source"].tolist() == ["first", "second", "default", "default"]
    assert out.loc["b", "height_res_m"] == 100.0 and out.loc["b", "height_quality"] == pytest.approx(0.8)
    t = coverage_table(out.reset_index())
    assert t.loc["default", "buildings"] == 2 and t.loc["default", "share_pct"] == 50.0


def test_coalesce_uses_row_labels():
    cfg = _cfg()
    first = pd.DataFrame({"building_id": ["a"], "height": [9.0], "quality": 0.5, "res_m": 0.0, "label": ["native_levels"]})
    out = coalesce(pd.Series(["a"]), [("native", first)], cfg)
    assert out["height_source"].tolist() == ["native_levels"]


def test_osm_height_parsing():
    s = pd.Series(["12", "12.5 m", "40'", "30 ft", "tall", None])
    out = parse_osm_height(s)
    assert out.iloc[:4].tolist() == pytest.approx([12.0, 12.5, 12.192, 9.144])
    assert out.iloc[4:].isna().all()
