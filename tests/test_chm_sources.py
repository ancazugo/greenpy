"""CHM sources for the Trees process: tile index, VRT mosaics, Meta quadkeys, config and geo_code ownership."""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from shapely.geometry import box

from greenpy.config.loader import load_config
from greenpy.optional.chm_sources import build_vrt, chm_mosaic, quadkeys_for_bounds, raster_index, tiles_for_boundary
from greenpy.optional.tree_segmentation import process_geo_code, segment_raster, get_params
from greenpy.utils.data_processing import load_trees_gdf
from test_tree_segmentation import _random_forest, write_tif

CRS = "EPSG:27700"


def test_meta_quadkeys():
    # Cambridge sits in zoom-9 tile 120202000 (verified against the AWS bucket)
    assert quadkeys_for_bounds(0.10, 52.18, 0.15, 52.22) == ["120202000"]
    # a box straddling a tile edge needs both tiles
    assert len(quadkeys_for_bounds(-0.01, 52.18, 0.01, 52.22)) == 2


def test_vrt_mosaic_later_tiles_win(tmp_path):
    a = write_tif(tmp_path / "a.tif", np.full((10, 10), 1, np.float32))
    b = write_tif(tmp_path / "b.tif", np.full((10, 10), 2, np.float32), origin=(500010, 200000))
    c = write_tif(tmp_path / "c.tif", np.full((10, 10), 3, np.float32), origin=(500005, 200000))
    vrt = build_vrt([str(a), str(b), str(c)], tmp_path / "m.vrt")
    with rasterio.open(vrt) as src:
        z = src.read(1)
        assert z.shape == (10, 20)
        assert (z[:, :5] == 1).all() and (z[:, 5:15] == 3).all() and (z[:, 15:] == 2).all()


def test_max_overlap_takes_per_pixel_maximum(tmp_path):
    """Two survey years of one tile plus a neighbour: 'max' keeps a tree seen in either year."""
    tiles = tmp_path / "tiles"
    for year in ("2018", "2020"):
        (tiles / year).mkdir(parents=True)
    old, new = np.zeros((10, 10), np.float32), np.zeros((10, 10), np.float32)
    old[2, 2], new[7, 7] = 9.0, 5.0  # each year sees one tree the other misses
    write_tif(tiles / "2018" / "t.tif", old)
    write_tif(tiles / "2020" / "t.tif", new)
    write_tif(tiles / "2020" / "n.tif", np.full((10, 10), 1, np.float32), origin=(500010, 200000))
    boundary = gpd.GeoDataFrame(geometry=[box(500000, 199990, 500020, 200000)], crs=CRS)

    latest = chm_mosaic(boundary, "chm_tiles", tmp_path / "cache", "x", chm_tiles_dir=str(tiles))
    layers = chm_mosaic(boundary, "chm_tiles", tmp_path / "cache", "x", chm_tiles_dir=str(tiles), overlap="max")
    assert len(latest) == 1 and len(layers) == 2
    with rasterio.open(latest[0]) as src:
        z = src.read(1)
        assert z[2, 2] == 0 and z[7, 7] == 5  # the 2020 tile hides the 2018 tree
    from greenpy.optional.tree_segmentation import _read_layers
    from rasterio.windows import Window
    z = _read_layers([str(p) for p in layers], Window(0, 0, 20, 10))
    assert z.shape == (10, 20) and z[2, 2] == 9 and z[7, 7] == 5 and (z[:, 10:] == 1).all()


def test_vrt_rejects_mixed_resolution(tmp_path):
    a = write_tif(tmp_path / "a.tif", np.zeros((4, 4), np.float32))
    b = write_tif(tmp_path / "b.tif", np.zeros((4, 4), np.float32), res=2.0)
    with pytest.raises(ValueError, match="pixel size"):
        build_vrt([str(a), str(b)], tmp_path / "m.vrt")


def test_vom_hillshades_never_read_as_chm(tmp_path):
    """T30/T30_buildings and Trees skip VOM_HS_ hillshades even with the default *.tif pattern."""
    from greenpy.utils.data_processing import find_overlapping_rasters
    tiles = tmp_path / "tiles" / "2019"
    tiles.mkdir(parents=True)
    write_tif(tiles / "VOM_TL0000_P_1_2019.tif", np.zeros((10, 10), np.float32))
    write_tif(tiles / "VOM_HS_TL0000_P_1_2019.tif", np.full((10, 10), 200, np.float32))
    boundary = gpd.GeoDataFrame(geometry=[box(500000, 199990, 500010, 200000)], crs=CRS)
    names = lambda ps: [Path(p).name for p in ps]
    assert names(find_overlapping_rasters(boundary, tmp_path / "tiles", "*.tif")) == ["VOM_TL0000_P_1_2019.tif"]
    assert names(tiles_for_boundary(boundary, tmp_path / "tiles", "*.tif", tmp_path / "cache")) == ["VOM_TL0000_P_1_2019.tif"]


def test_index_cached_with_unreadable_files(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    write_tif(tiles / "VOM_TL0000.tif", np.zeros((10, 10), np.float32))
    write_tif(tiles / "VOM_HS_TL0000.tif", np.zeros((10, 10), np.float32))
    (tiles / "VOM_TL0005.tif").write_bytes(b"not a tiff")
    cache = tmp_path / "cache"
    idx = raster_index(tiles, "VOM_[A-Z][A-Z][0-9]*.tif", cache)
    assert len(idx) == 2 and idx.geometry.isna().sum() == 1  # the hillshade is never indexed
    cached = next(cache.glob("raster_index_*.parquet"))
    mtime = cached.stat().st_mtime_ns
    boundary = gpd.GeoDataFrame(geometry=[box(500000, 199990, 500010, 200000)], crs=CRS)
    assert [p.split("/")[-1] for p in tiles_for_boundary(boundary, tiles, "VOM_[A-Z][A-Z][0-9]*.tif", cache)] == ["VOM_TL0000.tif"]
    assert cached.stat().st_mtime_ns == mtime  # reused, not rebuilt


MINIMAL = {
    "study_area_name": "testville",
    "crs": CRS,
    "data": {
        "buildings": "osm", "parks_sites": "osm", "parks_access": "osm", "roads": "osm",
        "census_boundaries": "census.gpkg",
    },
    "columns": {"geo_levels": ["district"]},
}


def _cfg(tmp_path, data=None, **ts):
    raw = {**MINIMAL, "output": {"base_dir": str(tmp_path / "out")}}
    raw["data"] = {**raw["data"], "trees_dir": str(tmp_path / "trees"), "chm_tiles_dir": str(tmp_path / "tiles"),
                   "chm_cache_dir": str(tmp_path / "cache"), **(data or {})}
    raw["tree_segmentation"] = ts
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_tree_segmentation_config(tmp_path):
    cfg = _cfg(tmp_path, preset="legacy_vom", params={"hmin": 3}, geometry="point")
    assert cfg.tree_segmentation.params == {"hmin": 3} and cfg.tree_segmentation.geometry == "point"
    with pytest.raises(ValueError, match="Unknown tree_segmentation.params"):
        _cfg(tmp_path, params={"window_size": 3})
    with pytest.raises(ValueError, match="Unknown tree_segmentation keys"):
        _cfg(tmp_path, smoothing="median")
    with pytest.raises(ValueError, match="source"):
        _cfg(tmp_path, source="lidar")
    assert _cfg(tmp_path).data.chm_overlap == "latest"
    assert _cfg(tmp_path, data={"chm_overlap": "max"}).data.chm_overlap == "max"
    with pytest.raises(ValueError, match="chm_overlap"):
        _cfg(tmp_path, data={"chm_overlap": "first"})


def _two_years(tmp_path):
    """One tile surveyed in 2018 and 2020 (each sees a tree the other misses) plus a 2020 neighbour."""
    tiles = tmp_path / "tiles"
    for year in ("2018", "2020"):
        (tiles / year).mkdir(parents=True, exist_ok=True)
    old, new = np.zeros((10, 10), np.float32), np.zeros((10, 10), np.float32)
    old[2, 2], new[7, 7] = 9.0, 5.0
    paths = [
        write_tif(tiles / "2018" / "t.tif", old),
        write_tif(tiles / "2020" / "t.tif", new),
        write_tif(tiles / "2020" / "n.tif", np.full((10, 10), 4, np.float32), origin=(500010, 200000)),
    ]
    return [str(p) for p in paths]


@pytest.mark.parametrize("overlap,expected", [("latest", (0, 1)), ("max", (1, 1))])
def test_t30_binarise_follows_overlap(tmp_path, overlap, expected):
    from greenpy.t30 import binarise_tiles
    paths = _two_years(tmp_path)
    binary = binarise_tiles(list(reversed(paths)), 3, 60, overlap=overlap)[0].values  # input order must not matter
    assert binary.shape == (10, 20)
    assert (binary[2, 2], binary[7, 7]) == expected
    assert (binary[:, 10:] == 1).all()


def test_resolve_overlap_passes_disjoint_tiles_through(tmp_path):
    from greenpy.optional.chm_sources import resolve_overlap
    paths = _two_years(tmp_path)
    disjoint = [paths[1], paths[2]]
    assert resolve_overlap(disjoint, "max", tmp_path / "comp") == sorted(disjoint)


@pytest.mark.parametrize("overlap,expected", [("latest", (0, 5)), ("max", (9, 5))])
def test_resolve_overlap_composites_without_overlaps(tmp_path, overlap, expected):
    from greenpy.optional.chm_sources import resolve_overlap, _non_overlapping_layers
    chunks = resolve_overlap(_two_years(tmp_path), overlap, tmp_path / "comp", chunk=8)
    assert len(_non_overlapping_layers(chunks)) == 1  # safe to sum per tile
    vrt = build_vrt(chunks, tmp_path / "check.vrt")
    with rasterio.open(vrt) as src:
        z = src.read(1)
    assert z.shape == (10, 20) and (z[2, 2], z[7, 7]) == expected and (z[:, 10:] == 4).all()


def test_geo_codes_own_disjoint_trees(tmp_path):
    """Two adjacent districts across two CHM tiles: every tree lands in exactly one district file."""
    (tmp_path / "tiles").mkdir()
    z = _random_forest(seed=3, shape=(200, 300), n=120)
    write_tif(tmp_path / "tiles" / "west.tif", np.ascontiguousarray(z[:, :150]))
    write_tif(tmp_path / "tiles" / "east.tif", np.ascontiguousarray(z[:, 150:]), origin=(500150, 200000))
    write_tif(tmp_path / "whole.tif", z)
    boundaries = gpd.GeoDataFrame(
        {"district": ["W", "E"]},
        geometry=[box(500000, 199800, 500137.3, 200000), box(500137.3, 199800, 500300, 200000)],
        crs=CRS,
    )
    cfg = _cfg(tmp_path, block_size=64, params={"ws_max": 14.0})
    for code in ["W", "E"]:
        assert process_geo_code("district", code, cfg, boundaries) is not None

    files = sorted((tmp_path / "trees").glob("trees_*.parquet"))
    per_code = pd.concat([gpd.read_parquet(f) for f in files], ignore_index=True)
    whole = segment_raster(tmp_path / "whole.tif", get_params(ws_max=14.0), block_size=10**6)
    key = lambda d: sorted(zip(d.top_x.round(3), d.top_y.round(3), d.height.round(4)))
    assert len(whole) > 40
    assert key(per_code) == key(whole)

    # T3's loader picks the parquet files up from the directory
    trees = load_trees_gdf(tmp_path / "trees", boundaries, cfg)
    assert len(trees) == len(whole) and {"tree_height", "tree_area"} <= set(trees.columns)
