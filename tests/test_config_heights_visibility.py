import pytest
import yaml

from greenpy.config.loader import load_config
from greenpy.config.schema import HeightSourceSpec

from test_config_dggs import MINIMAL


def _load(tmp_path, **overrides):
    raw = {**MINIMAL, **overrides}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_defaults(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.heights.sources == [HeightSourceSpec("native")]
    assert cfg.heights.default_height == 6.0 and cfg.heights.storey_height == 3.0
    assert cfg.visibility.engine == "raster" and cfg.visibility.vegetation == "auto"
    assert cfg.visibility.end_skip is None


def test_string_and_mapping_sources(tmp_path):
    cfg = _load(tmp_path, heights={
        "sources": [
            "native",
            "gba",
            {"source": "utglobus", "city": "bogota"},
            {"source": "open_buildings_temporal", "year": 2022},
            {"source": "file", "path": "/x/ndsm.tif", "name": "lidar"},
        ],
        "default_height": 4.5,
    })
    srcs = cfg.heights.sources
    assert [s.source for s in srcs] == ["native", "gba", "utglobus", "open_buildings_temporal", "file"]
    assert srcs[2].options == {"city": "bogota"}
    assert srcs[4].label == "lidar" and srcs[1].label == "gba"
    assert cfg.heights.default_height == 4.5


@pytest.mark.parametrize("heights, match", [
    ({"sources": ["nope"]}, "unknown height source"),
    ({"sources": ["utglobus"]}, "requires"),
    ({"sources": [{"source": "file"}]}, "requires"),
    ({"sources": [{"source": "gba", "city": "x"}]}, "unknown options"),
    ({"sources": [{"source": "file", "path": "a", "stat": "mode"}]}, "stat"),
    ({"sources": []}, "non-empty"),
    ({"sources": ["gba", "gba"]}, "unique"),
    ({"bogus": 1}, "Unknown heights keys"),
    ({"min_height": 10, "max_height": 5}, "below"),
    ({"min_overlap": 0}, "min_overlap"),
    ({"storey_height": 0}, "storey_height"),
])
def test_bad_heights_rejected(tmp_path, heights, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, heights=heights)


def test_repeated_source_with_names(tmp_path):
    cfg = _load(tmp_path, heights={"sources": [
        {"source": "file", "path": "a.tif", "name": "ndsm_2020"},
        {"source": "file", "path": "b.tif", "name": "ndsm_2016"},
    ]})
    assert [s.label for s in cfg.heights.sources] == ["ndsm_2020", "ndsm_2016"]


def test_visibility_options(tmp_path):
    cfg = _load(tmp_path, visibility={"engine": "vector", "resolution": 0.5, "crown_points": 0, "end_skip": 0.25})
    assert cfg.visibility.engine == "vector"
    assert cfg.visibility.resolution == 0.5 and cfg.visibility.crown_points == 0 and cfg.visibility.end_skip == 0.25


@pytest.mark.parametrize("visibility, match", [
    ({"engine": "gpu"}, "engine"),
    ({"vegetation": "lidar"}, "vegetation"),
    ({"resolution": 0}, "resolution"),
    ({"crown_points": -1}, "crown_points"),
    ({"crown_point_height": 1.5}, "crown_point_height"),
    ({"eye": 1.5}, "Unknown visibility keys"),
])
def test_bad_visibility_rejected(tmp_path, visibility, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, visibility=visibility)


def test_example_configs_still_load():
    from pathlib import Path
    for path in sorted(Path(__file__).parents[1].glob("examples/*.yaml")):
        load_config(path)


def test_terrain_and_context_defaults(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.terrain.source == "fabdem" and cfg.terrain.resolution == 30.0
    assert cfg.context_buffer == 100.0
    flat = _load(tmp_path, terrain=None, context_buffer=0)
    assert flat.terrain.source is None and flat.context_buffer == 0.0
    assert _load(tmp_path, terrain={"source": "/data/dtm.tif", "resolution": 5}).terrain.resolution == 5


@pytest.mark.parametrize("overrides, match", [
    ({"terrain": {"source": "fabdem", "res": 10}}, "Unknown terrain keys"),
    ({"terrain": {"resolution": 0}}, "resolution"),
    ({"terrain": "fabdem"}, "mapping"),
    ({"context_buffer": -5}, "context_buffer"),
])
def test_bad_terrain_rejected(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, **overrides)
