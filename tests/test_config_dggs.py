import pytest
import yaml

from greenpy.config.loader import load_config

MINIMAL = {
    "study_area_name": "testville",
    "crs": "EPSG:32632",
    "data": {
        "buildings": "buildings.gpkg",
        "parks_sites": "parks.gpkg",
        "parks_access": "parks_access.gpkg",
        "roads": "roads.gpkg",
        "census_boundaries": "census.gpkg",
    },
    "columns": {"geo_levels": ["district", "tract"], "building_id": "id"},
    "output": {"base_dir": "/tmp/testville"},
}


def _load(tmp_path, **overrides):
    raw = {**MINIMAL, **overrides}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_no_dggs_by_default(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.dggs is None and cfg.dggs_resolution is None


def test_dggs_pair_accepted(tmp_path):
    cfg = _load(tmp_path, dggs="s2", dggs_resolution=17)
    assert (cfg.dggs, cfg.dggs_resolution) == ("s2", 17)


def test_lone_dggs_rejected(tmp_path):
    with pytest.raises(ValueError, match="together"):
        _load(tmp_path, dggs="h3")
    with pytest.raises(ValueError, match="together"):
        _load(tmp_path, dggs_resolution=9)


def test_unknown_system_rejected(tmp_path):
    with pytest.raises(ValueError, match="dggs must be one of"):
        _load(tmp_path, dggs="hexbin", dggs_resolution=9)


@pytest.mark.parametrize("system,bad_res", [("h3", 16), ("s2", 31), ("geohash", 0), ("a5", 31), ("rhealpix", 16)])
def test_out_of_range_resolution_rejected(tmp_path, system, bad_res):
    with pytest.raises(ValueError, match="dggs_resolution"):
        _load(tmp_path, dggs=system, dggs_resolution=bad_res)


def test_deprecated_h3_resolution_maps_to_dggs(tmp_path):
    cfg = _load(tmp_path, h3_resolution=9)
    assert (cfg.dggs, cfg.dggs_resolution) == ("h3", 9)


def test_deprecated_alias_cannot_combine(tmp_path):
    with pytest.raises(ValueError, match="cannot be combined"):
        _load(tmp_path, h3_resolution=9, dggs="h3", dggs_resolution=9)
