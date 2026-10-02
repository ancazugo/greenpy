"""Overture Maps as a parks source: land-use selection and config validation."""

import geopandas as gpd
import pytest
import yaml
from shapely.geometry import LineString, box

from greenpy.config.loader import load_config
from greenpy.overture import select_parks


def _land_use():
    return gpd.GeoDataFrame(
        {
            "id": ["p1", "p2", "r1", "n1", "pitch", "line"],
            "subtype": ["park", "park", "recreation", "protected", "recreation", "park"],
            "class": ["park", "village_green", "recreation_ground", "national_park", "pitch", "park"],
            "names": [{"primary": "City Park"}, None, None, {"primary": "Nairobi National Park"}, None, None],
        },
        geometry=[box(0, 0, 1, 1), box(2, 0, 3, 1), box(4, 0, 5, 1), box(0, 2, 9, 9), box(6, 0, 7, 1),
                  LineString([(0, 0), (1, 1)])],
        crs=4326,
    )


def test_default_parks_match_subtype_or_class():
    parks = select_parks(_land_use(), ["park", "recreation_ground"])
    # park subtype (any class) + recreation_ground class; no protected areas, pitches or lines
    assert parks.park_id.tolist() == ["p1", "p2", "r1"]
    assert parks.name.tolist()[0] == "City Park"
    assert list(parks.columns) == ["park_id", "name", "subtype", "class", "geometry"]


def test_protected_areas_can_be_opted_in():
    assert "n1" in select_parks(_land_use(), ["park", "national_park"]).park_id.tolist()


MINIMAL = {
    "study_area_name": "testville",
    "crs": "EPSG:32737",
    "data": {"buildings": "overture", "parks_sites": "overture", "parks_access": "osm", "roads": "osm",
             "census_boundaries": "census.gpkg"},
    "columns": {"geo_levels": ["county"]},
    "output": {"base_dir": "/tmp/testville"},
}


def _load(tmp_path, data=None, **extra):
    raw = {**MINIMAL, "data": {**MINIMAL["data"], **(data or {})}, **extra}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)


def test_overture_parks_config(tmp_path):
    cfg = _load(tmp_path)
    assert cfg.data.parks_sites == "overture" and cfg.overture.park_land_use == ["park", "recreation_ground"]
    assert _load(tmp_path, overture={"park_land_use": ["park"]}).overture.park_land_use == ["park"]
    with pytest.raises(ValueError, match="park_land_use"):
        _load(tmp_path, overture={"park_land_use": "park"})
    with pytest.raises(ValueError, match="only for data.buildings and data.parks_sites"):
        _load(tmp_path, data={"roads": "overture"})
    with pytest.raises(ValueError, match="only for data.buildings and data.parks_sites"):
        _load(tmp_path, data={"parks_access": "overture"})
    with pytest.raises(ValueError, match="park_function_col"):
        _load(tmp_path, columns={"geo_levels": ["county"], "park_function_col": "kind"})
