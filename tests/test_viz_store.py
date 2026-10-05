import gzip
import json
import math
import os

import pytest

pytest.importorskip("duckdb")

from greenpy.viz.catalog import build_catalog
from greenpy.viz.store import connect, ensure_store, read_meta
from greenpy.viz.tiles import TileStore

from viz_fixtures import make_config, make_outputs, make_trees

LON, LAT = -0.1281, 51.5040  # ~ (530000, 180000) in EPSG:27700


def _tile_xy(z, lon=LON, lat=LAT):
    n = 2**z
    return int((lon + 180) / 360 * n), int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def _open(path):
    con = connect(path, read_only=True)
    try:
        meta = read_meta(con)
    finally:
        con.close()
    return TileStore(path, meta)


@pytest.fixture
def store(tmp_path):
    make_outputs(tmp_path, merged=True)
    trees = make_trees(tmp_path / "trees.gpkg")
    cat = build_catalog(make_config(tmp_path, trees))
    path = ensure_store(cat)
    ts = _open(path)
    yield cat, path, ts
    ts.close()


def test_layers_and_metrics(store):
    _, _, ts = store
    layers = ts.meta["layers"]
    assert set(layers) == {"buildings", "DIST", "TRACT"}
    bm = {m["name"]: m for m in layers["buildings"]["metrics"]}
    assert bm["distance_euclidean"]["threshold"] == 300
    # Merge's unit table reuses this name; the building metric still belongs to T3
    assert bm["tree_count_50m"]["module"] == "T3"
    assert {m["name"]: m["module"] for m in layers["DIST"]["metrics"]}["tree_count_50m"] == "Merge"
    assert bm["meets_3_30_300"]["kind"] == "boolean"
    # TRACT has no Merge table: native T30/Tree_count plus building averages
    tract = {m["name"] for m in layers["TRACT"]["metrics"]}
    assert {"canopy_cover", "tree_count", "n_buildings", "mean_tree_count_50m", "pct_meets_3"} <= tract
    # DIST is Merge's level: its own means, no recomputed averages
    dist = {m["name"] for m in layers["DIST"]["metrics"]}
    assert "pct_meets_3_30_300" in dist and "n_buildings" not in dist
    assert ts.meta["trees"] == {"count": 3, "sized": True, "has_height": True}
    w, s, e, n = ts.meta["bounds"]
    assert w < LON < e and s < LAT < n


def test_feature_lookup(store):
    _, _, ts = store
    b = ts.feature("buildings", "1")
    assert b["tree_count_50m"] == 3 and b["meets_3"] is True and b["TRACT"] == "T0"
    assert "geom" not in b and "minx" not in b
    t = ts.feature("TRACT", "T0")
    assert t["n_buildings"] == 2 and t["canopy_cover"] == 12.5 and t["mean_tree_count_50m"] == 2.5
    # omitted from the module CSVs, but inside a processed geo code: 0, not unknown
    assert ts.feature("buildings", "0")["tree_count_10m"] == 0
    assert t["tree_count"] == 0
    assert ts.feature("buildings", "nope") is None


def test_stats(store):
    _, _, ts = store
    s = json.loads(ts.stats("buildings", "tree_count_10m"))
    assert s["n"] == 3 and s["null"] == 0 and s["min"] == 0 and s["max"] == 4
    assert sum(s["hist"]) == 3 and len(s["equal"]) == 6
    # integer counts get one bin per whole value: 0..4 -> five bins [0, 5)
    assert s["integer"] and s["domain"] == [0, 5] and s["hist"] == [1, 1, 0, 0, 1]
    s = json.loads(ts.stats("buildings", "distance_euclidean"))
    assert not s["integer"] and len(s["hist"]) == 40
    s = json.loads(ts.stats("buildings", "distance_manhattan"))
    assert s["null"] == 1
    s = json.loads(ts.stats("buildings", "meets_3"))
    assert (s["true"], s["false"], s["null"]) == (2, 1, 0)


def test_tiles(store):
    _, _, ts = store
    x, y = _tile_xy(16)
    data = ts.tile("buildings", "tree_count_50m", 16, x, y)
    assert data and len(gzip.decompress(data)) > 0
    assert ts.tile("buildings", "tree_count_50m", 16, x + 40, y) is None
    assert ts.tile("buildings", "tree_count_50m", 10, *_tile_xy(10)) is None  # below min zoom
    assert ts.tile("TRACT", "canopy_cover", 10, *_tile_xy(10))  # units draw at every zoom
    assert ts.tile("trees", None, 16, x, y)
    with pytest.raises(KeyError):
        ts.tile("buildings", "geom; DROP TABLE buildings", 16, x, y)


def test_rebuild_when_outputs_change(store, tmp_path):
    cat, path, ts = store
    built = path.stat().st_mtime_ns
    assert ensure_store(cat) == path and path.stat().st_mtime_ns == built  # cached
    csv = tmp_path / "T300" / "T300_D0.csv"
    os.utime(csv, ns=(built + 10**9, built + 10**9))
    ts.close()
    ensure_store(cat)
    assert path.stat().st_mtime_ns != built


def test_point_trees_without_size(tmp_path):
    make_outputs(tmp_path)
    trees = make_trees(tmp_path / "trees.parquet", kind="points")  # stored in EPSG:4326
    ts = _open(ensure_store(build_catalog(make_config(tmp_path, trees))))
    try:
        assert ts.meta["trees"] == {"count": 3, "sized": False, "has_height": False}
        assert ts.tile("trees", None, 16, *_tile_xy(16))
    finally:
        ts.close()


def test_quantiles_survive_a_dominant_value():
    """83 of 85 units at 0 (e.g. % meeting 3-30-300): the non-zero units still get their own classes."""
    import duckdb
    from greenpy.viz.store import N_CLASSES, metric_stats

    con = duckdb.connect()
    con.execute("CREATE TABLE units AS SELECT 0.0 AS pct_meets_3 FROM range(83) UNION ALL SELECT 7.0 UNION ALL SELECT 22.6")
    q = metric_stats(con, "units", "pct_meets_3")["quantile"]
    assert min(q) == 7.0  # 0 sits below the first break, in a class of its own; 7 starts the next one
    assert sum(b > 0 for b in q) >= 3
    assert len(q) < N_CLASSES  # the client draws len(q) + 1 classes from an N_CLASSES ramp

    # dominant but not overwhelming (~80 % zeros): two quantiles already sit above 0, and the
    # fallback must replace them rather than add to them
    con.execute("CREATE TABLE mostly AS SELECT 0.0 AS pct_meets_3 FROM range(800) "
                "UNION ALL SELECT CAST(range AS DOUBLE) + 1 FROM range(200)")
    q = metric_stats(con, "mostly", "pct_meets_3")["quantile"]
    assert len(q) < N_CLASSES and min(q) == 1.0

    # an ordinary spread keeps plain quantiles
    con.execute("CREATE TABLE spread AS SELECT CAST(range AS DOUBLE) AS pct_meets_3 FROM range(100)")
    assert metric_stats(con, "spread", "pct_meets_3")["quantile"] == [round(99 * i / N_CLASSES, 4) for i in range(1, N_CLASSES)]


def test_rule_flags_link_to_their_gradients(store):
    _, _, ts = store
    bm = {m["name"]: m for m in ts.meta["layers"]["buildings"]["metrics"]}
    assert {f: bm[f].get("gradient") for f in ("meets_3", "meets_30", "meets_300", "meets_3_30_300")} == {
        "meets_3": "tree_count_50m", "meets_30": "unit_canopy_cover",
        "meets_300": "distance_euclidean", "meets_3_30_300": "criteria_met",
    }
    assert bm["criteria_met"]["threshold"] == 3 and bm["unit_canopy_cover"]["threshold"] == 30
    # B0 meets only 300, B1 only 3, B2 all three
    assert [ts.feature("buildings", i)["criteria_met"] for i in "012"] == [1, 1, 3]
    assert ts.feature("buildings", "2")["unit_canopy_cover"] == 41.0
    stats = json.loads(ts.stats("buildings", "criteria_met"))
    assert stats["min"] == 1 and stats["max"] == 3


def test_parks_layer(store):
    _, _, ts = store
    assert ts.meta["parks"] == {"count": 2, "used": 1, "min_area_ha": 0.5}
    assert ts.has_layer("parks")
    assert ts.tile("parks", None, 16, *_tile_xy(16))


def test_level_labels_and_unit_names(tmp_path):
    make_outputs(tmp_path, merged=True)
    cfg = make_config(tmp_path, geo_level_labels={"TRACT": "Tract"}, geo_level_names={"TRACT": "TRACT_NAME"})
    ts = _open(ensure_store(build_catalog(cfg)))
    try:
        assert ts.meta["layers"]["TRACT"]["label"] == "Tract" and ts.meta["layers"]["DIST"]["label"] == "DIST"
        assert ts.feature("TRACT", "T1")["name"] == "Southbank"
        b = ts.feature("buildings", "2")
        assert b["TRACT"] == "T1" and b["name:TRACT"] == "Southbank"
    finally:
        ts.close()


def test_merge_gaps_fall_back_to_module_outputs(tmp_path):
    """A unit missing from Merge's table (e.g. no buildings) still shows T30's canopy."""
    import pandas as pd

    make_outputs(tmp_path)
    pd.DataFrame({"TRACT": ["T0"], "canopy_cover": [99.0], "pct_meets_3_30_300": [50.0]}).to_parquet(
        tmp_path / "database" / "T3_30_300_spectral.parquet", index=False
    )
    ts = _open(ensure_store(build_catalog(make_config(tmp_path))))
    try:
        assert ts.feature("TRACT", "T0")["canopy_cover"] == 99.0  # Merge's value wins where present
        assert ts.feature("TRACT", "T1")["canopy_cover"] == 41.0  # T30's where Merge has none
        tract = {m["name"]: m["module"] for m in ts.meta["layers"]["TRACT"]["metrics"]}
        assert tract["canopy_cover"] == "Merge" and list(tract).count("canopy_cover") == 1
    finally:
        ts.close()
