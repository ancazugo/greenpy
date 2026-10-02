import shutil

import pytest

from greenpy.viz.catalog import build_catalog, describe

from viz_fixtures import make_config, make_outputs


def _building_metrics(cat):
    return {m for s in cat.building_sources for m in s.columns.values()}


def test_csv_only_outputs(tmp_path):
    make_outputs(tmp_path)
    cat = build_catalog(make_config(tmp_path))

    assert _building_metrics(cat) == {"tree_count_10m", "tree_count_50m", "distance_euclidean", "distance_manhattan"}
    assert all(s.fmt == "csv" for s in cat.building_sources)
    layers = {u.name: u for u in cat.unit_layers}
    assert list(layers) == ["DIST", "TRACT"]
    assert {m for s in layers["TRACT"].sources for m in s.columns.values()} == {"canopy_cover", "tree_count"}
    assert layers["DIST"].sources == []
    assert not layers["TRACT"].merged
    assert cat.trees is None


def test_merge_outputs_preferred(tmp_path):
    make_outputs(tmp_path, merged=True)
    cat = build_catalog(make_config(tmp_path))

    by_metric = {m: s for s in cat.building_sources for m in s.columns.values()}
    assert by_metric["tree_count_50m"].fmt == "parquet"
    assert by_metric["tree_count_10m"].fmt == "csv"  # Merge didn't consolidate this buffer
    assert {"meets_3", "meets_30", "meets_300", "meets_3_30_300"} <= set(by_metric)

    dist = next(u for u in cat.unit_layers if u.name == "DIST")
    assert dist.merged
    assert set(dist.sources[0].columns) == {"total_trees", "tree_count_50m", "canopy_cover", "park_distance_euclidean", "pct_meets_3_30_300"}


def test_missing_buildings_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="run at least one greenpy module"):
        build_catalog(make_config(tmp_path))


def test_dggs_layer_discovered(tmp_path):
    make_outputs(tmp_path)
    db = tmp_path / "database"
    shutil.copy(db / "census_boundaries.parquet", db / "h3_boundaries_res9.parquet")
    cat = build_catalog(make_config(tmp_path))
    h3 = next(u for u in cat.unit_layers if u.name == "h3_9")
    assert h3.label == "H3 res 9" and h3.overlay is None


@pytest.mark.parametrize("name, kind, threshold, better", [
    ("tree_count_50m", "count", 3, "high"),
    ("mean_tree_count_50m", "count", 3, "high"),
    ("distance_euclidean", "distance", 300, "low"),
    ("park_distance_manhattan", "distance", 300, "low"),
    ("canopy_cover", "percent", 30, "high"),
    ("building_canopy_cover_25m", "percent", 30, "high"),
    ("meets_3_30_300", "boolean", None, "high"),
    ("pct_meets_300", "percent", None, "high"),
    ("visible_trees_50m", "count", None, "high"),
    ("NDVI", "value", None, None),
])
def test_describe(name, kind, threshold, better):
    m = describe(name)
    assert (m.kind, m.threshold, m.better) == (kind, threshold, better)
