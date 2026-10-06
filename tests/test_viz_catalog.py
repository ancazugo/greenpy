import shutil

import pytest

from greenpy.viz.catalog import build_catalog, describe

from viz_fixtures import make_config, make_outputs, make_trees


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
    ("visible_trees_50m", "count", 3, "high"),
    ("NDVI", "value", None, None),
])
def test_describe(name, kind, threshold, better):
    m = describe(name)
    assert (m.kind, m.threshold, m.better) == (kind, threshold, better)


def test_tree_directory_with_parquet_and_gpkg(tmp_path):
    """A trees_dir is read like T3 reads it: .gpkg tiles and the .parquet files written by -p Trees."""
    make_outputs(tmp_path)
    trees_dir = tmp_path / "trees"
    trees_dir.mkdir()
    make_trees(trees_dir / "trees_A.parquet", kind="points")
    make_trees(trees_dir / "legacy.gpkg")
    (trees_dir / "notes.txt").write_text("not a tree file")
    cat = build_catalog(make_config(tmp_path, trees_dir))
    assert [p.name for p in cat.trees.paths] == ["legacy.gpkg", "trees_A.parquet"]


def test_grid_runs_sit_beside_census_runs(tmp_path):
    """H3 runs of T30/Tree_count/Merge write grid-suffixed outputs that feed the grid layer only."""
    import geopandas as gpd
    import pandas as pd
    import shapely

    make_outputs(tmp_path, merged=True)
    db = tmp_path / "database"
    gpd.GeoDataFrame(
        {"h3_9": ["c0", "c1"], "DIST": ["D0", "D0"], "TRACT": ["T0", "T1"]},
        geometry=[shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)], crs=27700,
    ).to_parquet(db / "h3_boundaries_res9.parquet", index=False)
    pd.DataFrame({"building_id": [0, 1, 2], "DIST": ["D0"] * 3, "TRACT": ["T0", "T0", "T1"], "h3_9": ["c0", "c0", "c1"]}) \
        .to_parquet(db / "h3_buildings_overlay_res9.parquet", index=False)
    for module, cols in (("T30", {"canopy_cover": [5.0, 60.0], "total_pixels": [100, 100]}), ("Tree_count", {"tree_count": [1, 7]})):
        (tmp_path / f"{module}_h3_9").mkdir()
        pd.DataFrame({"h3_9": ["c0", "c1"], **cols}).to_csv(tmp_path / f"{module}_h3_9" / f"{module}_D0.csv", index=False)
    pd.DataFrame({"h3_9": ["c0", "c1"], "total_trees": [1, 7], "pct_meets_3_30_300": [0.0, 100.0]}) \
        .to_parquet(db / "T3_30_300_spectral_h3_9.parquet", index=False)
    pd.DataFrame({"building_id": [0, 1, 2], "meets_3_30_300": [True, True, True]}) \
        .to_parquet(db / "T3_30_300_buildings_h3_9.parquet", index=False)

    cat = build_catalog(make_config(tmp_path))
    layers = {u.name: u for u in cat.unit_layers}
    assert list(layers) == ["DIST", "TRACT", "h3_9"]
    grid = layers["h3_9"]
    assert grid.merged and grid.label == "H3 res 9"
    by_module = {s.module: s for s in grid.sources}
    assert by_module["Merge"].path.endswith("T3_30_300_spectral_h3_9.parquet")
    assert by_module["T30"].path.endswith("T30_h3_9/*.csv") and by_module["T30"].columns == {"canopy_cover": "canopy_cover"}
    assert by_module["Tree_count"].zero_fill == ["D0"]
    # the census layers keep their own outputs, and building flags stay the census-unit rule
    assert {s.path for s in layers["TRACT"].sources} == {str(tmp_path / "T30" / "*.csv"), str(tmp_path / "Tree_count" / "*.csv")}
    rule = next(s for s in cat.building_sources if s.module == "Rule")
    assert rule.path.endswith("T3_30_300_buildings.parquet")


def test_visibility_and_height_metrics():
    from greenpy.viz.catalog import describe

    m = describe("visible_trees_ground_50m")
    assert m.kind == "count" and m.threshold == 3 and m.module == "Visibility"
    assert describe("visible_trees_50m").threshold == 3
    assert describe("building_height").module == "Heights"
    assert describe("meets_3_visibility").kind == "boolean"
    assert describe("pct_meets_3_visibility").label == "% of buildings meeting 3 (trees in view)"


def test_rule_gradients_follow_merge_metadata(tmp_path):
    import pandas as pd

    from greenpy.merge import write_rule_parquet
    from greenpy.viz.catalog import _rule_gradients

    df = pd.DataFrame({"building_id": [1], "tree_count_50m": [4], "visible_trees_50m": [1],
                       "distance_euclidean": [10.0], "meets_3": [False], "meets_3_30_300": [False]})
    write_rule_parquet(df, tmp_path / "T3_30_300_buildings.parquet", {"t3_metric": "visibility", "t3_col": "visible_trees_50m"})
    g = _rule_gradients(tmp_path)
    assert g["meets_3"] == "visible_trees_50m"
    assert g["meets_3_proximity"] == "tree_count_50m" and g["meets_3_visibility"] == "visible_trees_50m"
    # tables written before the metadata existed fall back to the T3 column
    df.to_parquet(tmp_path / "T3_30_300_buildings.parquet")
    assert _rule_gradients(tmp_path)["meets_3"] == "tree_count_50m"


def test_per_metric_rule_labels():
    assert describe("meets_3_30_300_visibility").label == "Meets 3-30-300 with 3 = trees in view"
    assert describe("pct_meets_3_30_300_proximity").label == "% of buildings meeting 3-30-300 with 3 = trees nearby"
    assert describe("pct_meets_3_visibility").note and describe("meets_3").note is None
    m = describe("share_visible_100m")
    assert (m.kind, m.module, m.threshold, m.better) == ("percent", "Visibility", None, "high")
    assert describe("mean_share_visible_50m").label.startswith("Mean trees within 50 m")
    assert "pairs" in describe("share_visible_50m", "Merge").note and "pairs" not in m.note
    assert describe("criteria_met_visibility").threshold == 3
