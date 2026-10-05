"""Merge's unit table: every unit with canopy, trees or buildings gets a row (local Spark, no Sedona)."""

import os

import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from greenpy.merge import fill_unit_counts, merge_all


@pytest.fixture(scope="module")
def spark():
    from greenpy.utils.constants import JAVA_HOME
    from pyspark.sql import SparkSession

    if JAVA_HOME:
        os.environ["JAVA_HOME"] = JAVA_HOME
    s = (SparkSession.builder.master("local[1]").appName("greenpy-merge-test")
         .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "1").getOrCreate())
    yield s
    s.stop()


def _view(spark, name, df):
    spark.createDataFrame(df).createOrReplaceTempView(name)


def test_units_without_buildings_keep_canopy_and_trees(spark):
    # cells A (buildings) and B (a forest: canopy and trees, no buildings); C has nothing
    _view(spark, "boundaries", pd.DataFrame({"cell": ["A", "B", "C"]}))
    _view(spark, "t3_300_agg", pd.DataFrame({"cell": ["A"], "tree_count_50m": [4.0], "park_distance_euclidean": [120.0]}))
    _view(spark, "t30_spectral", pd.DataFrame({"cell": ["A", "B"], "canopy_cover": [12.0, 97.1]}))
    _view(spark, "tree_count_agg", pd.DataFrame({"cell": ["B"], "total_trees": [2616]}))
    _view(spark, "compliance_agg", pd.DataFrame({"cell": ["A"], "pct_meets_3_30_300": [25.0]}))

    df = merge_all(spark, "cell").toPandas().set_index("cell")
    assert sorted(df.index) == ["A", "B"]  # C has no data at all
    assert df.loc["B", "canopy_cover"] == 97.1 and df.loc["B", "total_trees"] == 2616
    assert pd.isna(df.loc["B", "pct_meets_3_30_300"]) and pd.isna(df.loc["B", "tree_count_50m"])
    assert bool(df.loc["A", "has_buildings"]) and not bool(df.loc["B", "has_buildings"])


def test_fill_unit_counts():
    df = pd.DataFrame({
        "cell": ["A", "B"], "tree_count_50m": [None, None], "total_trees": [None, 2616], "has_buildings": [True, False],
    })
    out = fill_unit_counts(df, [50]).set_index("cell")
    assert "has_buildings" not in out.columns
    assert out.loc["A", "tree_count_50m"] == 0 and out.loc["A", "total_trees"] == 0  # buildings, no trees
    assert pd.isna(out.loc["B", "tree_count_50m"]) and out.loc["B", "total_trees"] == 2616


def _compliance_views(spark, with_visibility=True):
    _view(spark, "t3_300", pd.DataFrame({
        "building_id": ["a", "b", "c"], "tree_count_50m": [5, 3, 0], "distance_euclidean": [100.0, 100.0, 100.0],
    }))
    _view(spark, "boundaries_buildings_overlay", pd.DataFrame({"building_id": ["a", "b", "c"], "unit": ["U", "U", "U"]}))
    _view(spark, "t30", pd.DataFrame({"unit": ["U"], "canopy_cover": [40.0]}))
    if with_visibility:
        _view(spark, "visibility_50m", pd.DataFrame({
            "building_id": ["a", "b", "c"], "visible_trees": [4, 1, 0], "visible_trees_ground": [2, 0, 0],
        }))


def _cfg(tmp_path):
    from greenpy.config.schema import ColumnMapping, DataPaths, GreenPyConfig, OutputPaths
    (tmp_path / "database").mkdir(exist_ok=True)
    return GreenPyConfig(study_area_name="t", crs="EPSG:32632", data=DataPaths("b", "p", "a", "r", "c"),
                         columns=ColumnMapping(geo_levels=["unit"]), output=OutputPaths(str(tmp_path)))


@pytest.mark.parametrize("metric, expected", [("proximity", [True, True, False]), ("visibility", [True, False, False])])
def test_compliance_rule_t3_metric(spark, tmp_path, metric, expected):
    from greenpy.merge import compute_compliance, read_rule_metadata

    _compliance_views(spark)
    compute_compliance(spark, _cfg(tmp_path), "unit", "unit", [50], [], rule_t3_buffer=50,
                       rule_t3_metric=metric, visibility_buffers=[50])
    path = tmp_path / "database" / "T3_30_300_buildings.parquet"
    out = pd.read_parquet(path).set_index("building_id").sort_index()
    assert out["meets_3"].tolist() == expected
    assert out["meets_3_proximity"].tolist() == [True, True, False]
    assert out["meets_3_visibility"].tolist() == [True, False, False]
    assert out["visible_trees_ground_50m"].tolist() == [2, 0, 0]
    meta = read_rule_metadata(path)
    assert meta["t3_metric"] == metric
    assert meta["t3_col"] == ("visible_trees_50m" if metric == "visibility" else "tree_count_50m")
    agg = spark.table("compliance_agg").toPandas()
    assert {"pct_meets_3_proximity", "pct_meets_3_visibility", "pct_meets_3_30_300_proximity",
            "pct_meets_3_30_300_visibility"} <= set(agg.columns)
    assert out["meets_3_30_300_proximity"].tolist() == [True, True, False]  # unit canopy 40 %, parks at 100 m
    assert out["meets_3_30_300_visibility"].tolist() == [True, False, False]


def test_compliance_visibility_requires_output(spark, tmp_path):
    from greenpy.merge import compute_compliance

    _compliance_views(spark, with_visibility=False)
    with pytest.raises(FileNotFoundError, match="Visibility"):
        compute_compliance(spark, _cfg(tmp_path), "unit", "unit", [50], [], rule_t3_buffer=50,
                           rule_t3_metric="visibility", visibility_buffers=[])


def test_aggregate_visibility_share_in_view(spark):
    from greenpy.merge import aggregate_visibility

    _view(spark, "boundaries_buildings_overlay", pd.DataFrame({"building_id": ["a", "b", "c"], "unit": ["U", "U", "V"]}))
    _view(spark, "visibility_50m", pd.DataFrame({
        "building_id": ["a", "b", "c"], "visible_trees": [4, 1, 0], "visible_trees_ground": [2, 0, 0],
        "candidate_trees": [5, 5, 0],
    }))
    out = aggregate_visibility(spark, "unit", [50]).toPandas().set_index("unit")
    # pairs, not buildings: (4 + 1) of (5 + 5) trees nearby are in view; V has no nearby trees
    assert out.loc["U", "share_visible_50m"] == 50.0 and pd.isna(out.loc["V", "share_visible_50m"])
    assert out.loc["U", "visible_trees_50m"] == 2.5
