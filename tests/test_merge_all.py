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
