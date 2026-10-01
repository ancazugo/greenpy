"""Merge helpers that don't need Spark: buffer discovery and 3-30-300 rule evaluation."""

import numpy as np
import pandas as pd

from greenpy.merge import _buffers_in, evaluate_rule, summarise_rule


def test_buffers_in_parses_radii_without_collisions(tmp_path):
    for name in ["T30_buildings_D1_50m.csv", "T30_buildings_D1_150m.csv", "T30_buildings_D2_0m.csv",
                 "Visibility_D1_50m.csv", "T30_buildings_notes.txt"]:
        (tmp_path / name).write_text("x")
    assert _buffers_in(tmp_path, "T30_buildings", "csv") == [0, 50, 150]
    assert _buffers_in(tmp_path, "Visibility", "csv") == [50]
    assert _buffers_in(tmp_path / "missing", "Visibility", "csv") == []


def _buildings():
    return pd.DataFrame({
        "building_id": ["a", "b", "c", "d", "e"],
        "LAD": ["L1", "L1", "L1", "L2", "L2"],
        "tree_count_50m": [3, 2, 5, 0, 10],
        "canopy_cover": [30.0, 45.0, 10.0, 31.0, np.nan],
        "distance_euclidean": [300.0, 120.0, 50.0, 301.0, 10.0],
    })


def test_evaluate_rule_thresholds_are_inclusive():
    out = evaluate_rule(_buildings(), "tree_count_50m", "canopy_cover", "distance_euclidean")
    assert out["meets_3"].tolist() == [True, False, True, False, True]
    assert out["meets_30"].tolist()[:4] == [True, True, False, True]
    assert out["meets_30"].isna().tolist() == [False, False, False, False, True]
    assert out["meets_300"].tolist() == [True, True, True, False, True]


def test_combined_rule_null_unless_all_known():
    out = evaluate_rule(_buildings(), "tree_count_50m", "canopy_cover", "distance_euclidean")
    # e fails nothing known but canopy is missing -> unknown; d fails -> False
    assert out["meets_3_30_300"].isna().tolist() == [False, False, False, False, True]
    assert out["meets_3_30_300"].tolist()[:4] == [True, False, False, False]


def test_missing_criterion_column_is_null():
    out = evaluate_rule(_buildings(), None, "canopy_cover", "distance_euclidean")
    assert out["meets_3"].isna().all() and out["meets_3_30_300"].isna().all()


def test_summarise_rule_percentages_over_known_buildings():
    out = evaluate_rule(_buildings(), "tree_count_50m", "canopy_cover", "distance_euclidean")
    summary = summarise_rule(out, "LAD").set_index("LAD")
    assert summary.loc["L1", "pct_meets_3"] == 66.67
    assert summary.loc["L1", "pct_meets_3_30_300"] == 33.33
    # L2: canopy known only for d (31 %) -> 100 %; combined known only for d (fails) -> 0 %
    assert summary.loc["L2", "pct_meets_30"] == 100.0
    assert summary.loc["L2", "pct_meets_3_30_300"] == 0.0
