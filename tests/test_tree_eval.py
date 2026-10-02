"""Matching and scoring of detected trees against reference trees."""

import numpy as np
import pytest

from greenpy.optional.tree_eval import box_scores, match_points, point_scores


def test_matching_is_one_to_one_nearest_first():
    ref = [(0, 0), (10, 0)]
    det = [(1, 0), (0.5, 0), (10, 2.5)]
    d, r, dist = match_points(det, ref, radius=3)
    assert dict(zip(r.tolist(), d.tolist())) == {0: 1, 1: 2}  # the nearer of the two candidates wins ref 0
    np.testing.assert_allclose(sorted(dist), [0.5, 2.5])


def test_per_reference_radius():
    ref = [(0, 0), (20, 0)]
    det = [(4, 0), (24, 0)]
    _, r, _ = match_points(det, ref, radius=[3, 5])  # only the second tree's spread reaches its detection
    assert r.tolist() == [1]


def test_point_scores_complete_and_partial_reference():
    ref = np.array([(0, 0), (10, 0), (20, 0), (30, 0)])
    det = np.array([(0, 1), (10, 1), (50, 0)])
    partial = point_scores(det, ref, 2)
    assert partial["recall"] == 0.5 and "precision" not in partial
    full = point_scores(det, ref, 2, complete_reference=True)
    assert full["precision"] == pytest.approx(2 / 3) and full["f1"] == pytest.approx(4 / 7)


def test_height_and_crown_errors():
    s = point_scores(
        [(0, 0), (10, 0)], [(0, 0), (10, 0)], 1,
        det_height=[12, 8], ref_height=[10, 10],
        det_area=[np.pi * 9, np.pi * 4], ref_spread=[6, 6],
    )
    assert s["height_bias"] == pytest.approx(0) and s["height_rmse"] == pytest.approx(2)
    assert s["crown_diam_bias"] == pytest.approx(-1)  # diameters 6 and 4 vs 6


def test_box_scores():
    gt = [(0, 0, 10, 10), (20, 0, 30, 10)]
    pred = [(1, 1, 11, 11), (40, 0, 50, 10)]
    s = box_scores(pred, gt, iou_threshold=0.4)
    assert (s["tp"], s["precision"], s["recall"]) == (1, 0.5, 0.5)
    assert box_scores([], gt)["recall"] == 0
