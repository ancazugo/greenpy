"""
Accuracy of segmented trees against reference trees (optional module).

Detections are matched one-to-one to reference trees, nearest pairs first,
within a radius per reference tree (e.g. max(3 m, half the surveyed crown
spread)). Recall is meaningful for any survey; precision only where the
reference is complete — council inventories omit private trees, so an
unmatched detection in a garden is not an error there.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def match_points(det_xy: np.ndarray, ref_xy: np.ndarray, radius) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Greedy one-to-one matching by distance; pair (d, r) allowed when dist <= radius[r].

    Returns matched detection indices, reference indices and distances.
    """
    det_xy, ref_xy = np.asarray(det_xy, float), np.asarray(ref_xy, float)
    radius = np.broadcast_to(np.asarray(radius, float), (len(ref_xy),))
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0))
    if len(det_xy) == 0 or len(ref_xy) == 0:
        return empty
    pairs = cKDTree(ref_xy).sparse_distance_matrix(cKDTree(det_xy), float(radius.max()), output_type="coo_matrix")
    r, d, dist = pairs.row, pairs.col, pairs.data
    keep = dist <= radius[r]
    r, d, dist = r[keep], d[keep], dist[keep]
    order = np.argsort(dist, kind="stable")
    used_r = np.zeros(len(ref_xy), bool)
    used_d = np.zeros(len(det_xy), bool)
    out_d, out_r, out_dist = [], [], []
    for i in order:
        if not used_r[r[i]] and not used_d[d[i]]:
            used_r[r[i]] = used_d[d[i]] = True
            out_d.append(d[i]); out_r.append(r[i]); out_dist.append(dist[i])
    if not out_d:
        return empty
    return np.array(out_d), np.array(out_r), np.array(out_dist)


def point_scores(
    det_xy, ref_xy, radius,
    det_height=None, ref_height=None, det_area=None, ref_spread=None,
    complete_reference: bool = False,
) -> dict:
    """Recall (and precision/F1 when complete_reference) plus height and crown-size errors of matches.

    Crown size compares the detected crown's equivalent diameter
    2*sqrt(area/pi) with the surveyed spread.
    """
    di, ri, dist = match_points(det_xy, ref_xy, radius)
    n_det, n_ref, n = len(det_xy), len(ref_xy), len(di)
    out = {"n_det": n_det, "n_ref": n_ref, "matched": n, "recall": n / n_ref if n_ref else np.nan,
           "mean_dist": float(dist.mean()) if n else np.nan}
    if complete_reference:
        precision = n / n_det if n_det else np.nan
        out["precision"] = precision
        out["f1"] = 2 * n / (n_det + n_ref) if (n_det + n_ref) else np.nan
    if det_height is not None and ref_height is not None and n:
        err = np.asarray(det_height, float)[di] - np.asarray(ref_height, float)[ri]
        err = err[np.isfinite(err)]
        out["height_bias"] = float(err.mean()) if err.size else np.nan
        out["height_rmse"] = float(np.sqrt((err**2).mean())) if err.size else np.nan
    if det_area is not None and ref_spread is not None and n:
        diam = 2 * np.sqrt(np.asarray(det_area, float)[di] / np.pi)
        err = diam - np.asarray(ref_spread, float)[ri]
        err = err[np.isfinite(err)]
        out["crown_diam_bias"] = float(err.mean()) if err.size else np.nan
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix of axis-aligned boxes given as (xmin, ymin, xmax, ymax) rows."""
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    iw = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])
    return inter / (area(a) + area(b) - inter)


def box_scores(pred_boxes, gt_boxes, iou_threshold: float = 0.4) -> dict:
    """Precision/recall of crown boxes, one-to-one by maximum IoU (NeonTreeEvaluation convention)."""
    pred_boxes, gt_boxes = np.asarray(pred_boxes, float).reshape(-1, 4), np.asarray(gt_boxes, float).reshape(-1, 4)
    n_pred, n_gt = len(pred_boxes), len(gt_boxes)
    tp = 0
    mean_iou = np.nan
    if n_pred and n_gt:
        iou = box_iou(pred_boxes, gt_boxes)
        pi, gi = linear_sum_assignment(-iou)
        hits = iou[pi, gi] >= iou_threshold
        tp = int(hits.sum())
        mean_iou = float(iou[pi, gi][hits].mean()) if tp else np.nan
    return {"n_pred": n_pred, "n_gt": n_gt, "tp": tp,
            "precision": tp / n_pred if n_pred else np.nan,
            "recall": tp / n_gt if n_gt else np.nan,
            "mean_iou": mean_iou}
