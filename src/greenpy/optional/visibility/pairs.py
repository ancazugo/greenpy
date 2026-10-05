"""Building–tree pairs to test, and per-building visible-tree counts."""

import numpy as np
import pandas as pd
import shapely


def eligible_pairs(footprints: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray, buffer: float) -> tuple[np.ndarray, np.ndarray]:
    """(building index, tree index) for trees whose crown centroid is within `buffer` m of the footprint.

    The same rule as T3 (a tree counts when its centroid falls in the buffered
    footprint), so every building's candidates are its T3 trees.
    """
    if len(ref_x) == 0 or len(footprints) == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    tree = shapely.STRtree(shapely.points(ref_x, ref_y))
    b, t = tree.query(footprints, predicate="dwithin", distance=buffer)
    order = np.lexsort((t, b))
    return b[order].astype(np.int64), t[order].astype(np.int64)


def building_counts(
    n_buildings: int, pair_b: np.ndarray, z_req: np.ndarray, z_top: np.ndarray, eye: float
) -> pd.DataFrame:
    """Per building: visible_trees (seen from some floor), visible_trees_ground (from the ground floor), candidate_trees.

    z_req is the lowest eye height from which each pair's tree is visible
    (inf when never); a tree is visible from a floor whose eye is above it.
    """
    visible = z_req < z_top[pair_b]
    ground = z_req < eye
    return pd.DataFrame({
        "visible_trees": np.bincount(pair_b[visible], minlength=n_buildings),
        "visible_trees_ground": np.bincount(pair_b[ground], minlength=n_buildings),
        "candidate_trees": np.bincount(pair_b, minlength=n_buildings),
    })
