"""
Vector engine: exact line of sight against footprint and crown prisms, in Sedona SQL.

The reference for the raster engine on small areas. Observers, targets and
candidate pairs are the shared definitions (observers / targets / pairs):
every front-facing facade point of a building is joined to every target
point of each candidate tree. Each sightline is intersected with every
obstacle — all buildings, including the observer's own, and every tree crown
except the target's — modelled as flat-topped prisms from the ground. For a
crossing over fractions [f_lo, f_hi] of the sightline (crowns clipped by the
target's crown, inside which only buildings block), an obstacle of height
h needs an eye height above max over {f_lo, f_hi} of (h - f*zt) / (1 - f);
the sightline needs the maximum over its crossings, the pair the minimum
over its sightlines. Ground is flat (terrain is raster-engine only); skip
metres at both ends are ignored.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from loguru import logger
from pyspark.sql.session import SparkSession

from ...config.schema import GreenPyConfig
from ...utils.data_processing import view_suffix
from .inputs import VisibilityInputs
from .observers import facade_points, floor_eyes
from .pairs import building_counts, eligible_pairs
from .params import VisibilityParams
from .raster_engine import CROWN_MARGIN, OUTPUT_COLUMNS, load_trees, target_mask
from .targets import crown_polygons, tree_targets

# Inputs are loaded once per (config, overlay) and reused across geo codes
_INPUTS: dict = {}
_NO_OBSTACLE = -1e30


def _inputs(cfg: GreenPyConfig, overlay_path: Path | None) -> VisibilityInputs:
    key = (cfg.output.base_dir, str(overlay_path))
    if key not in _INPUTS:
        _INPUTS[key] = VisibilityInputs.load(cfg, overlay_path)
    return _INPUTS[key]


def sightlines(fp, targets, pair_b, pair_t) -> pd.DataFrame:
    """One row per (pair, front-facing facade point, target point): pair, target tree, endpoints."""
    rows = []
    for p, (b, t) in enumerate(zip(pair_b, pair_t)):
        ko = np.arange(fp.start[b], fp.start[b] + fp.count[b])
        kt = np.arange(targets.start[t], targets.start[t] + targets.count[t])
        o, g = np.repeat(ko, len(kt)), np.tile(kt, len(ko))
        front = fp.nx[o] * (targets.x[g] - fp.x[o]) + fp.ny[o] * (targets.y[g] - fp.y[o]) > 0
        o, g = o[front], g[front]
        rows.append(pd.DataFrame({
            "pair_id": p, "t_id": t, "ox": fp.x[o], "oy": fp.y[o],
            "tx": targets.x[g], "ty": targets.y[g], "tz": targets.z[g],
        }))
    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["pair_id", "t_id", "ox", "oy", "tx", "ty", "tz"])
    df.insert(0, "ray_id", np.arange(len(df)))
    df["len"] = np.hypot(df["tx"] - df["ox"], df["ty"] - df["oy"])
    return df


def pair_zreq_sql(sfx: str, skip: float) -> str:
    """Lowest visible eye height per pair (NULL: no front-facing sightline, -1e30: unobstructed)."""
    return f"""
    WITH hits AS (
        SELECT r.ray_id, r.ox, r.oy, r.tx, r.ty, r.tz, r.len, o.h,
               CASE WHEN o.kind = 't' THEN ST_Difference(ST_Intersection(r.geometry, o.geometry), c.geometry)
                    ELSE ST_Intersection(r.geometry, o.geometry) END AS inter
        FROM vis_rays_{sfx} r
        JOIN vis_obstacles_{sfx} o ON ST_Intersects(r.geometry, o.geometry)
        JOIN vis_crowns_{sfx} c ON c.t_id = r.t_id
        WHERE r.len > {2 * skip}
          AND NOT (o.kind = 't' AND o.idx = r.t_id)
    ),
    crossings AS (
        SELECT ray_id, tz, h,
               GREATEST(ST_Distance(ST_Point(ox, oy), inter) / len, {skip} / len) AS f_lo,
               LEAST(1.0D - ST_Distance(ST_Point(tx, ty), inter) / len, 1.0D - {skip} / len) AS f_hi
        FROM hits
        WHERE NOT ST_IsEmpty(inter)
    ),
    ray_z AS (
        SELECT ray_id, MAX(GREATEST((h - f_lo * tz) / (1.0D - f_lo), (h - f_hi * tz) / (1.0D - f_hi))) AS zreq
        FROM crossings WHERE f_lo <= f_hi
        GROUP BY ray_id
    )
    SELECT r.pair_id, MIN(COALESCE(z.zreq, CAST({_NO_OBSTACLE} AS DOUBLE))) AS zreq
    FROM vis_rays_{sfx} r LEFT JOIN ray_z z ON r.ray_id = z.ray_id
    GROUP BY r.pair_id
    """


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    sub_geo_level: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    buffer: int = 100,
    tree_area: int = 10,
    tree_height: int = 3,
    overwrite: bool = True,
    overlay_path: Path | None = None,
) -> pd.DataFrame | None:
    """Visible trees per building of one geo_code with exact geometry -> Visibility_<geo_code>_<buffer>m.csv."""
    start = time.time()
    out_path = Path(output_dir) / f"Visibility_{geo_code}_{buffer}m.csv"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)
    sfx = view_suffix(geo_code)
    params = VisibilityParams.from_cfg(cfg, buffer, tree_area, tree_height)
    if cfg.terrain.source is not None:
        logger.warning(f"Visibility (vector): terrain ({cfg.terrain.source}) is modelled by the raster engine only — "
                       "the vector engine assumes flat ground")
    try:
        inputs = _inputs(cfg, overlay_path)
        idx = inputs.owned(geo_level, geo_code)
        if len(idx) == 0:
            logger.warning(f"Visibility (vector): no buildings in {geo_code}")
            return None
        geoms = inputs.geoms[idx]
        minx, miny, maxx, maxy = shapely.total_bounds(geoms)
        pad = params.buffer + CROWN_MARGIN
        bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        near = inputs.tree.query(shapely.box(*bounds))
        trees = load_trees(cfg, bounds, params)
        targets = tree_targets(trees[target_mask(trees, params)], params.crown_points, params.crown_point_height)
        target_rows = np.flatnonzero(target_mask(trees, params))

        fp = facade_points(geoms, params.facade_spacing, params.facade_offset,
                           blockers=shapely.STRtree(inputs.geoms[near]))
        pair_b, pair_t = eligible_pairs(geoms, targets.ref_x, targets.ref_y, params.buffer)
        rays = sightlines(fp, targets, pair_b, pair_t)

        # obstacles: buildings (b, index) and every crown (t, target index or -1 for non-targets)
        tree_idx = np.full(len(trees), -1)
        tree_idx[target_rows] = np.arange(len(target_rows))
        obstacles = pd.concat([
            pd.DataFrame({"kind": "b", "idx": near, "h": inputs.height[near],
                          "wkb": shapely.to_wkb(inputs.geoms[near])}),
            pd.DataFrame({"kind": "t", "idx": tree_idx, "h": trees["tree_height"].to_numpy(dtype=float),
                          "wkb": shapely.to_wkb(crown_polygons(trees))}),
        ], ignore_index=True)

        sedona.createDataFrame(obstacles).selectExpr(
            "kind", "CAST(idx AS BIGINT) AS idx", "CAST(h AS DOUBLE) AS h", "ST_GeomFromWKB(wkb) AS geometry"
        ).createOrReplaceTempView(f"vis_obstacles_{sfx}")
        sedona.createDataFrame(pd.DataFrame({"t_id": np.arange(len(targets)), "wkb": shapely.to_wkb(targets.crowns)})) \
            .selectExpr("CAST(t_id AS BIGINT) AS t_id", "ST_GeomFromWKB(wkb) AS geometry") \
            .createOrReplaceTempView(f"vis_crowns_{sfx}")
        sedona.createDataFrame(rays).selectExpr(
            "CAST(ray_id AS BIGINT) AS ray_id", "CAST(pair_id AS BIGINT) AS pair_id", "CAST(t_id AS BIGINT) AS t_id",
            "ox", "oy", "tx", "ty", "tz", "len", "ST_MakeLine(ST_Point(ox, oy), ST_Point(tx, ty)) AS geometry",
        ).createOrReplaceTempView(f"vis_rays_{sfx}")

        z = sedona.sql(pair_zreq_sql(sfx, params.skip)).toPandas()
        z_req = np.full(len(pair_b), np.inf)
        z_req[z["pair_id"].to_numpy(dtype=np.int64)] = z["zreq"].to_numpy(dtype=float)

        n_floors, z_top = floor_eyes(inputs.height[idx], params.storey_height, params.eye_height)
        counts = building_counts(len(idx), pair_b, z_req, z_top, params.eye_height)
        counts.insert(0, "building_id", inputs.building_id[idx])
        counts["n_floors"] = n_floors
        counts["building_height"] = inputs.height[idx]
        counts["height_source"] = inputs.height_source[idx]
        sub = inputs.overlay[["building_id", sub_geo_level]].drop_duplicates("building_id")
        result = counts[OUTPUT_COLUMNS].merge(sub, on="building_id", how="left")
        result.to_csv(out_path, index=False)
        logger.info(
            f"Visibility (vector): {geo_code} — {len(result)} buildings, {len(pair_b)} pairs, {len(rays)} sightlines "
            f"in {time.time() - start:.1f}s"
        )
        return result
    except Exception:
        logger.exception(f"Visibility (vector): error processing {geo_code}")
        return None
    finally:
        for name in (f"vis_rays_{sfx}", f"vis_obstacles_{sfx}", f"vis_crowns_{sfx}"):
            sedona.catalog.dropTempView(name)
