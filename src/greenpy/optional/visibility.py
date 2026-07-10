"""
Visibility module: 2.5D line-of-sight from buildings to trees.

For each building–tree pair within `buffer` metres, 9 sightlines are evaluated
from 3 observer levels on the building (bottom z=0, middle z=H/2, top z=H) to
3 target levels on the tree (z=0, h/2, h). A sightline is blocked when another
building footprint or tree canopy polygon crosses it and that obstacle's
height reaches the sightline's interpolated height at the crossing.

Model assumptions: buildings are flat-topped prisms, trees are solid
ground-to-crown prisms (sightlines under a canopy count as blocked), terrain
is flat (shared ground level z=0), and grazing contact counts as blocked.
Buildings or trees with missing height are skipped entirely — as observers,
targets and obstacles — so complete height data is expected.
"""

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.session import SparkSession

from ..config.schema import GreenPyConfig
from ..utils.data_processing import (
    drop_geo_views,
    get_geometries,
    load_trees_gdf,
    view_suffix,
)

# Observer/target levels as SQL expressions of a height column
_LEVELS = {"bottom": "0.0D", "middle": "{h} / 2.0D", "top": "{h}"}

# Sightline endpoints can graze a neighbouring polygon (facade point on the
# observer's own wall touching an adjacent footprint, target centroid touched
# by an overlapping canopy). Crossings shorter than this are ignored (metres).
_ENDPOINT_EPSILON = 0.05


def _register_vis_buildings(sedona: SparkSession, geo_code: str, buffer: int) -> None:
    """Register vis_buildings_<sfx> (obstacles, buffered extent) and vis_observers_<sfx>.

    Obstacle buildings are clipped to the boundary buffered by `buffer` metres so
    buildings just outside the study area still occlude; observers are the subset
    intersecting the unbuffered boundary. Buildings with missing or non-positive
    height are dropped from both, with a warning.
    """
    sfx = view_suffix(geo_code)

    counts = sedona.sql(
        f"""
        SELECT COUNT(*) AS n_all,
               SUM(CASE WHEN b.building_height IS NULL OR b.building_height <= 0 THEN 1 ELSE 0 END) AS n_bad
        FROM buildings b, geo_boundary_{sfx} g
        WHERE ST_Intersects(b.geometry, ST_Buffer(g.geometry, {buffer}))
        """
    ).collect()[0]
    if counts["n_bad"]:
        logger.warning(
            f"Visibility: skipping {counts['n_bad']}/{counts['n_all']} buildings with missing or "
            f"invalid height in {geo_code} — complete building height data is expected"
        )

    sedona.sql(
        f"""
        SELECT b.building_id, b.geometry, CAST(b.building_height AS DOUBLE) AS building_height
        FROM buildings b, geo_boundary_{sfx} g
        WHERE ST_Intersects(b.geometry, ST_Buffer(g.geometry, {buffer}))
          AND b.building_height IS NOT NULL AND b.building_height > 0
        """
    ).createOrReplaceTempView(f"vis_buildings_{sfx}")

    sedona.sql(
        f"""
        SELECT b.* FROM vis_buildings_{sfx} b, geo_boundary_{sfx} g
        WHERE ST_Intersects(b.geometry, g.geometry)
        """
    ).createOrReplaceTempView(f"vis_observers_{sfx}")


def _register_vis_trees(
    sedona: SparkSession,
    trees_dir: Path,
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    geo_code: str,
    tree_area: int,
    tree_height: int,
    tree_paths: list[Path] | None = None,
) -> DataFrame:
    """Load trees and register vis_trees_<sfx> keeping canopy polygons.

    Unlike t3.read_trees_unique, geometries stay as polygons (needed for
    obstruction) with the centroid added as tree_pt (the sightline target).
    Trees without a height are dropped with a warning; the tree_area /
    tree_height thresholds then apply (sub-threshold trees are not considered
    as obstacles either).
    """
    geo_trees_gdf = load_trees_gdf(trees_dir, geo_boundary_gdf, cfg, tree_paths=tree_paths)

    n_all = len(geo_trees_gdf)
    n_bad = int(geo_trees_gdf["tree_height"].isna().sum()) if "tree_height" in geo_trees_gdf.columns else 0
    if n_bad:
        logger.warning(
            f"Visibility: skipping {n_bad}/{n_all} trees with missing height in {geo_code} — "
            f"complete tree height data is expected"
        )

    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
    if "geom" in geo_trees_sdf.columns:
        geo_trees_sdf = geo_trees_sdf.withColumnRenamed("geom", "geometry")
    geo_trees_sdf = (
        geo_trees_sdf
        .where(f"tree_height IS NOT NULL AND tree_area > {tree_area} AND tree_height > {tree_height} AND geometry IS NOT NULL")
        .selectExpr("geometry", "ST_Centroid(geometry) AS tree_pt", "CAST(tree_height AS DOUBLE) AS tree_height")
        .withColumn("tree_id", monotonically_increasing_id())
    )
    geo_trees_sdf.createOrReplaceTempView(f"vis_trees_{view_suffix(geo_code)}")
    return geo_trees_sdf


def _blocked_flags_sql(sfx: str) -> str:
    """SQL computing, per building–tree pair, one blocked flag per of the 9 sightlines.

    A single 2D sightline per pair is intersected with every obstacle; since the
    sightline is straight, the fraction of any intersection point along it is
    dist(obs_pt, .) / sight_len, and the sightline z is linear in that fraction,
    so the minimum z over a crossing is attained at f_lo or f_hi. An obstacle
    blocks the (z_obs, z_tgt) sightline iff its height reaches that minimum.
    """
    obs_levels = {k: v.format(h="bh") for k, v in _LEVELS.items()}
    tgt_levels = {k: v.format(h="th") for k, v in _LEVELS.items()}
    flags = ",\n            ".join(
        f"MAX(CASE WHEN obs_h >= LEAST({zo} + f_lo * ({zt} - ({zo})), {zo} + f_hi * ({zt} - ({zo}))) "
        f"THEN 1 ELSE 0 END) AS blk_{ko}_{kt}"
        for ko, zo in obs_levels.items()
        for kt, zt in tgt_levels.items()
    )
    return f"""
        SELECT building_id, tree_id,
            {flags}
        FROM (
            SELECT building_id, tree_id, bh, th, obs_h,
                   GREATEST(ST_Distance(obs_pt, inter) / sight_len, {_ENDPOINT_EPSILON} / sight_len) AS f_lo,
                   LEAST(1.0D - ST_Distance(tgt_pt, inter) / sight_len, 1.0D - {_ENDPOINT_EPSILON} / sight_len) AS f_hi
            FROM (
                SELECT s.building_id, s.tree_id, s.bh, s.th, s.obs_pt, s.tgt_pt, s.sight_len,
                       o.obs_h, ST_Intersection(s.sightline, o.geometry) AS inter
                FROM vis_pairs_{sfx} s
                JOIN vis_obstacles_{sfx} o ON ST_Intersects(s.sightline, o.geometry)
                WHERE s.sight_len > 0
                  AND o.obs_id <> CONCAT('b_', CAST(s.building_id AS STRING))
                  AND o.obs_id <> CONCAT('t_', CAST(s.tree_id AS STRING))
            )
        )
        WHERE f_lo <= f_hi
        GROUP BY building_id, tree_id
    """


def compute_visibility(sedona: SparkSession, geo_code: str, buffer: int, observer_mode: str) -> pd.DataFrame:
    """Run the pair/obstruction/aggregation queries and return per-building counts."""
    sfx = view_suffix(geo_code)

    obs_expr = (
        "ST_Centroid(b.geometry)"
        if observer_mode == "centroid"
        else "ST_ClosestPoint(ST_Boundary(b.geometry), t.tree_pt)"
    )
    sedona.sql(
        f"""
        SELECT building_id, tree_id, bh, th, obs_pt, tgt_pt,
               ST_MakeLine(obs_pt, tgt_pt) AS sightline,
               ST_Distance(obs_pt, tgt_pt) AS sight_len
        FROM (
            SELECT b.building_id, b.building_height AS bh, t.tree_id, t.tree_height AS th,
                   {obs_expr} AS obs_pt, t.tree_pt AS tgt_pt
            FROM vis_observers_{sfx} b
            JOIN vis_trees_{sfx} t ON ST_DWithin(b.geometry, t.tree_pt, {buffer})
        )
        """
    ).createOrReplaceTempView(f"vis_pairs_{sfx}")

    sedona.sql(
        f"""
        SELECT CONCAT('b_', CAST(building_id AS STRING)) AS obs_id, geometry, building_height AS obs_h
        FROM vis_buildings_{sfx}
        UNION ALL
        SELECT CONCAT('t_', CAST(tree_id AS STRING)) AS obs_id, geometry, tree_height AS obs_h
        FROM vis_trees_{sfx}
        """
    ).createOrReplaceTempView(f"vis_obstacles_{sfx}")

    sedona.sql(_blocked_flags_sql(sfx)).createOrReplaceTempView(f"vis_blocked_{sfx}")

    # A tree is visible from a building level iff at least one of its 3 target
    # levels has a clear sightline. Pairs with no intersecting obstacle have no
    # row in vis_blocked (inner join) -> COALESCE to unblocked.
    vis_level = {
        ko: " AND ".join(f"COALESCE(k.blk_{ko}_{kt}, 0) = 1" for kt in _LEVELS)
        for ko in _LEVELS
    }
    visible_df = sedona.sql(
        f"""
        SELECT o.building_id,
               COALESCE(SUM(CASE WHEN v.vis_bottom THEN 1 ELSE 0 END), 0) AS visible_trees_bottom,
               COALESCE(SUM(CASE WHEN v.vis_middle THEN 1 ELSE 0 END), 0) AS visible_trees_middle,
               COALESCE(SUM(CASE WHEN v.vis_top THEN 1 ELSE 0 END), 0) AS visible_trees_top,
               COALESCE(SUM(CASE WHEN v.vis_bottom OR v.vis_middle OR v.vis_top THEN 1 ELSE 0 END), 0) AS visible_trees
        FROM vis_observers_{sfx} o
        LEFT JOIN (
            SELECT p.building_id, p.tree_id,
                   NOT ({vis_level["bottom"]}) AS vis_bottom,
                   NOT ({vis_level["middle"]}) AS vis_middle,
                   NOT ({vis_level["top"]}) AS vis_top
            FROM vis_pairs_{sfx} p
            LEFT JOIN vis_blocked_{sfx} k
              ON p.building_id = k.building_id AND p.tree_id = k.tree_id
        ) v ON o.building_id = v.building_id
        GROUP BY o.building_id
        """
    ).toPandas()

    sedona.catalog.dropTempView(f"vis_blocked_{sfx}")
    return visible_df


def _attach_sub_geo_level(
    sedona, visible_df: pd.DataFrame, geo_level: str, geo_code: str, sub_geo_level: str
) -> pd.DataFrame:
    """Join the sub_geo_level code of each observer building from the overlay lookup."""
    building_level_df = sedona.sql(
        f"""
        SELECT building_id, {sub_geo_level}
        FROM boundaries_buildings_overlay
        WHERE {geo_level} = '{geo_code}'
        """
    ).toPandas()
    return visible_df.merge(building_level_df, on="building_id", how="left")


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
    observer_mode: str = "facade",
    overwrite: bool = True,
    # tile-mode optional args
    overlapping_tiles_lst: list | None = None,
) -> pd.DataFrame | None:
    """Compute tree visibility from buildings for one geo_code.

    Writes `Visibility_<geo_code>_<buffer>m.csv` to output_dir with columns
    building_id, visible_trees_bottom, visible_trees_middle, visible_trees_top,
    visible_trees, <sub_geo_level>. Returns the DataFrame, the cached CSV when
    it exists and overwrite is False, or None on error.
    """
    start_time = time.time()
    logger.info(f"Visibility: processing {geo_code} with buffer {buffer}m ({observer_mode} observer)")

    out_path = output_dir / f"Visibility_{geo_code}_{buffer}m.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    if "building_height" not in sedona.table("buildings").columns:
        logger.error(
            "Visibility requires building heights: set columns.building_height_col in the config, "
            "then delete <output.base_dir>/database/buildings.parquet so the cache is rebuilt"
        )
        return None

    try:
        geo_boundary_gdf = gpd.GeoDataFrame(
            get_geometries(sedona, geo_level, geo_code, dissolve=True).toPandas(),
            geometry="geometry", crs=cfg.crs,
        )
        _register_vis_buildings(sedona, geo_code, buffer)

        trees_dir = Path(cfg.data.trees_dir)
        tree_paths = None
        if cfg.tile_system.enabled and overlapping_tiles_lst is not None:
            # Tile-mode: pre-filter tile files by name substrings
            tree_paths = [p for p in trees_dir.glob("*.gpkg") if any(t in p.name for t in overlapping_tiles_lst)]

        _register_vis_trees(
            sedona, trees_dir, geo_boundary_gdf, cfg, geo_code, tree_area, tree_height, tree_paths=tree_paths
        )

        visible_df = compute_visibility(sedona, geo_code, buffer, observer_mode)
        visible_df = _attach_sub_geo_level(sedona, visible_df, geo_level, geo_code, sub_geo_level)
        visible_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"Visibility: {geo_code} — {len(visible_df)} records in {end_time - start_time:.2f}s")
        return visible_df

    except Exception:
        logger.exception(f"Visibility: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
        sedona.catalog.dropTempView(f"vis_blocked_{view_suffix(geo_code)}")
