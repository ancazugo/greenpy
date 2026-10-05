"""Raster engine: visible trees per building by ray casting over a building + vegetation surface.

Per geo code, the owned buildings are processed in square tiles (tile_size):
each tile gets a surface model over its buildings plus the search buffer,
observer points on their facades, the T3 candidate trees within the buffer,
and one lowest-visible-eye-height per building–tree pair from the numba
kernel. No Spark is needed.
"""

import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger
from shapely.geometry import box

from ...config.schema import GreenPyConfig
from ...terrain import dem_layers as terrain_layers
from ...utils.data_processing import load_trees_gdf
from .dsm import build_dsm
from .inputs import VisibilityInputs
from .los import pair_min_eye_height
from .observers import facade_points, floor_eyes
from .pairs import building_counts, eligible_pairs
from .params import VisibilityParams
from .targets import crown_polygons, tree_targets

# Crowns can reach this far (m) beyond their centroid: trees are searched this much beyond the buffer
CROWN_MARGIN = 15.0

OUTPUT_COLUMNS = [
    "building_id", "visible_trees", "visible_trees_ground", "candidate_trees", "n_floors",
    "building_height", "height_source",
]


def vegetation_mode(cfg: GreenPyConfig, params: VisibilityParams) -> str:
    """'chm' when a canopy height model is configured (auto) or requested, else 'crowns'."""
    if params.vegetation != "auto":
        return params.vegetation
    explicit = cfg.data.chm_tiles_dir or cfg.tree_segmentation.source
    return "chm" if explicit else "crowns"


def _chm_layers(cfg: GreenPyConfig, bounds: tuple, name: str) -> list:
    from ..chm_sources import chm_mosaic
    from ..tree_segmentation import cache_dir_for, chm_source

    area = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=cfg.crs)
    return chm_mosaic(
        area, chm_source(cfg), cache_dir_for(cfg), name,
        chm_tiles_dir=cfg.data.chm_tiles_dir, chm_pattern=cfg.data.chm_pattern, overlap=cfg.data.chm_overlap,
    )


def load_trees(cfg: GreenPyConfig, bounds: tuple, params: VisibilityParams) -> gpd.GeoDataFrame:
    """Trees whose files overlap bounds, with canonical tree_height (and tree_area); invalid rows dropped."""
    area = gpd.GeoDataFrame(geometry=[box(*bounds)], crs=cfg.crs)
    trees = load_trees_gdf(Path(cfg.data.trees_dir), area, cfg)
    if "tree_height" not in trees.columns:
        raise ValueError(f"Trees have no height column '{cfg.columns.tree_height_col}' (columns.tree_height_col)")
    trees = trees[trees.geometry.notna() & ~trees.geometry.is_empty]
    trees = trees.assign(tree_height=pd.to_numeric(trees["tree_height"], errors="coerce"))
    trees = trees[trees["tree_height"] > 0]
    return trees.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].reset_index(drop=True)


def target_mask(trees: gpd.GeoDataFrame, params: VisibilityParams) -> np.ndarray:
    """Trees that count (as T3): area and height strictly above the thresholds."""
    ok = trees["tree_height"].to_numpy(dtype=float) > params.tree_height
    if "tree_area" in trees.columns:
        ok &= pd.to_numeric(trees["tree_area"], errors="coerce").to_numpy(dtype=float) > params.tree_area
    return ok


def visibility_for_buildings(
    idx: np.ndarray, inputs: VisibilityInputs, trees: gpd.GeoDataFrame, params: VisibilityParams,
    cfg: GreenPyConfig, chm_layers: list | None, n_floors: np.ndarray, z_top: np.ndarray,
    dem_layers: list | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Counts for buildings idx (one tile); also returns (pair building position, z_req) for inspection."""
    geoms = inputs.geoms[idx]
    minx, miny, maxx, maxy = shapely.total_bounds(geoms)
    pad = params.buffer + CROWN_MARGIN + 2 * params.resolution
    bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)

    near = inputs.tree.query(box(*bounds))
    blockers = shapely.STRtree(inputs.geoms[near])

    tb = trees.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]]
    targets_gdf = tb[target_mask(tb, params)]
    targets = tree_targets(targets_gdf, params.crown_points, params.crown_point_height)
    pair_b, pair_t = eligible_pairs(geoms, targets.ref_x, targets.ref_y, params.buffer)

    # without a CHM every tree obstructs as its crown (point trees as discs of their crown area)
    dsm = build_dsm(
        bounds, params.resolution, cfg.crs, inputs.geoms[near], inputs.height[near],
        target_crowns=targets.crowns,
        veg_geoms=None if chm_layers else crown_polygons(tb),
        veg_h=None if chm_layers else tb["tree_height"].to_numpy(dtype=float),
        chm_layers=chm_layers, mask_chm_buildings=params.mask_chm_buildings, dem_layers=dem_layers,
    )

    fp = facade_points(geoms, params.facade_spacing, params.facade_offset, blockers=blockers)
    z_req = pair_min_eye_height(
        dsm.dsm, dsm.bldg, dsm.crown_id, dsm.x0, dsm.y0, dsm.res,
        fp.x, fp.y, fp.nx, fp.ny, fp.start, fp.count,
        targets.x, targets.y, targets.z, targets.start, targets.count,
        pair_b, pair_t, z_top, params.eye_height, params.skip, params.skip,
        go=dsm.ground_at(fp.x, fp.y), gt=dsm.ground_at(targets.x, targets.y), skip_ground=not dsm.terrain,
    )
    counts = building_counts(len(idx), pair_b, z_req, z_top, params.eye_height)
    counts.insert(0, "building_id", inputs.building_id[idx])
    counts["n_floors"] = n_floors
    counts["building_height"] = inputs.height[idx]
    counts["height_source"] = inputs.height_source[idx]
    return counts, pair_b, z_req


def process_geo_code_raster(
    geo_level: str, geo_code: str, sub_geo_level: str, cfg: GreenPyConfig, inputs: VisibilityInputs,
    params: VisibilityParams, output_dir: Path, overwrite: bool = True,
) -> pd.DataFrame | None:
    """Visible trees per building of one geo_code -> Visibility_<geo_code>_<buffer>m.csv."""
    start = time.time()
    out_path = Path(output_dir) / f"Visibility_{geo_code}_{int(params.buffer)}m.csv"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)
    try:
        idx = inputs.owned(geo_level, geo_code)
        if len(idx) == 0:
            logger.warning(f"Visibility: no buildings in {geo_code}")
            return None
        mode = vegetation_mode(cfg, params)
        n_floors, z_top = floor_eyes(inputs.height[idx], params.storey_height, params.eye_height)

        # tiles by representative point
        rep = shapely.point_on_surface(inputs.geoms[idx])
        key = np.floor(np.c_[shapely.get_x(rep), shapely.get_y(rep)] / params.tile_size).astype(np.int64)
        tiles = pd.Series(np.arange(len(idx))).groupby([key[:, 0], key[:, 1]]).indices

        minx, miny, maxx, maxy = shapely.total_bounds(inputs.geoms[idx])
        pad = params.buffer + CROWN_MARGIN + 2 * params.resolution
        all_bounds = (minx - pad, miny - pad, maxx + pad, maxy + pad)
        trees = load_trees(cfg, all_bounds, params)

        parts, n_pairs = [], 0
        for (kx, ky), pos in tiles.items():
            b = shapely.total_bounds(inputs.geoms[idx[pos]])
            tile_bounds = (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)
            chm_layers = _chm_layers(cfg, tile_bounds, f"vis_{geo_code}_{kx}_{ky}") if mode == "chm" else None
            counts, pair_b, _ = visibility_for_buildings(
                idx[pos], inputs, trees, params, cfg, chm_layers, n_floors[pos], z_top[pos],
                dem_layers=terrain_layers(cfg, tile_bounds),
            )
            parts.append(counts)
            n_pairs += len(pair_b)
        result = pd.concat(parts, ignore_index=True)

        sub = inputs.overlay[["building_id", sub_geo_level]].drop_duplicates("building_id")
        result = result.merge(sub, on="building_id", how="left")
        result.to_csv(out_path, index=False)
        logger.info(
            f"Visibility: {geo_code} — {len(result)} buildings, {n_pairs} building-tree pairs, {len(tiles)} tiles "
            f"({mode} vegetation, {'terrain ' + cfg.terrain.source if cfg.terrain.source else 'flat ground'}) "
            f"in {time.time() - start:.1f}s; mean visible {result['visible_trees'].mean():.2f} "
            f"of {result['candidate_trees'].mean():.2f} candidates"
        )
        return result
    except Exception:
        logger.exception(f"Visibility: error processing {geo_code}")
        return None
