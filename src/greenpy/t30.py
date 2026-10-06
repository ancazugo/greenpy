from pathlib import Path

import time
import numpy as np
import pandas as pd
import shapely
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterstats import zonal_stats
from loguru import logger
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import (
    get_sub_geo_boundaries,
    find_overlapping_rasters,
    load_trees_gdf,
    drop_geo_views,
)


def binarise_tiles(
    chm_paths_lst: list, low_threshold: float, high_threshold: float,
    target_crs: str | None = None, overlap: str = "latest",
) -> xr.DataArray:
    """Merge CHM raster tiles and binarise to a canopy (1) / no-canopy (0) mask.

    Tiles are opened with their nodata mask applied, so nodata pixels stay NaN
    in the result and are excluded from canopy-cover statistics rather than
    being counted as no-canopy. Where tiles overlap, overlap="latest" keeps
    the last path in sorted order and "max" the per-pixel maximum height
    (data.chm_overlap).
    """
    if overlap not in ("latest", "max"):
        raise ValueError(f"overlap must be 'latest' or 'max', got {overlap!r}")
    chm_paths_lst = sorted(str(p) for p in chm_paths_lst)
    logger.info(f"Binarising {len(chm_paths_lst)} CHM tiles")

    chm_xr_lst = []
    for file in chm_paths_lst:
        try:
            temp_rast = rxr.open_rasterio(file, masked=True)
            temp_rast.values
            chm_xr_lst.append(temp_rast)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    if not chm_xr_lst:
        raise FileNotFoundError("No readable CHM tiles overlap the boundary")
    merged_chm_xr = merge_arrays(chm_xr_lst, method="last" if overlap == "latest" else "max")
    if target_crs is not None and merged_chm_xr.rio.crs is not None:
        from pyproj import CRS as ProjCRS
        if merged_chm_xr.rio.crs != ProjCRS.from_user_input(target_crs):
            merged_chm_xr = merged_chm_xr.rio.reproject(target_crs)
    binary = ((merged_chm_xr >= low_threshold) & (merged_chm_xr <= high_threshold)).astype(float)
    return binary.where(merged_chm_xr.notnull())


def get_canopy_cover_raster(subgeo_gdf: gpd.GeoDataFrame, binary_chm_xr: xr.DataArray) -> gpd.GeoDataFrame:
    """Per-unit canopy cover (%) via zonal statistics on a canopy-fraction raster.

    Pixels hold canopy fraction in [0, 1]: 1/0 for a binarised CHM, or the
    fraction of canopy sub-pixels for a coarsened GEE download. Cover is the
    mean fraction over valid pixels (pixel centres inside the unit). Nodata
    (NaN) pixels are excluded from both numerator and denominator;
    total_pixels is the number of valid pixels per unit.
    """
    logger.debug("Calculating canopy cover from raster")

    zs = zonal_stats(
        subgeo_gdf,
        binary_chm_xr[0].values,
        affine=binary_chm_xr.rio.transform(),
        stats=["sum", "count"],
        nodata=np.nan,
    )
    canopy = [z["sum"] or 0 for z in zs]
    valid = [z["count"] or 0 for z in zs]
    subgeo_gdf = subgeo_gdf.copy()
    subgeo_gdf["canopy_cover"] = [
        round(100 * c / t, 3) if t else np.nan for c, t in zip(canopy, valid)
    ]
    subgeo_gdf["total_pixels"] = valid
    return subgeo_gdf


def get_canopy_cover_vector(subgeo_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Vector-based canopy cover: canopy area inside each census unit / unit area.

    Polygon crowns are clipped to each unit and unioned per unit, so a crown
    straddling two units contributes only its own part to each, and
    overlapping crowns are not double-counted. Point trees (no crown geometry)
    fall back to their stored `tree_area`, credited to the unit containing the
    point — their overlaps cannot be resolved. `total_pixels` holds the unit
    area in m² (1 m² ≈ one pixel) so the Merge step can area-weight its
    aggregation, consistent with the raster path.
    """
    logger.debug("Calculating canopy cover from tree vectors")

    subgeo_gdf = subgeo_gdf.copy()
    unit_areas = subgeo_gdf.geometry.area
    units = subgeo_gdf[["geometry"]].reset_index(names="_idx")
    trees_gdf = trees_gdf[trees_gdf.geometry.notna() & ~trees_gdf.geometry.is_empty]

    is_poly = trees_gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    canopy_area = pd.Series(0.0, index=subgeo_gdf.index)

    crowns = trees_gdf.loc[is_poly, ["geometry"]]
    if not crowns.empty:
        pieces = gpd.overlay(crowns, units, how="intersection", keep_geom_type=True)
        if not pieces.empty:
            per_unit = pieces.groupby("_idx").geometry.agg(lambda g: shapely.union_all(g.values).area)
            canopy_area = canopy_area.add(per_unit, fill_value=0)

    points = trees_gdf.loc[~is_poly]
    if not points.empty:
        if "tree_area" not in points.columns:
            raise ValueError(
                "Non-polygon trees need a tree_area column (columns.tree_area_col) to compute canopy cover"
            )
        hits = gpd.sjoin(points[["tree_area", "geometry"]], units, predicate="within")
        canopy_area = canopy_area.add(hits.groupby("_idx")["tree_area"].sum(), fill_value=0)

    subgeo_gdf["canopy_cover"] = (canopy_area.reindex(subgeo_gdf.index).fillna(0) / unit_areas * 100).round(3)
    subgeo_gdf["total_pixels"] = unit_areas.round().astype(int)
    return subgeo_gdf


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    sub_geo_level: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    low_threshold: int = 3,
    high_threshold: int = 60,
    gee_scale: float = 1.0,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    """Compute T30 (canopy cover %) per sub_geo_level unit within one geo_code.

    Canopy source, in priority order: local CHM raster tiles
    (cfg.data.chm_tiles_dir) > the Meta/WRI CHM from AWS (cfg.data.meta_chm,
    v1 or v2) > a GEE canopy-height asset binarised server-side and
    downloaded at gee_scale metres (cfg.data.canopy_height_ee_path) > tree
    polygon areas (cfg.data.trees_dir). Writes `T30_<geo_code>.csv` with
    columns <sub_geo_level>, canopy_cover, total_pixels. Returns the
    DataFrame, the cached CSV when overwrite is False, or None on error.
    """
    start_time = time.time()
    logger.info(f"T30: processing {geo_code}")

    out_path = output_dir / f"T30_{geo_code}.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        geo_boundary_sdf = get_sub_geo_boundaries(sedona, geo_level, geo_code, sub_geo_level)
        geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry="geometry", crs=cfg.crs)

        if cfg.data.chm_tiles_dir:
            chm_dir = Path(cfg.data.chm_tiles_dir)
            chm_paths = find_overlapping_rasters(geo_boundary_gdf, chm_dir, pattern=cfg.data.chm_pattern)
            binary_chm_xr = binarise_tiles(chm_paths, low_threshold, high_threshold, target_crs=cfg.crs, overlap=cfg.data.chm_overlap)
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.meta_chm:
            from .optional.canopy_meta import meta_binary_canopy, meta_cache_path
            binary_chm_xr = meta_binary_canopy(
                geo_boundary_gdf, cfg, low_threshold, high_threshold,
                cache_path=meta_cache_path(cfg, geo_code, low_threshold, high_threshold), overwrite=overwrite,
            )
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.canopy_height_ee_path:
            from .optional.canopy_gee import download_binary_canopy, gee_cache_path  # lazy: keeps ee/xee optional
            cache = gee_cache_path(cfg, geo_code, low_threshold, high_threshold, gee_scale)
            binary_chm_xr = download_binary_canopy(
                geo_boundary_gdf, cfg, low_threshold, high_threshold,
                scale=gee_scale, cache_path=cache, overwrite=overwrite,
            )
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.trees_dir:
            trees_gdf = load_trees_gdf(Path(cfg.data.trees_dir), geo_boundary_gdf, cfg)
            geo_canopy_cover_df = get_canopy_cover_vector(geo_boundary_gdf, trees_gdf)

        else:
            raise ValueError(
                "No canopy source configured — set one of chm_tiles_dir, meta_chm, "
                "canopy_height_ee_path, or trees_dir to compute T30"
            )

        geo_canopy_cover_df = geo_canopy_cover_df[[sub_geo_level, "canopy_cover", "total_pixels"]]
        geo_canopy_cover_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"T30: {geo_code} — {len(geo_canopy_cover_df)} records in {end_time - start_time:.2f}s")
        return geo_canopy_cover_df

    except Exception:
        logger.exception(f"T30: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
