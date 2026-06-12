from pathlib import Path

import time
import numpy as np
import pandas as pd
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
    find_overlapping_files,
    rename_tree_columns,
    drop_geo_views,
)


def binarise_tiles(chm_paths_lst: list, low_threshold: float, high_threshold: float, target_crs: str | None = None) -> xr.DataArray:
    """Merge CHM raster tiles and binarise to a canopy (1) / no-canopy (0) mask.

    Tiles are opened with their nodata mask applied, so nodata pixels stay NaN
    in the result and are excluded from canopy-cover statistics rather than
    being counted as no-canopy.
    """
    logger.info(f"Binarising {len(chm_paths_lst)} CHM tiles")

    chm_xr_lst = []
    for file in chm_paths_lst:
        try:
            temp_rast = rxr.open_rasterio(file, masked=True)
            temp_rast.values
            chm_xr_lst.append(temp_rast)
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    merged_chm_xr = merge_arrays(chm_xr_lst)
    if target_crs is not None and merged_chm_xr.rio.crs is not None:
        from pyproj import CRS as ProjCRS
        if merged_chm_xr.rio.crs != ProjCRS.from_user_input(target_crs):
            merged_chm_xr = merged_chm_xr.rio.reproject(target_crs)
    binary = ((merged_chm_xr >= low_threshold) & (merged_chm_xr <= high_threshold)).astype(float)
    return binary.where(merged_chm_xr.notnull())


def get_canopy_cover_raster(subgeo_gdf: gpd.GeoDataFrame, binary_chm_xr: xr.DataArray) -> gpd.GeoDataFrame:
    """Per-unit canopy cover (%) via zonal statistics on a binary CHM raster.

    Nodata (NaN) pixels are excluded from both numerator and denominator;
    total_pixels is the number of valid pixels per unit.
    """
    logger.debug("Calculating canopy cover from raster")

    zs = zonal_stats(
        subgeo_gdf,
        binary_chm_xr[0].values,
        affine=binary_chm_xr.rio.transform(),
        categorical=True,
        nodata=np.nan,
    )
    # category keys may be ints or floats depending on the array dtype
    canopy = [sum(v for k, v in z.items() if k == 1) if z else 0 for z in zs]
    valid = [sum(z.values()) if z else 0 for z in zs]
    subgeo_gdf = subgeo_gdf.copy()
    subgeo_gdf["canopy_cover"] = [
        round(100 * c / t, 3) if t else np.nan for c, t in zip(canopy, valid)
    ]
    subgeo_gdf["total_pixels"] = valid
    return subgeo_gdf


def get_canopy_cover_vector(subgeo_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Vector-based canopy cover: sum of tree canopy area / census unit area.
    Uses `tree_area` column when trees are points with a stored area; otherwise
    computes from polygon geometry. `total_pixels` holds the unit area in m²
    (1 m² ≈ one pixel) so the Merge step can area-weight its aggregation,
    consistent with the raster path.
    """
    logger.debug("Calculating canopy cover from tree vectors")

    subgeo_gdf = subgeo_gdf.copy()
    unit_areas = subgeo_gdf.geometry.area
    tree_clip = gpd.clip(trees_gdf, subgeo_gdf)

    if tree_clip.empty:
        subgeo_gdf["canopy_cover"] = 0.0
        subgeo_gdf["total_pixels"] = unit_areas.round().astype(int)
        return subgeo_gdf

    tree_clip = tree_clip.copy()
    if "tree_area" in tree_clip.columns:
        tree_clip["clipped_area"] = tree_clip["tree_area"]
    else:
        tree_clip["clipped_area"] = tree_clip.geometry.area
    tree_area_by_unit = tree_clip.sjoin(subgeo_gdf[["geometry"]].reset_index().rename(columns={"index": "_idx"}))
    total_tree_area = tree_area_by_unit.groupby("_idx")["clipped_area"].sum()

    subgeo_gdf["canopy_cover"] = (total_tree_area.reindex(subgeo_gdf.index).fillna(0) / unit_areas * 100).round(3)
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
    (cfg.data.chm_tiles_dir) > a GEE canopy-height asset binarised server-side
    and downloaded at gee_scale metres (cfg.data.canopy_height_ee_path) > tree
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
            chm_paths = find_overlapping_rasters(geo_boundary_gdf, chm_dir, pattern="*.tif")
            binary_chm_xr = binarise_tiles(chm_paths, low_threshold, high_threshold, target_crs=cfg.crs)
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.canopy_height_ee_path:
            from .optional.canopy_gee import download_binary_canopy  # lazy: keeps ee/xee optional
            cache = Path(cfg.output.base_dir) / "database" / "gee_canopy" / f"{geo_code}.tif"
            binary_chm_xr = download_binary_canopy(
                geo_boundary_gdf, cfg, low_threshold, high_threshold,
                scale=gee_scale, cache_path=cache, overwrite=overwrite,
            )
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.trees_dir:
            trees_dir = Path(cfg.data.trees_dir)
            if trees_dir.is_file():
                suffix = trees_dir.suffix.lower()
                trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
                trees_gdf = trees_gdf.to_crs(cfg.crs) if trees_gdf.crs is not None else trees_gdf.set_crs(cfg.crs)
            else:
                tree_paths = find_overlapping_files(geo_boundary_gdf, trees_dir)
                parts = [gpd.read_file(p) for p in tree_paths]
                parts = [g.to_crs(cfg.crs) if g.crs is not None else g.set_crs(cfg.crs) for g in parts]
                trees_gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=cfg.crs)
            trees_gdf = rename_tree_columns(trees_gdf, cfg)
            geo_canopy_cover_df = get_canopy_cover_vector(geo_boundary_gdf, trees_gdf)

        else:
            raise ValueError(
                "No canopy source configured — set one of chm_tiles_dir, "
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
