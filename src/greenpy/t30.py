from pathlib import Path

import time
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterstats import zonal_stats
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import get_geometries, find_overlapping_rasters


def binarise_tiles(chm_paths_lst: list, low_threshold: float, high_threshold: float) -> xr.DataArray:
    """Merge CHM raster tiles and binarise to a canopy/no-canopy mask."""
    logging.info(f"Binarising {len(chm_paths_lst)} CHM tiles")

    chm_xr_lst = []
    for file in chm_paths_lst:
        try:
            temp_rast = rxr.open_rasterio(file)
            temp_rast.values
            chm_xr_lst.append(temp_rast)
        except Exception as e:
            logging.error(f"Error reading {file}: {e}")

    merged_chm_xr = merge_arrays(chm_xr_lst)
    binary = (merged_chm_xr >= low_threshold) & (merged_chm_xr <= high_threshold)
    return binary.astype(int).fillna(0)


def get_canopy_cover_raster(subgeo_gdf: gpd.GeoDataFrame, binary_chm_xr: xr.DataArray) -> gpd.GeoDataFrame:
    """Zonal statistics-based canopy cover from a binary CHM raster."""
    logging.debug("Calculating canopy cover from raster")

    zs = zonal_stats(
        subgeo_gdf,
        binary_chm_xr[0].values,
        affine=binary_chm_xr.rio.transform(),
        categorical=True,
    )
    subgeo_gdf = subgeo_gdf.copy()
    subgeo_gdf["canopy_cover"] = [
        round(100 * z.get(1, 0) / sum(z.values()), 3) if z else np.nan for z in zs
    ]
    subgeo_gdf["total_pixels"] = [z.get(0, 0) + z.get(1, 0) for z in zs]
    return subgeo_gdf


def get_canopy_cover_vector(subgeo_gdf: gpd.GeoDataFrame, trees_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Vector-based canopy cover: sum of tree canopy area / census unit area.
    Uses `tree_area` column when trees are points with a stored area; otherwise
    computes from polygon geometry.
    """
    logging.debug("Calculating canopy cover from tree vectors")

    subgeo_gdf = subgeo_gdf.copy()
    tree_clip = gpd.clip(trees_gdf, subgeo_gdf)

    if tree_clip.empty:
        subgeo_gdf["canopy_cover"] = 0.0
        subgeo_gdf["total_pixels"] = 0
        return subgeo_gdf

    tree_clip = tree_clip.copy()
    if "tree_area" in tree_clip.columns:
        tree_clip["clipped_area"] = tree_clip["tree_area"]
    else:
        tree_clip["clipped_area"] = tree_clip.geometry.area
    tree_area_by_unit = tree_clip.sjoin(subgeo_gdf[["geometry"]].reset_index().rename(columns={"index": "_idx"}))
    total_tree_area = tree_area_by_unit.groupby("_idx")["clipped_area"].sum()

    unit_areas = subgeo_gdf.geometry.area
    subgeo_gdf["canopy_cover"] = (total_tree_area.reindex(subgeo_gdf.index).fillna(0) / unit_areas * 100).round(3)
    subgeo_gdf["total_pixels"] = 0
    return subgeo_gdf


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    low_threshold: int = 3,
    high_threshold: int = 60,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    start_time = time.time()
    logging.info(f"T30: processing {geo_code}")

    out_path = output_dir / f"T30_{geo_code}.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, dissolve=False)
        geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry="geometry", crs=cfg.crs)

        geo_levels = cfg.columns.geo_levels
        output_cols = geo_levels + ["canopy_cover", "total_pixels"]

        if cfg.data.chm_tiles_dir:
            chm_dir = Path(cfg.data.chm_tiles_dir)
            chm_paths = find_overlapping_rasters(geo_boundary_gdf, chm_dir, pattern="*.tif")
            binary_chm_xr = binarise_tiles(chm_paths, low_threshold, high_threshold)
            geo_canopy_cover_df = get_canopy_cover_raster(geo_boundary_gdf, binary_chm_xr)

        elif cfg.data.trees_dir:
            from .utils.data_processing import find_overlapping_files
            trees_dir = Path(cfg.data.trees_dir)
            col = cfg.columns
            if trees_dir.is_file():
                suffix = trees_dir.suffix.lower()
                trees_gdf = gpd.read_parquet(trees_dir) if suffix in (".parquet", ".geoparquet") else gpd.read_file(trees_dir)
                trees_gdf = trees_gdf.rename(columns={col.tree_area_col: "tree_area"})
            else:
                tree_paths = find_overlapping_files(geo_boundary_gdf, trees_dir)
                trees_gdf = gpd.GeoDataFrame(
                    pd.concat([gpd.read_file(p) for p in tree_paths], ignore_index=True)
                )
            geo_canopy_cover_df = get_canopy_cover_vector(geo_boundary_gdf, trees_gdf)

        else:
            raise ValueError("Neither chm_tiles_dir nor trees_dir is configured — cannot compute T30")

        # Keep only geo columns that exist in this geography
        present_geo_cols = [c for c in geo_levels if c in geo_canopy_cover_df.columns]
        geo_canopy_cover_df = geo_canopy_cover_df[present_geo_cols + ["canopy_cover", "total_pixels"]]
        geo_canopy_cover_df.to_csv(out_path, index=False)

        end_time = time.time()
        logging.info(f"T30: {geo_code} — {len(geo_canopy_cover_df)} records in {end_time - start_time:.2f}s")
        return geo_canopy_cover_df

    except Exception as e:
        logging.error(f"T30: error processing {geo_code}: {e}")
