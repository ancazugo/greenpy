
from utils.paths import T30_dir
from utils.constants import PROJECT_CRS
from utils.data_processing import generate_tile_paths, get_geometries

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

# TODO: Do the zonal statistics with sedona
def binarise_tiles(vom_paths_lst: list, low_threshold: float, high_threshold: float) -> xr.DataArray:
    """
    Binarise the canopy height model (CHM) tiles based on given thresholds.

    This function reads a list of CHM files, merges them into a single xarray DataArray,
    and then binarises the merged array based on the provided low and high thresholds.
    The resulting binary array will have values of 1 where the CHM values are within the
    specified range, and 0 otherwise.

    Args:
        selected_chm_path_lst (list): List of file paths to the CHM tiles to be processed.
        low_threshold (float): The lower threshold for binarisation.
        high_threshold (float): The upper threshold for binarisation.

    Returns:
        xr.DataArray: A binary xarray DataArray where values are 1 if within the threshold range, and 0 otherwise.
    """

    logging.info(f"Binarising {len(vom_paths_lst)} VOM tiles")

    chm_xr_lst = []
    for file in vom_paths_lst:
        try:
            temp_rast = rxr.open_rasterio(file)
            temp_rast.values
            chm_xr_lst.append(temp_rast)
            
        except Exception as e:
            logging.error(f"Error reading file: {file} - {e}")
    merged_chm_xr = merge_arrays(chm_xr_lst)

    binary_merged_chm_xr = (merged_chm_xr >= low_threshold) & (merged_chm_xr <= high_threshold)
    binary_merged_chm_xr = binary_merged_chm_xr.astype(int).fillna(0)

    return binary_merged_chm_xr

def get_canopy_cover(subgeo_filt_gdf: gpd.GeoDataFrame, binary_merged_chm_xr: xr.DataArray) -> pd.DataFrame:
    """
    Calculate the canopy cover percentage for each geometry in the given GeoDataFrame.

    Args:
        subgeo_filt_gdf (gpd.GeoDataFrame): A GeoDataFrame containing the geometries for which the canopy cover is to be calculated.
        binary_merged_chm_xr (xr.DataArray): A DataArray containing binary canopy height model data.

    Returns:
        pd.DataFrame: A DataFrame containing the original geometries and their corresponding canopy cover percentages.
    """

    logging.debug("Calculating canopy cover")

    zs_categorical = zonal_stats(subgeo_filt_gdf, binary_merged_chm_xr[0].values, 
                                affine=binary_merged_chm_xr.rio.transform(), categorical=True)

    # subgeo_filt_gdf['canopy_cover'] = [round(100 * z.get(1, 0) / (z.get(0, 0) + z.get(1, 0)), 3) for z in zs_categorical]
    subgeo_filt_gdf['canopy_cover'] = [round(100 * z.get(1, 0) / sum(z.values()), 3) if z else np.nan for z in zs_categorical]
    subgeo_filt_gdf['total_pixels'] = [z.get(0, 0) + z.get(1, 0) for z in zs_categorical]

    geo_canopy_cover_df = subgeo_filt_gdf.copy()
    
    return geo_canopy_cover_df

def process_geo_code(sedona: SparkSession, geo_level: str, geo_code: str, output_areas_os_tile_overlay_df: pd.DataFrame, 
                     vom_raster_paths_df: pd.DataFrame, tree_vector_paths_df: pd.DataFrame,
                     low_threshold: int=3, high_threshold: int=60, overwrite: bool=True) -> pd.DataFrame:
    """
    Processes a given geo_code for T30.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo level.
        geo_code (str): The geo code.
        output_areas_os_tile_overlay_df (pd.DataFrame): The output areas OS tile overlay dataframe. 
        vom_raster_paths_df (pd.DataFrame): The VOM raster paths dataframe.
        tree_vector_paths_df (pd.DataFrame): The tree vector paths dataframe.
        low_threshold (int): The low threshold for binarisation.
        high_threshold (int): The high threshold for binarisation.
        overwrite (bool): Whether to overwrite the existing file.

    Returns:    
        pd.DataFrame: The canopy cover dataframe.
    """
    
    start_time = time.time()
    logging.info(f"Processing data for {geo_code}")

    geo_canopy_cover_path = T30_dir / f"T30_{geo_code}.csv"

    if not geo_canopy_cover_path.exists() or overwrite:
        try:
            # subgeo_filt_gdf = output_areas_boundaries_gdf.copy()[output_areas_boundaries_gdf[geo_level].isin([geo_code])].reset_index(drop=True)
            geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, False)
            geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
            geo_tiles_df = generate_tile_paths(geo_level, geo_code, output_areas_os_tile_overlay_df, vom_raster_paths_df, tree_vector_paths_df)
            vom_paths_lst = geo_tiles_df.groupby('TILE_NAME').first().reset_index()['path_vom'].tolist()
            binary_merged_chm_xr = binarise_tiles(vom_paths_lst, low_threshold, high_threshold)
            geo_canopy_cover_df = get_canopy_cover(geo_boundary_gdf, binary_merged_chm_xr)
            geo_canopy_cover_df = geo_canopy_cover_df[['OA21CD', 'LSOA21CD', 'MSOA21CD', 'LAD22CD', 'RGN22CD', 'canopy_cover', 'total_pixels']]

            geo_canopy_cover_df.to_csv(geo_canopy_cover_path, index=False)

            end_time = time.time()
            logging.info(f"Processing for {geo_code} with {len(geo_canopy_cover_df)} records took {end_time - start_time:.2f} seconds")

            return geo_canopy_cover_df

        except Exception as e:
            logging.error(f"Error processing {geo_code}: {e}")
            