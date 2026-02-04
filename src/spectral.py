
from utils.constants import GEE_PROJECT_NAME 
from utils.paths import Spectral_dir, output_areas_boundaries_ee_path

import time
import logging
import ee
import eemont
import pandas as pd

def setup_gee() -> None:
    """
    Sets up the GEE project.
    """

    logging.debug(f"Initializing GEE for project: {GEE_PROJECT_NAME}")

    ee.Authenticate()
    ee.Initialize(project=GEE_PROJECT_NAME, opt_url='https://earthengine-highvolume.googleapis.com')

def get_imagery(geo_level_filt_fc: ee.featurecollection.FeatureCollection, imagery_ee_path: str,
                start_date: str, end_date: str, cloud_coverage: float, spectral_indexes: list[str]) -> ee.image.Image:
    """
    Gets the imagery from GEE.

    Args:
        geo_level_filt_fc (ee.featurecollection.FeatureCollection): The filtered feature collection.
        imagery_ee_path (str): The path to the imagery in GEE.
        start_date (str): The start date.
        end_date (str): The end date.
        cloud_coverage (float): The cloud coverage.
        spectral_indexes (list[str]): The spectral indexes.

    Returns:
        ee.image.Image: The imagery.
    """

    logging.debug("Querying GEE for imagery")

    geo_level_union_geometry = geo_level_filt_fc.union().geometry()

    imagery_ic = (ee.ImageCollection(imagery_ee_path) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_coverage)) \
        .filterBounds(geo_level_union_geometry) \
        .spectralIndices(spectral_indexes).max() \
        .select(spectral_indexes))
    
    return imagery_ic

def calculate_median_index(imagery_ic: ee.image.Image, geometries: ee.featurecollection.FeatureCollection, 
                           scale: float=10.0, tile_scale: int=4) -> ee.featurecollection.FeatureCollection:
    """
    Calculate the median index of an image collection over specified geometries.

    Args:
        image_collection (ee.image.Image): The image collection to process.
        geometries (ee.featurecollection.FeatureCollection): The geometries over which to calculate the median index.
        scale (float, optional): The scale in meters at which to perform the reduction. Default is 10.0.
        tile_scale (int, optional): A scaling factor used to reduce aggregation tile size; may help avoid memory errors. Default is 4.

    Returns:
        ee.featurecollection.FeatureCollection: A FeatureCollection containing the median index values for each geometry.
    """

    logging.debug("Calculating median index for geometries")

    median_dict = imagery_ic.reduceRegions(
        collection=geometries,
        reducer=ee.Reducer.median(), scale=scale, tileScale=tile_scale
    )
    return median_dict

def process_geo_code(geo_code: str, geo_level: str, sub_geo_level: str, imagery_ee_path: str, 
                     start_date: str, end_date: str, cloud_coverage: float, 
                     spectral_indexes: list[str], overwrite: bool=True) -> pd.DataFrame:
    """
    Processes a given geo_code for spectral indices.

    Args:
        geo_code (str): The geo code.
        geo_level (str): The geo level.
        sub_geo_level (str): The sub geo level.
        imagery_ee_path (str): The path to the imagery in GEE.
        start_date (str): The start date.
        end_date (str): The end date.
        cloud_coverage (float): The cloud coverage.
        spectral_indexes (list[str]): The spectral indexes.
        overwrite (bool): Whether to overwrite the existing file.
        
    Returns:
        pd.DataFrame: The spectral indices dataframe.
    """
    
    start_time = time.time()
    logging.info(f"Processing data for {geo_code}")

    geo_spectral_index_path = Spectral_dir / f"Spectral_{geo_code}.csv"

    if not geo_spectral_index_path.exists() or overwrite:
        try:
            output_areas_boundaries_fc = ee.FeatureCollection(output_areas_boundaries_ee_path)
            sub_geo_level_filt_fc = output_areas_boundaries_fc.filter(ee.Filter.eq(geo_level, geo_code))
            sub_geo_code_values = sub_geo_level_filt_fc.aggregate_array(sub_geo_level).distinct()
            imagery_ic = get_imagery(sub_geo_level_filt_fc, imagery_ee_path, start_date, end_date, cloud_coverage, spectral_indexes)

            def dissolve_by_code(sub_geo_code):

                geo_code = ee.String(sub_geo_code)
                temp_fc = output_areas_boundaries_fc \
                    .filter(ee.Filter.eq(sub_geo_level, geo_code))
                union_geom = temp_fc.union().geometry()

                return ee.Feature(union_geom).set(sub_geo_level, geo_code)
            
            sub_geo_boundaries_union_fc = ee.FeatureCollection(sub_geo_code_values.map(dissolve_by_code))

            # spectral_results_lst = calculate_zonal_statistics(imagery_ic, sub_geo_boundaries_union_fc)    
            spectral_results_lst = calculate_median_index(imagery_ic, sub_geo_boundaries_union_fc).getInfo()['features']
            properties_lst = [feature['properties'] for feature in spectral_results_lst]
            # import geopandas as gpd
            # from shapely.geometry import shape
            # geometry_lst = [shape(feature['geometry']) for feature in spectral_results_lst]
            # spectral_index_gdf = gpd.GeoDataFrame(properties_lst, geometry=geometry_lst, crs='EPSG:4326').to_crs(PROJECT_CRS)
            geo_spectral_index_df = pd.DataFrame(properties_lst)

            geo_spectral_index_df.to_csv(geo_spectral_index_path, index=False)

            end_time = time.time()
            logging.info(f"Processing for {geo_code} with {len(geo_spectral_index_df)} records took {end_time - start_time:.2f} seconds")
            
            return geo_spectral_index_df
        
        except Exception as e:
            logging.error(f"Error processing {geo_code}: {e}")