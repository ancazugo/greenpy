import argparse
from pathlib import Path
import logging
import concurrent.futures
from tqdm import tqdm

from utils.constants import PROJECT_CRS
from utils.paths import T3_30_300_DIR, trees_unique_dir
from tables_setup import *
from utils.logging_config import setup_logger
from utils.data_processing import get_geometries, get_overlapping_grid_tiles, save_temp_file
from utils.sedona_config import get_spark
from t3 import read_vom_trees_unique

def sum_trees_area(sedona, geo_level, geo_code, sub_geo_level):

    overlapping_tiles_lst = get_overlapping_grid_tiles(output_areas_boundaries_gdf, os_tile_boundaries_gdf, geo_level, geo_code, tile_level)
    overlapping_tiles_lst = [tile_name.upper() for tile_name in overlapping_tiles_lst]
    
    geo_trees_sdf = read_vom_trees_unique(sedona, overlapping_tiles_lst, 0, 0)
    geo_trees_sdf.createOrReplaceTempView('trees')

    geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, False)
    geo_boundary_sdf.createOrReplaceTempView('geo_boundary')

    output_path = T3_30_300_DIR / "Tree_area" / f"{geo_code}_tree_area.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_df = sedona.sql(f"""
        SELECT 
            b.{sub_geo_level} as {sub_geo_level},
            SUM(t.area) as total_tree_area,
            COUNT(t.*) as tree_count
        FROM trees t
        JOIN geo_boundary b ON ST_Within(t.geometry, b.geometry)
        GROUP BY b.{sub_geo_level}
    """)

    result_df = save_temp_file(result_df, output_path)

    return result_df

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='This script executes the module to calculate the 3-30-300 metric and spectral indexes for all of England.')
    parser.add_argument('--process', type=str, required=False, choices=['Tree_area'], help='Name of the component of the module to run')
    parser.add_argument('--parallel', action='store_true', help='Run job in parallel')
    parser.add_argument('--n_workers', type=int, required=False, default=2, help='Number of workers')
    parser.add_argument('--log_level', type=str, required=False, default='INFO', help='Logging level')
    parser.add_argument('--tile_level', type=str, required=False, default='TILE_NAME_5KM_int', help='Name/Code of the desired geography')
    parser.add_argument('--geo_level', type=str, required=False, default='LAD22CD', choices=['RGN22CD', 'MSOA21CD', 'LAD22CD', 'LSOA21CD'], help='Name/Code of the desired geography')
    parser.add_argument('--sub_geo_level', type=str, required=False, default='OA21CD', choices=['MSOA21CD', 'LAD22CD', 'LSOA21CD', 'OA21CD'], help='Name/Code of the desired geography')
    parser.add_argument('--geo_code', type=str, required=False, help='Geographical variable name')
    
    args = parser.parse_args()

    args_dict = vars(args)
    process = args_dict['process']
    geo_level = args_dict['geo_level']
    geo_code = args_dict['geo_code']
    sub_geo_level = args_dict['sub_geo_level']
    tile_level = args_dict['tile_level']
    
    sedona = get_spark()

    tables = load_tables(sedona)
    
    log_path = Path(f"logs/{process}_processing.log")
    setup_logger(log_path=log_path, log_level=args_dict['log_level'])

    output_areas_boundaries_gdf = tables['output_areas_boundaries_gdf'] 
    os_tile_boundaries_gdf = tables['os_tile_boundaries_gdf'] 
    output_areas_os_tile_overlay_df = tables['output_areas_os_tile_overlay_df']
    output_areas_buildings_overlay_sdf = tables['output_areas_buildings_overlay_sdf']
    vom_raster_paths_df = tables['vom_raster_paths_df'] 
    tree_vector_paths_df = tables['tree_vector_paths_df']

    output_areas_boundaries_sdf = sedona.createDataFrame(output_areas_boundaries_gdf)
    output_areas_boundaries_sdf.createOrReplaceTempView('boundaries')

    if geo_code:
        geo_level_codes = [geo_code]

    else:
        geo_level_codes = tables['output_areas_boundaries_gdf'][geo_level].unique()

    try:

        if args_dict['parallel']:
            logging.debug("Running in parallel")

            with concurrent.futures.ThreadPoolExecutor(max_workers=args_dict['n_workers']) as executor:
                futures = [executor.submit(sum_trees_area, sedona, geo_level, geo_code, sub_geo_level) for geo_code in geo_level_codes]
                
                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Regions processed'):
                    result = future.result()

        else:
            logging.debug("Running sequentially")

            for geo_code in tqdm(geo_level_codes, desc='Regions processed'):   
                result = sum_trees_area(sedona, geo_level, geo_code, sub_geo_level)

    except Exception as e:
        logging.error(f"Error processing {process}: {e}")
        raise e
    



