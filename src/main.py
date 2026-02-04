#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: train_model.py
Description: Trains a machine learning model using scikit-learn.
Author: Andrés C. Zúñiga-González
Date: 2025-04-03
"""

from tables_setup import load_tables
from utils.paths import database_dir
from utils.logging_config import setup_logger
from utils.sedona_config import get_spark
from t3 import process_geo_code as process_geo_code_t3
from t30 import process_geo_code as process_geo_code_t30
from t300 import process_geo_code as process_geo_code_t300
from spectral import process_geo_code as process_geo_code_spectral
from tree_count import process_geo_code as process_geo_code_tree_count

import argparse
import logging
import inspect
import concurrent.futures
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def main(process, args_dict, geo_code):

    # Define required arguments per process
    process_args = {
        "T3": list(inspect.signature(process_geo_code_t3).parameters.keys()),
        "T30": list(inspect.signature(process_geo_code_t30).parameters.keys()),
        "T300": list(inspect.signature(process_geo_code_t300).parameters.keys()),
        "Spectral": list(inspect.signature(process_geo_code_spectral).parameters.keys()),
        "Tree_count": list(inspect.signature(process_geo_code_tree_count).parameters.keys())
    }

    # Filter only required arguments
    filtered_args = {k: v for k, v in args_dict.items() if k in process_args[process]}
    filtered_args['geo_code'] = geo_code

    # Call the appropriate function with filtered arguments
    match process:
        case "T3": return process_geo_code_t3(**filtered_args)
        case "T30": return process_geo_code_t30(**filtered_args)
        case "T300": return process_geo_code_t300(**filtered_args)
        case "Spectral": return process_geo_code_spectral(**filtered_args)
        case "Tree_count": return process_geo_code_tree_count(**filtered_args)
        case _: raise ValueError(f"Unknown process: {process}")       

if __name__ == "__main__":                     

    parser = argparse.ArgumentParser(description='This script executes the module to calculate the 3-30-300 metric and spectral indexes for all of England.')
    parser.add_argument('--process', type=str, required=True, choices=['T3', 'T30', 'T300', 'Spectral', 'Tree_count'], help='Name of the component of the module to run')
    parser.add_argument('--tile_level', type=str, required=False, default='TILE_NAME_50KM', help='Name/Code of the desired geography')
    parser.add_argument('--geo_level', type=str, required=False, default='LAD22CD', choices=['RGN22CD', 'MSOA21CD', 'LAD22CD', 'LSOA21CD'], help='Name/Code of the desired geography')
    parser.add_argument('--sub_geo_level', type=str, required=False, default='LSOA21CD', choices=['MSOA21CD', 'LAD22CD', 'LSOA21CD', 'OA21CD'], help='Name/Code of the desired geography')
    parser.add_argument('--geo_code', type=str, required=False, help='Geographical variable name')
    parser.add_argument('--query_method', type=str, required=False, default='rdd', choices=['sql', 'rdd'], help='Type of data to use with Apache Sedona')
    parser.add_argument('--buffer', type=int, required=False, default=100, help='Buffer size in meters')
    parser.add_argument('--tree_area', type=int, required=False, default=10, help='Tree area in square meters')
    parser.add_argument('--tree_height', type=int, required=False, default=3, help='Tree height in meters')
    parser.add_argument('--start_date', type=str, required=False, default='2024-01-01', help='Start date for querying remote sensing data')
    parser.add_argument('--end_date', type=str, required=False, default='2024-12-31', help='End date for querying remote sensing data')
    parser.add_argument('--imagery_ee_path', type=str, required=False, default='COPERNICUS/S2_HARMONIZED', help='Imagery name from GEE')
    parser.add_argument('--cloud_coverage', type=float, required=False, default=10.0, help='Cloud Pixel Percentage')
    parser.add_argument('--spectral_indexes', type=str, nargs='+', required=False, default=['NDVI', 'NDWI', 'NDBI'], help='List of indexes to calculate')
    parser.add_argument('--parallel', action='store_true', help='Run job in parallel')
    parser.add_argument('--n_workers', type=int, required=False, default=2, help='Number of workers')
    parser.add_argument('--log_level', type=str, required=False, default='INFO', help='Logging level')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files')

    args = parser.parse_args()

    args_dict = vars(args)
    process = args_dict['process']

    log_path = Path(f"logs/{process}_processing.log")
    setup_logger(log_path=log_path, log_level=args_dict['log_level'])

    sedona = get_spark()
    tables = load_tables(sedona)    
    
    args_dict['sedona'] = sedona
    args_dict.update(tables)

    output_path = database_dir / f"{process}.parquet" if process in ['T30', 'T300', 'Spectral', 'Tree_count'] else database_dir / f"{process}_{args_dict['buffer']}m.parquet"

    if process == 'Spectral':
        from spectral import setup_gee
        setup_gee()

    if args_dict['geo_code']:
        geo_level_codes = [args_dict['geo_code']]

    else:
        geo_level_codes = tables['output_areas_boundaries_gdf'][args_dict['geo_level']].unique()

    logging.info(f"Running process: {process} for {len(geo_level_codes)} regions")
        
    if args_dict['parallel']:
        logging.debug("Running in parallel")

        with concurrent.futures.ThreadPoolExecutor(max_workers=args_dict['n_workers']) as executor:
            futures = [executor.submit(main, process, args_dict, geo_code) for geo_code in geo_level_codes]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Regions processed'):
                result = future.result()

    else:
        logging.debug("Running sequentially")

        for geo_code in tqdm(geo_level_codes, desc='Regions processed'):   
            result = main(process, args_dict, geo_code)

    logging.info(f"All processes completed successfully with {len(geo_level_codes)} records.")
