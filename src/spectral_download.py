
from utils.paths import OUTPUT_DIR
from utils.logging_config import setup_logger

import time
import logging
import ee
import eemont
import argparse
import logging
from pathlib import Path
from spectral import *

def download_imagery(imagery_ic: ee.image.Image, output_path: str) -> None:
    """
    Download Earth Engine imagery to Google Drive
    
    Args:
        imagery_ic: Earth Engine image to download
        output_path: Path in Google Drive where the file will be saved
    """
    logging.debug(f"Downloading imagery to Google Drive: {output_path}")
    
    # Export to Google Drive
    task = ee.batch.Export.image.toDrive(
        image=imagery_ic,
        description=f"Spectral_imagery_{int(time.time())}",
        folder='EarthEngine_Exports',  # Folder in Google Drive
        fileNamePrefix=output_path,
        scale=10,  # 10m resolution, adjust as needed
        region=imagery_ic.geometry(),
        fileFormat='GeoTIFF',
        maxPixels=1e13
    )
    
    task.start()
    
    # Wait for the task to complete
    logging.info("Export task started. Waiting for completion...")
    while task.status()['state'] in ['READY', 'RUNNING']:
        logging.info(f"Task status: {task.status()['state']}")
        time.sleep(30)  # Check every 30 seconds
    
    if task.status()['state'] == 'COMPLETED':
        logging.info("Download completed successfully!")
    else:
        logging.error(f"Download failed: {task.status()}")
        raise Exception(f"Download failed: {task.status()}")

def process_gee(imagery_ee_path: str, start_date: str, end_date: str, cloud_coverage: float, spectral_indexes: list[str]) -> None:
    
    setup_gee()


    spectral_index_path = f"spectral_{start_date}_{end_date}.tif"

    england_boundary_fc = ee.FeatureCollection("FAO/GAUL/2015/level1").filter(ee.Filter.eq('ADM0_NAME', 'England'))

    imagery_ic = get_imagery(england_boundary_fc, imagery_ee_path, start_date, end_date, cloud_coverage, spectral_indexes)

    download_imagery(imagery_ic, spectral_index_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='This script executes the module to download spectral indexes from GEE.')
    parser.add_argument('--start_date', type=str, required=False, default='2024-01-01', help='Start date for querying remote sensing data')
    parser.add_argument('--end_date', type=str, required=False, default='2024-12-31', help='End date for querying remote sensing data')
    parser.add_argument('--imagery_ee_path', type=str, required=False, default='COPERNICUS/S2_HARMONIZED', help='Imagery name from GEE')
    parser.add_argument('--cloud_coverage', type=float, required=False, default=10.0, help='Cloud Pixel Percentage')
    parser.add_argument('--spectral_indexes', type=str, nargs='+', required=False, default=['NDVI', 'NDWI', 'NDBI'], help='List of indexes to calculate')
    
    args = parser.parse_args()

    log_path = Path(f"logs/spectral_download.log")
    setup_logger(log_path=log_path, log_level="DEBUG")
    
    process_gee(args.imagery_ee_path, args.start_date, args.end_date, args.cloud_coverage, args.spectral_indexes)

    logging.info("Spectral indexes downloaded successfully!")
