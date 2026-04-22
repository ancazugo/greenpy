"""
Google Earth Engine spectral index computation (optional module).
Requires gee_project and gee_boundaries_asset in config.
"""

import time
from loguru import logger
from pathlib import Path

import ee
import pandas as pd

from ..utils.constants import GEE_PROJECT_NAME


def setup_gee(project: str | None = None) -> None:
    project = project or GEE_PROJECT_NAME
    logger.debug(f"Initializing GEE for project: {project}")
    ee.Authenticate()
    ee.Initialize(project=project, opt_url="https://earthengine-highvolume.googleapis.com")


def get_imagery(
    geo_level_filt_fc: "ee.FeatureCollection",
    imagery_ee_path: str,
    start_date: str,
    end_date: str,
    cloud_coverage: float,
    spectral_indexes: list[str],
) -> "ee.Image":
    logger.debug("Querying GEE for imagery")
    union_geom = geo_level_filt_fc.union().geometry()
    import eemont  # noqa: F401 — registers .spectralIndices() on ee.ImageCollection
    return (
        ee.ImageCollection(imagery_ee_path)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_coverage))
        .filterBounds(union_geom)
        .spectralIndices(spectral_indexes)
        .max()
        .select(spectral_indexes)
    )


def calculate_median_index(
    imagery_ic: "ee.Image",
    geometries: "ee.FeatureCollection",
    scale: float = 10.0,
    tile_scale: int = 4,
) -> "ee.FeatureCollection":
    return imagery_ic.reduceRegions(
        collection=geometries,
        reducer=ee.Reducer.median(),
        scale=scale,
        tileScale=tile_scale,
    )


def process_geo_code(
    geo_code: str,
    geo_level: str,
    sub_geo_level: str,
    imagery_ee_path: str,
    start_date: str,
    end_date: str,
    cloud_coverage: float,
    spectral_indexes: list[str],
    output_dir: Path,
    gee_boundaries_asset: str,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    start_time = time.time()
    logger.info(f"Spectral: processing {geo_code}")

    out_path = output_dir / f"Spectral_{geo_code}.csv"
    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        boundaries_fc = ee.FeatureCollection(gee_boundaries_asset)
        filt_fc = boundaries_fc.filter(ee.Filter.eq(geo_level, geo_code))
        sub_values = filt_fc.aggregate_array(sub_geo_level).distinct()
        imagery_ic = get_imagery(filt_fc, imagery_ee_path, start_date, end_date, cloud_coverage, spectral_indexes)

        def dissolve_by_code(sub_code):
            code = ee.String(sub_code)
            geom = boundaries_fc.filter(ee.Filter.eq(sub_geo_level, code)).union().geometry()
            return ee.Feature(geom).set(sub_geo_level, code)

        sub_boundaries_fc = ee.FeatureCollection(sub_values.map(dissolve_by_code))
        results = calculate_median_index(imagery_ic, sub_boundaries_fc).getInfo()["features"]
        geo_spectral_df = pd.DataFrame([f["properties"] for f in results])

        geo_spectral_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"Spectral: {geo_code} — {len(geo_spectral_df)} records in {end_time - start_time:.2f}s")
        return geo_spectral_df

    except Exception as e:
        logger.error(f"Spectral: error processing {geo_code}: {e}")
