"""
Canopy cover from a Google Earth Engine canopy-height dataset (optional module).

Binarisation (height within [low, high]) runs server-side in GEE at the
dataset's native resolution; when a coarser download scale is requested the
mask is averaged (reduceResolution) into canopy *fraction* per output pixel, so
cover stays unbiased instead of thresholding GEE's height pyramids. The result
is pulled down with xee as an xarray array and flows through the same
zonal-statistics path as local CHM tiles (t30.get_canopy_cover_raster).

Designed for the global 1 m Meta/WRI dataset
`projects/sat-io/open-datasets/facebook/meta-canopy-height`, but works with any
single-band canopy-height ImageCollection.
"""

import math
from pathlib import Path

import ee
import affine
import xarray as xr
import rioxarray as rxr
import geopandas as gpd
from loguru import logger

from ..config.schema import GreenPyConfig
from ..utils.gee import ensure_gee, write_raster


def gee_cache_path(cfg: GreenPyConfig, name: str, low_threshold: float, high_threshold: float, scale: float) -> Path:
    """Cache location for a downloaded canopy raster, keyed by every parameter that changes its pixels."""
    tag = f"{name}_h{low_threshold:g}-{high_threshold:g}_s{scale:g}m"
    return Path(cfg.output.base_dir) / "database" / "gee_canopy" / f"{tag}.tif"


def download_binary_canopy(
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    low_threshold: float,
    high_threshold: float,
    scale: float = 1.0,
    cache_path: Path | None = None,
    overwrite: bool = True,
) -> xr.DataArray:
    """Download a server-side canopy mask for the boundary as a (band, y, x) DataArray.

    At the dataset's native scale, canopy pixels (height in [low_threshold,
    high_threshold]) are 1 and other mapped pixels 0; at a coarser `scale`
    each pixel holds the canopy fraction of the native pixels it covers.
    Unmapped pixels are NaN — matching the nodata semantics of
    t30.binarise_tiles so get_canopy_cover_raster works unchanged. When
    cache_path is given the array is cached as GeoTIFF and reused if
    overwrite is False (use gee_cache_path so the name encodes the settings).
    """
    if cache_path is not None and cache_path.exists() and not overwrite:
        logger.debug(f"Using cached GEE canopy raster {cache_path}")
        return rxr.open_rasterio(cache_path, masked=True)

    ensure_gee(cfg.gee_project)
    logger.info(f"Downloading GEE canopy mask from {cfg.data.canopy_height_ee_path} at {scale} m")

    # Build the output pixel grid directly in the projected CRS (metres), so GEE
    # reprojects the dataset onto it. xee 0.1.x takes crs + crs_transform + shape.
    minx, miny, maxx, maxy = geo_boundary_gdf.dissolve().total_bounds
    width = max(1, math.ceil((maxx - minx) / scale))
    height = max(1, math.ceil((maxy - miny) / scale))
    transform = affine.Affine(scale, 0.0, minx, 0.0, -scale, maxy)

    collection = ee.ImageCollection(cfg.data.canopy_height_ee_path)
    # mosaic() drops the source projection; pin it back so the threshold is
    # evaluated on native pixels rather than on GEE's averaged height pyramids
    native = collection.first().projection()
    img = collection.mosaic().setDefaultProjection(native)
    # toFloat() so masked pixels (filled with xee's int32 sentinel) become NaN cleanly
    binary = (
        img.gte(low_threshold)
        .And(img.lte(high_threshold))
        .toFloat()
        .rename("canopy")
    )
    native_scale = native.nominalScale().getInfo()
    if scale > native_scale * 1.01:
        # canopy fraction of the native pixels inside each output pixel
        logger.info(f"Aggregating {native_scale:.2f} m canopy mask to {scale} m canopy fraction")
        binary = binary.reduceResolution(ee.Reducer.mean(), maxPixels=65535)

    ds = xr.open_dataset(
        ee.ImageCollection([binary]),
        engine="ee",
        crs=cfg.crs,
        crs_transform=transform,
        shape_2d=(width, height),
    )

    da = ds["canopy"]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    # xee returns dims (X, Y); orient to (y, x) descending-y like a GeoTIFF
    rename = {d: ("x" if d.lower() in ("x", "lon") else "y") for d in da.dims}
    da = da.rename(rename).transpose("y", "x")
    da = da.sortby("y", ascending=False).sortby("x")
    da = da.rio.write_crs(cfg.crs).expand_dims("band")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        # tag NaN as nodata so raster readers (rioxarray masked=True, Sedona
        # RS_ZonalStats) can exclude unmapped pixels from pixel counts
        da = da.rio.write_nodata(float("nan"))
        write_raster(da, cache_path)

    return da
