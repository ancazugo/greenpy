"""
Canopy cover from a Google Earth Engine canopy-height dataset (optional module).

Binarisation (height within [low, high]) runs server-side in GEE; the binary
mask is pulled down with xee as an xarray array and then flows through the same
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
from .spectral import setup_gee

_GEE_READY = False


def _ensure_gee(project: str | None) -> None:
    """Initialise GEE once per process (avoids re-authenticating per geo code)."""
    global _GEE_READY
    if not _GEE_READY:
        setup_gee(project)
        _GEE_READY = True


def download_binary_canopy(
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    low_threshold: float,
    high_threshold: float,
    scale: float = 1.0,
    cache_path: Path | None = None,
    overwrite: bool = True,
) -> xr.DataArray:
    """Download a server-side binarised canopy mask for the boundary as a (band, y, x) DataArray.

    Canopy pixels (height in [low_threshold, high_threshold]) are 1, other
    mapped pixels 0, and unmapped pixels NaN — matching the shape and nodata
    semantics of t30.binarise_tiles so get_canopy_cover_raster works unchanged.
    When cache_path is given the array is cached as GeoTIFF and reused if
    overwrite is False.
    """
    if cache_path is not None and cache_path.exists() and not overwrite:
        logger.debug(f"Using cached GEE canopy raster {cache_path}")
        return rxr.open_rasterio(cache_path, masked=True)

    _ensure_gee(cfg.gee_project)
    logger.info(f"Downloading GEE canopy mask from {cfg.data.canopy_height_ee_path} at {scale} m")

    # Build the output pixel grid directly in the projected CRS (metres), so GEE
    # reprojects the dataset onto it. xee 0.1.x takes crs + crs_transform + shape.
    minx, miny, maxx, maxy = geo_boundary_gdf.dissolve().total_bounds
    width = max(1, math.ceil((maxx - minx) / scale))
    height = max(1, math.ceil((maxy - miny) / scale))
    transform = affine.Affine(scale, 0.0, minx, 0.0, -scale, maxy)

    img = ee.ImageCollection(cfg.data.canopy_height_ee_path).mosaic()
    # toFloat() so masked pixels (filled with xee's int32 sentinel) become NaN cleanly
    binary = (
        img.gte(low_threshold)
        .And(img.lte(high_threshold))
        .toFloat()
        .rename("canopy")
    )

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
        da.rio.to_raster(cache_path)

    return da
