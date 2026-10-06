"""
Canopy cover from the Meta/WRI global canopy height model on AWS (optional module).

The CHM tiles of the configured release (data.meta_chm, v1 or v2) covering
the boundary are downloaded once into the CHM cache (chm_sources), cropped to
the boundary, binarised at native resolution (height within [low, high]) and
reprojected to cfg.crs. The resulting canopy mask flows through the same
zonal-statistics paths as the GEE source: t30.get_canopy_cover_raster and,
as an already-binary tile, t30_buildings.get_canopy_cover_buildings_raster.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import xarray as xr
from loguru import logger
from rasterio.enums import Resampling

from ..config.schema import GreenPyConfig
from ..utils.gee import write_raster
from .chm_sources import boundary_bounds_in, build_vrt, download_meta_tiles, footprint_dates


def meta_cache_path(cfg: GreenPyConfig, name: str, low_threshold: float, high_threshold: float) -> Path:
    """Cache location for a cropped canopy mask, keyed by release and height band."""
    tag = f"{name}_{cfg.data.meta_chm}_h{low_threshold:g}-{high_threshold:g}"
    return Path(cfg.output.base_dir) / "database" / "meta_canopy" / f"{tag}.tif"


def meta_binary_canopy(
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    low_threshold: float,
    high_threshold: float,
    cache_path: Path | None = None,
    overwrite: bool = True,
) -> xr.DataArray:
    """Canopy mask over the boundary as a (band, y, x) DataArray in cfg.crs.

    Canopy pixels (height in [low_threshold, high_threshold]) are 1, other
    mapped pixels 0, unmapped pixels NaN (tile nodata, including v2 pixels
    outside the imagery footprints) — the nodata semantics of
    t30.binarise_tiles. Binarised before reprojection (nearest neighbour), so
    no height is ever interpolated. When cache_path is given the mask is
    cached as GeoTIFF and reused if overwrite is False.
    """
    if cache_path is not None and cache_path.exists() and not overwrite:
        logger.debug(f"Using cached Meta canopy raster {cache_path}")
        return rxr.open_rasterio(cache_path, masked=True)

    from .tree_segmentation import cache_dir_for  # lazy: heavy module

    version = cfg.data.meta_chm or "v1"
    cache_dir = cache_dir_for(cfg)
    paths = download_meta_tiles(geo_boundary_gdf, cache_dir, version=version)
    if not paths:
        raise FileNotFoundError(f"No Meta CHM {version} tiles cover the boundary")
    dates = footprint_dates(paths)
    if dates:
        logger.info(f"Meta CHM {version} imagery dates for this area: {dates[0]} to {dates[1]}")

    key = "_".join(sorted(Path(p).stem for p in paths))
    vrt = build_vrt(paths, cache_dir / "vrt" / f"meta_{version}_{key}.vrt")
    # pad by a few native pixels so reprojection has no empty edge
    bounds = boundary_bounds_in(vrt, geo_boundary_gdf, pad=5.0)
    heights = rxr.open_rasterio(vrt, masked=True).rio.clip_box(*bounds)

    binary = ((heights >= low_threshold) & (heights <= high_threshold)).astype("float32")
    binary = binary.where(heights.notnull()).rio.write_nodata(np.nan)
    da = binary.rio.reproject(cfg.crs, resampling=Resampling.nearest, nodata=np.nan)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        write_raster(da, cache_path)

    return da
