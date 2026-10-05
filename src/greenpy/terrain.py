"""Ground elevation (DTM) under buildings and trees, for Visibility's raster engine.

Visibility needs bare-earth ground: building and tree heights are measured
from it, and the ground itself (hills, valleys) blocks or opens views.
FABDEM — Copernicus GLO-30 with buildings and forests removed — is the
default; Copernicus GLO-30 and NASADEM are surface models (in cities they
include rooftops and canopy, so building heights would be counted twice).
GEE sources are downloaded at terrain.resolution metres onto tiles aligned to
a fixed grid in the study CRS (cached under the CHM cache dir) and resampled
bilinearly; a local DTM raster file or directory can be given instead.
"""

from pathlib import Path

from .config.schema import GreenPyConfig

# name -> (GEE asset, band, is ImageCollection, licence)
DEM_SOURCES = {
    "fabdem": ("projects/sat-io/open-datasets/FABDEM", "b1", True, "CC BY-NC-SA 4.0 (University of Bristol)"),
    "copernicus": ("COPERNICUS/DEM/GLO30", "DEM", True, "Copernicus DEM licence (ESA)"),
    "nasadem": ("NASA/NASADEM_HGT/001", "elevation", False, "public domain (NASA)"),
}


def uses_terrain(cfg: GreenPyConfig) -> bool:
    return cfg.terrain.source is not None


def dem_layers(cfg: GreenPyConfig, bounds: tuple) -> list[Path]:
    """DEM rasters covering bounds (cfg.crs): downloaded GEE tiles or the local DTM files."""
    source = cfg.terrain.source
    if source is None:
        return []
    if source in DEM_SOURCES:
        return _gee_dem(cfg, source, bounds)
    path = Path(source)
    if path.is_dir():
        paths = sorted(path.rglob("*.tif"))
    elif path.exists():
        paths = [path]
    else:
        raise FileNotFoundError(
            f"terrain.source {source!r} is neither one of {sorted(DEM_SOURCES)} nor an existing raster path"
        )
    if not paths:
        raise FileNotFoundError(f"No .tif DEM rasters under {path}")
    return paths


def _gee_dem(cfg: GreenPyConfig, source: str, bounds: tuple) -> list[Path]:
    from .heights.gee import download_image_tiles, tag, tiles_for_extent
    from .optional.tree_segmentation import cache_dir_for

    asset, band, is_collection, _ = DEM_SOURCES[source]
    res = float(cfg.terrain.resolution)
    out_dir = cache_dir_for(cfg) / "terrain" / source / tag({"asset": asset, "crs": cfg.crs, "res": res})

    def image():
        import ee

        if is_collection:
            col = ee.ImageCollection(asset).select(band)
            img = col.mosaic().setDefaultProjection(col.first().projection())
        else:
            img = ee.Image(asset).select(band)
        return img.resample("bilinear").rename(band)

    tiles = tiles_for_extent(bounds, res)
    return download_image_tiles(image, band, cfg.crs, res, tiles, out_dir, cfg.gee_project)
