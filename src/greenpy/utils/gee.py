"""Google Earth Engine session shared by every GEE-backed source."""

import threading

_GEE_READY = False
_LOCK = threading.Lock()


def ensure_gee(project: str | None) -> None:
    """Initialise GEE once per process (avoids re-authenticating per geo code or download thread)."""
    global _GEE_READY
    with _LOCK:
        if not _GEE_READY:
            from ..optional.spectral import setup_gee

            setup_gee(project)
            _GEE_READY = True


def write_raster(da, path, **kwargs) -> None:
    """Write an xee-downloaded DataArray (CRS and nodata already set) to a GeoTIFF, atomically.

    xee records the download scale in encoding["scale_factor"]; rioxarray's
    to_raster would divide the stored values by it and tag the band with that
    scale, which readers such as Sedona's RS_FromGeoTiff ignore — so the
    encoding is dropped and values are written as they are.
    """
    from pathlib import Path

    da = da.copy()
    da.encoding = {}
    path = Path(path)
    tmp = path.with_name(path.name + ".part")
    da.rio.to_raster(tmp, driver="GTiff", **kwargs)
    tmp.replace(path)
