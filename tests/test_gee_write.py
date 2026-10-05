"""xee download scale must not leak into cached GeoTIFF values."""

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (registers .rio)
import xarray as xr

from greenpy.utils.gee import write_raster


def test_write_raster_ignores_xee_scale_factor(tmp_path):
    da = xr.DataArray(
        np.array([[0.0, 0.5], [1.0, np.nan]], dtype="float32"), dims=("y", "x"),
        coords={"y": [15.0, 5.0], "x": [5.0, 15.0]},
    )
    da.encoding = {"scale_factor": 10.0, "dtype": "float32"}  # as xee sets it at a 10 m download
    da = da.rio.write_crs("EPSG:32618").rio.write_nodata(float("nan"))
    path = tmp_path / "c.tif"
    write_raster(da, path)
    with rasterio.open(path) as src:
        assert src.scales == (1.0,)
        out = src.read(1)
    assert out[0, 1] == 0.5 and out[1, 0] == 1.0 and np.isnan(out[1, 1])
    assert da.encoding["scale_factor"] == 10.0  # caller's array untouched
