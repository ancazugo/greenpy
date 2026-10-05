"""GHS-BUILT-H R2023A (JRC): average net building height on a global 100 m grid (2018).

Too coarse to describe single buildings — it gives every footprint the mean
height of the built-up cells it sits in — so it belongs at the end of the
chain, as a global fallback. Cells without buildings (0 m) are masked. The
100 m cells are fetched by nearest-neighbour onto a finer grid
(options.resolution, default 10 m) so the per-footprint mean approximates
an area-weighted mean of the cells it overlaps.
"""

from pathlib import Path

import geopandas as gpd

from .base import HeightContext, RasterHeightSource
from .gee import download_image_tiles, tag, tiles_with_footprints

ASSET = "JRC/GHSL/P2023A/GHS_BUILT_H/2018"
BAND = "built_height"


class GHSBuiltHeights(RasterHeightSource):
    license = "CC BY 4.0 (European Commission, JRC)"
    stat = "mean"

    def rasters(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> list[Path]:
        import ee

        res = float(self.options.get("resolution", 10.0))
        out_dir = ctx.cache_dir / tag({"asset": ASSET, "band": BAND, "crs": self.cfg.crs, "res": res})

        def image():
            img = ee.Image(ASSET).select(BAND)
            return img.updateMask(img.gt(0))

        tiles = tiles_with_footprints(buildings, res)
        return download_image_tiles(image, BAND, self.cfg.crs, res, tiles, out_dir, self.cfg.gee_project)

