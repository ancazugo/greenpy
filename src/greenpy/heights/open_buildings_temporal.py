"""Google Open Buildings 2.5D Temporal: building height rasters from Sentinel-2 (2016-2023).

Annual 0.5 m rasters with an effective resolution of ~4 m, covering Africa,
South and Southeast Asia, and Latin America & the Caribbean only. Pixels
whose building_presence is below options.presence_threshold (default 0.5)
are masked; the year's mosaic is fetched at options.resolution metres
(default 2) and summarised by the median over each footprint.
"""

from pathlib import Path

import geopandas as gpd
from loguru import logger

from .base import HeightContext, RasterHeightSource
from .gee import bounds_4326, download_image_tiles, tag, tiles_with_footprints

COLLECTION = "GOOGLE/Research/open-buildings-temporal/v1"
BAND = "building_height"
YEARS = range(2016, 2024)


class OpenBuildingsTemporalHeights(RasterHeightSource):
    license = "CC BY 4.0 (Google)"
    stat = "median"

    def rasters(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> list[Path]:
        import ee

        year = int(self.options.get("year", max(YEARS)))
        if year not in YEARS:
            raise ValueError(f"open_buildings_temporal.year must be in {YEARS.start}-{YEARS.stop - 1}, got {year}")
        threshold = float(self.options.get("presence_threshold", 0.5))
        res = float(self.options.get("resolution", 2.0))
        out_dir = ctx.cache_dir / tag({
            "collection": COLLECTION, "year": year, "presence": threshold, "crs": self.cfg.crs, "res": res,
        })
        minx, miny, maxx, maxy = bounds_4326(buildings)

        def image():
            region = ee.Geometry.Rectangle([minx, miny, maxx, maxy], proj="EPSG:4326", geodesic=False)
            col = ee.ImageCollection(COLLECTION).filterBounds(region).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            if col.size().getInfo() == 0:
                logger.warning(
                    "Open Buildings 2.5D Temporal has no images over the study area — it covers Africa, "
                    "South/Southeast Asia and Latin America & the Caribbean only"
                )
                return ee.Image.constant(0).toFloat().updateMask(0).rename(BAND)
            # mosaic() drops the source projection; pin the native 0.5 m grid back
            img = col.mosaic().setDefaultProjection(col.first().select(BAND).projection())
            return img.select(BAND).updateMask(img.select("building_presence").gte(threshold))

        tiles = tiles_with_footprints(buildings, res)
        return download_image_tiles(image, BAND, self.cfg.crs, res, tiles, out_dir, self.cfg.gee_project)
