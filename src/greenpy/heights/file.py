"""Local building heights: a vector file with a height column, or height rasters.

options.path is a vector file (matched by footprint overlap like the remote
vector sources; options.column, default `height`; options.layer for
multi-layer files) or a raster file / directory of .tif rasters sharing CRS
and pixel size (e.g. an nDSM or GBA.Height 3 m tiles), summarised per
footprint by options.stat (median, mean or max; default median).
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from .base import HeightContext, HeightSource
from .match import match_by_overlap, zonal_heights

RASTER_SUFFIXES = (".tif", ".tiff", ".vrt", ".img")


class FileHeights(HeightSource):
    def __init__(self, spec, cfg):
        super().__init__(spec, cfg)
        self.path = Path(self.options["path"])
        self.kind = "raster" if self.path.is_dir() or self.path.suffix.lower() in RASTER_SUFFIXES else "vector"

    def cache_key(self) -> str:
        # a changed file must not reuse cached heights
        import hashlib
        files = sorted(self.path.rglob("*.tif")) if self.path.is_dir() else [self.path]
        stamp = "|".join(f"{p}:{p.stat().st_size}:{p.stat().st_mtime_ns}" for p in files)
        return hashlib.sha1((super().cache_key() + stamp).encode()).hexdigest()[:10]

    def heights(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> pd.DataFrame:
        if not self.path.exists():
            raise FileNotFoundError(f"heights source file: {self.path} not found")
        if self.kind == "raster":
            paths = sorted(self.path.rglob("*.tif")) if self.path.is_dir() else [self.path]
            return zonal_heights(buildings, paths, self.options.get("stat", "median"), ctx.cache_dir)
        source = self._read_vector(buildings)
        return match_by_overlap(buildings, source, self.cfg.heights.min_overlap)

    def _read_vector(self, buildings: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        column = self.options.get("column", "height")
        if self.path.suffix.lower() in (".parquet", ".geoparquet"):
            gdf = gpd.read_parquet(self.path)
        else:
            kwargs = {"layer": self.options["layer"]} if self.options.get("layer") else {}
            gdf = gpd.read_file(self.path, **kwargs)
        if column not in gdf.columns:
            raise ValueError(f"heights source file: column {column!r} not in {self.path} ({list(gdf.columns)})")
        gdf = gdf.rename(columns={column: "height"})[["height", "geometry"]]
        gdf = gdf.to_crs(self.cfg.crs) if gdf.crs is not None else gdf.set_crs(self.cfg.crs)
        return gdf.cx[slice(*buildings.total_bounds[[0, 2]]), slice(*buildings.total_bounds[[1, 3]])]
