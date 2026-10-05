"""Height source interface shared by every source in the heights chain."""

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ..config.schema import GreenPyConfig, HeightSourceSpec

# Columns every HeightSource.heights() result carries (one row per building).
# `label` is optional and overrides the source label in height_source for that
# row (e.g. native heights derived from floor counts).
RESULT_COLUMNS = ["building_id", "height", "quality", "res_m"]


@dataclass
class HeightContext:
    """Where a source may cache downloads, and the study area it serves."""

    cfg: GreenPyConfig
    # Raw downloads (GEE rasters, feature chunks), shared across runs: <chm cache>/heights/<source>
    cache_dir: Path
    # Study-area census boundaries in cfg.crs (the fetch region for remote sources)
    boundaries: gpd.GeoDataFrame


class HeightSource(ABC):
    """A source of building heights; see greenpy.heights for the registry."""

    # "native" (footprint attributes), "vector" (overlap match) or "raster" (zonal statistic)
    kind: str = ""
    license: str = ""
    # Bump to invalidate cached results when a source's logic changes
    version: int = 1

    def __init__(self, spec: HeightSourceSpec, cfg: GreenPyConfig):
        self.spec = spec
        self.cfg = cfg
        self.options = dict(spec.options)

    @property
    def label(self) -> str:
        return self.spec.label

    def cache_key(self) -> str:
        """Hash of everything that changes this source's heights for given footprints."""
        payload = {
            "source": self.spec.source, "options": self.options, "version": self.version,
            "storey_height": self.cfg.heights.storey_height, "min_overlap": self.cfg.heights.min_overlap,
            "crs": self.cfg.crs,
        }
        return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]

    @abstractmethod
    def heights(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> pd.DataFrame:
        """Heights for `buildings` (building_id + geometry in cfg.crs) as RESULT_COLUMNS.

        height is NaN where the source has no value; quality is in [0, 1]
        (share of the footprint the source covers); res_m is the source's
        spatial resolution (0 for exact vector attributes).
        """


class VectorHeightSource(HeightSource):
    """Polygons with a height attribute, matched to footprints by area overlap."""

    kind = "vector"

    @abstractmethod
    def fetch(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> gpd.GeoDataFrame:
        """Source polygons with a numeric `height` column covering the footprints, in cfg.crs."""

    def heights(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> pd.DataFrame:
        from .match import match_by_overlap

        source = self.fetch(buildings, ctx)
        return match_by_overlap(buildings, source, self.cfg.heights.min_overlap)


class RasterHeightSource(HeightSource):
    """A height raster summarised per footprint (zonal statistic over its pixels)."""

    kind = "raster"
    stat: str = "median"

    @abstractmethod
    def rasters(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> list[Path]:
        """Single-band height rasters (shared CRS and pixel size, NaN/nodata = no value) covering the footprints."""

    def heights(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> pd.DataFrame:
        from .match import zonal_heights

        return zonal_heights(buildings, self.rasters(buildings, ctx), self.stat, ctx.cache_dir)


def empty_result(buildings: gpd.GeoDataFrame) -> pd.DataFrame:
    """All-missing result for a source that covers none of the footprints."""
    return pd.DataFrame({
        "building_id": buildings["building_id"].values,
        "height": float("nan"), "quality": 0.0, "res_m": float("nan"),
    })
