"""Inputs the raster engine loads once per run: footprints with heights, the ownership overlay, a spatial index."""

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger

from ...config.schema import GreenPyConfig


@dataclass
class VisibilityInputs:
    building_id: np.ndarray  # str
    geoms: np.ndarray  # footprints in cfg.crs
    height: np.ndarray  # metres (every building has one: the heights chain fills defaults)
    height_source: np.ndarray
    tree: shapely.STRtree  # over geoms
    overlay: pd.DataFrame  # building_id + geo level columns (one unit per building)

    @classmethod
    def load(cls, cfg: GreenPyConfig, overlay_path: Path | None = None) -> "VisibilityInputs":
        from ...heights.enrich import load_building_heights

        db = Path(cfg.output.base_dir) / "database"
        buildings = gpd.read_parquet(db / "buildings.parquet", columns=["building_id", "geometry"])
        buildings["building_id"] = buildings["building_id"].astype(str)
        heights = load_building_heights(cfg).drop_duplicates("building_id").set_index("building_id")
        h = heights.reindex(buildings["building_id"])
        if h["building_height"].isna().any():
            raise ValueError("Building heights do not match database/buildings.parquet — rerun `-p Heights`")
        overlay = pd.read_parquet(overlay_path or db / "census_buildings_overlay.parquet")
        overlay["building_id"] = overlay["building_id"].astype(str)
        geoms = shapely.make_valid(buildings.geometry.values)
        logger.info(f"Visibility: {len(buildings)} buildings; heights from {h['height_source'].value_counts().to_dict()}")
        return cls(
            building_id=buildings["building_id"].to_numpy(),
            geoms=np.asarray(geoms),
            height=h["building_height"].to_numpy(dtype=float),
            height_source=h["height_source"].to_numpy(),
            tree=shapely.STRtree(geoms),
            overlay=overlay,
        )

    def owned(self, geo_level: str, geo_code: str) -> np.ndarray:
        """Indices of the buildings that geo_code owns (overlay: representative point in the unit)."""
        ids = self.overlay.loc[self.overlay[geo_level].astype(str) == str(geo_code), "building_id"]
        return np.flatnonzero(np.isin(self.building_id, ids.to_numpy()))
