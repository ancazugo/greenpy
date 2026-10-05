"""Inputs the raster engine loads once per run: footprints with heights, the ownership overlay, a spatial index."""

from dataclasses import dataclass, field
from pathlib import Path

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
    overlay: pd.DataFrame  # building_id, pos (index into the arrays above) + geo level columns
    _codes: dict = field(default_factory=dict)  # geo level -> codes as str, per overlay row

    @classmethod
    def load(cls, cfg: GreenPyConfig, overlay_path: Path | None = None) -> "VisibilityInputs":
        from ...heights.enrich import load_building_heights
        from ...pipeline import all_buildings, ensure_context_buildings

        db = Path(cfg.output.base_dir) / "database"
        ensure_context_buildings(cfg)
        # context-ring buildings have no overlay row: obstacles only, never observers
        buildings = all_buildings(cfg, columns=["building_id", "geometry"])
        heights = load_building_heights(cfg).drop_duplicates("building_id").set_index("building_id")
        h = heights.reindex(buildings["building_id"])
        if h["building_height"].isna().any():
            raise ValueError("Building heights do not match database/buildings.parquet — rerun `-p Heights`")
        overlay = pd.read_parquet(overlay_path or db / "census_buildings_overlay.parquet")
        overlay["building_id"] = overlay["building_id"].astype(str)
        # each overlay row's index into the footprint arrays, so owned() is a plain filter
        pos = pd.Series(np.arange(len(buildings)), index=buildings["building_id"].to_numpy())
        overlay["pos"] = overlay["building_id"].map(pos[~pos.index.duplicated()])
        overlay = overlay[overlay["pos"].notna()].astype({"pos": np.int64})
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
        if geo_level not in self._codes:
            self._codes[geo_level] = self.overlay[geo_level].astype(str).to_numpy()
        return np.sort(self.overlay["pos"].to_numpy()[self._codes[geo_level] == str(geo_code)])
