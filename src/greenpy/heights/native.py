"""Heights already carried by the footprints in data.buildings.

`building_height` (Overture `height`, OSM `height`, or a file's
columns.building_height_col) is used as is; footprints without it fall back
to a floor count x heights.storey_height, from options.levels_col or, when
unset, `num_floors` (Overture, OSM building:levels) if present.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

from .base import HeightContext, HeightSource

DEFAULT_LEVELS_COLS = ("num_floors",)


class NativeHeights(HeightSource):
    kind = "native"

    def heights(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> pd.DataFrame:
        n = len(buildings)
        height = (
            pd.to_numeric(buildings["building_height"], errors="coerce").to_numpy(dtype=float)
            if "building_height" in buildings.columns else np.full(n, np.nan)
        )
        label = np.full(n, None, dtype=object)
        quality = np.where(np.isfinite(height), 1.0, 0.0)

        levels_col = self.options.get("levels_col") or next(
            (c for c in DEFAULT_LEVELS_COLS if c in buildings.columns), None
        )
        if levels_col is not None:
            if levels_col not in buildings.columns:
                raise ValueError(
                    f"heights source native: levels_col '{levels_col}' not in the buildings columns "
                    f"{list(buildings.columns)} — delete database/buildings.parquet if it was added to the input since"
                )
            levels = pd.to_numeric(buildings[levels_col], errors="coerce").to_numpy(dtype=float)
            from_levels = ~(np.isfinite(height) & (height > 0)) & np.isfinite(levels) & (levels > 0)
            height = np.where(from_levels, levels * self.cfg.heights.storey_height, height)
            label[from_levels] = f"{self.label}_levels"
            quality[from_levels] = 0.5

        return pd.DataFrame({
            "building_id": buildings["building_id"].values,
            "height": height, "quality": quality, "res_m": 0.0, "label": label,
        })
