"""UT-GLOBUS (Kamath et al. 2024): building heights for ~1,100 cities from ICESat-2/GEDI and machine learning.

Published on GEE as one FeatureCollection per city under the sat-io community
catalog (options.city, e.g. `bogota`, `london`; `cambridge` is Cambridge,
Massachusetts). Heights of 0 m mean "no estimate". Licensed CC BY 4.0.
"""

import difflib

import geopandas as gpd

from .base import HeightContext, VectorHeightSource
from .gee import bounds_4326, fetch_features, list_asset_names

ASSET_ROOT = "projects/sat-io/open-datasets/UT-GLOBUS"


def list_cities(ctx: HeightContext, project: str | None) -> list[str]:
    """City collection names under ASSET_ROOT (cached in the source cache dir)."""
    return list_asset_names(ASSET_ROOT, ctx.cache_dir / "cities.json", project)


class UTGlobusHeights(VectorHeightSource):
    license = "CC BY 4.0 (Kamath et al. 2024)"

    def fetch(self, buildings: gpd.GeoDataFrame, ctx: HeightContext) -> gpd.GeoDataFrame:
        city = str(self.options["city"]).strip().lower()
        cities = list_cities(ctx, self.cfg.gee_project)
        if city not in cities:
            close = difflib.get_close_matches(city, cities, n=5)
            raise ValueError(f"UT-GLOBUS has no city {city!r}; closest names: {close} ({len(cities)} cities)")
        gdf = fetch_features(
            [(f"{ASSET_ROOT}/{city}", bounds_4326(buildings))], ctx.cache_dir / "chunks",
            self.cfg.gee_project, properties=["height"],
        )
        return gdf.to_crs(self.cfg.crs)
