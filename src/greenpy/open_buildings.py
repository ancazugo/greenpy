"""
Building footprints from Google Open Buildings v3 (data.buildings: open_buildings).

Queries the polygons FeatureCollection through Google Earth Engine, so it needs
`gee_project` (or the GEE_PROJECT_NAME env var) and a prior `earthengine
authenticate`. The dataset covers Africa, South and Southeast Asia, and Latin
America & the Caribbean only — NOT Europe or North America — and ships no
building heights: Visibility needs heights.sources beyond `native` (e.g.
open_buildings_temporal, gba, ghs_built_h).
"""

import geopandas as gpd
from loguru import logger
from shapely.geometry import Polygon, MultiPolygon, mapping

from .utils.gee import ensure_gee

OPEN_BUILDINGS_ASSET = "GOOGLE/Research/open-buildings/v3/polygons"


def fetch_open_buildings(
    polygon_4326: Polygon | MultiPolygon,
    confidence_threshold: float,
    crs: str,
    gee_project: str | None,
) -> gpd.GeoDataFrame:
    """Fetch Open Buildings v3 footprints with canonical column `building_id` (no heights)."""
    import ee

    logger.warning(
        "Google Open Buildings provides no building heights — add height sources to heights.sources "
        "(e.g. open_buildings_temporal, gba) before running Visibility"
    )
    logger.warning(
        "Google Open Buildings v3 covers Africa, South and Southeast Asia, and Latin America & "
        "the Caribbean only — Europe and North America are NOT covered"
    )

    ensure_gee(gee_project)
    region = ee.Geometry(mapping(polygon_4326))
    fc = (
        ee.FeatureCollection(OPEN_BUILDINGS_ASSET)
        .filterBounds(region)
        .filter(ee.Filter.gte("confidence", confidence_threshold))
    )
    logger.info(f"Fetching Open Buildings polygons (confidence >= {confidence_threshold})")
    gdf = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    if gdf.empty:
        raise ValueError(
            "Google Open Buildings returned no footprints for the study area — the boundary is likely "
            "outside dataset coverage (Africa, South/Southeast Asia, Latin America & Caribbean only; "
            f"NOT Europe or North America), or open_buildings.confidence_threshold ({confidence_threshold}) "
            "is too high"
        )
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    # filterBounds matches by intersection with the region bounds server-side; trim to the polygon
    gdf = gdf[gdf.geometry.intersects(polygon_4326)]
    if "full_plus_code" in gdf.columns:
        gdf["building_id"] = gdf["full_plus_code"]
    else:
        gdf["building_id"] = [f"ob_{i}" for i in range(len(gdf))]
    gdf = gdf[[c for c in ("building_id", "confidence", "geometry") if c in gdf.columns]]
    logger.info(f"Fetched {len(gdf)} Open Buildings footprints")
    return gdf.to_crs(crs).reset_index(drop=True)
