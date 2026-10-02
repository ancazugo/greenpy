"""
Building footprints and parks from Overture Maps (data.buildings / data.parks_sites: overture).

Reads Overture's GeoParquet directly from its public S3 release via the
`overturemaps` package. Coverage is global, but the building `height`
attribute is sparse — many footprints (especially outside major cities) carry
no height, and the Visibility module drops those buildings. Parks come from
the base theme's land_use features (see overture.park_land_use).
"""

import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import Polygon, MultiPolygon


def _read_overture(overture_type: str, polygon_4326: Polygon | MultiPolygon) -> gpd.GeoDataFrame:
    """Features of one Overture type intersecting the polygon (EPSG:4326)."""
    from overturemaps import core

    reader = core.record_batch_reader(overture_type, polygon_4326.bounds)
    if reader is None:
        raise ValueError(f"Could not open the Overture Maps {overture_type} dataset — check network access to S3")
    table = reader.read_all()

    try:
        gdf = gpd.GeoDataFrame.from_arrow(table)
    except ValueError:
        # older releases without geoarrow metadata: geometry column is plain WKB
        import shapely
        df = table.to_pandas()
        gdf = gpd.GeoDataFrame(df, geometry=shapely.from_wkb(df["geometry"]), crs="EPSG:4326")
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    # the S3 read is bbox-based, so trim to the actual study-area polygon
    return gdf[gdf.geometry.intersects(polygon_4326)]


def fetch_overture_buildings(polygon_4326: Polygon | MultiPolygon, crs: str) -> gpd.GeoDataFrame:
    """Fetch Overture building footprints with canonical columns `building_id`/`building_height`."""
    logger.info("Fetching Overture Maps buildings (GeoParquet from S3 — this can take a few minutes)")
    gdf = _read_overture("building", polygon_4326)
    if gdf.empty:
        raise ValueError("Overture returned no building footprints for the study area — check the boundary")

    gdf = gdf.rename(columns={"id": "building_id", "height": "building_height"})
    if "building_height" in gdf.columns:
        gdf["building_height"] = pd.to_numeric(gdf["building_height"], errors="coerce")
        n_total = len(gdf)
        n_missing = int((gdf["building_height"].isna() | (gdf["building_height"] <= 0)).sum())
        if n_missing:
            logger.warning(
                f"Overture buildings: {n_missing}/{n_total} ({n_missing / n_total:.1%}) footprints have no height — "
                "the Visibility module drops NULL-height buildings, so visibility results will be biased "
                "(Overture height coverage is sparse outside major cities)"
            )
    gdf = gdf[[c for c in ("building_id", "building_height", "subtype", "class", "geometry") if c in gdf.columns]]
    logger.info(f"Fetched {len(gdf)} Overture buildings")
    return gdf.to_crs(crs).reset_index(drop=True)


def select_parks(land_use: gpd.GeoDataFrame, park_land_use: list[str]) -> gpd.GeoDataFrame:
    """land_use polygons whose subtype or class is in park_land_use, as park_id/name/subtype/class."""
    wanted = set(park_land_use)
    keep = land_use["subtype"].isin(wanted) | land_use["class"].isin(wanted)
    parks = land_use[keep & land_use.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    names = parks["names"] if "names" in parks.columns else None
    parks["name"] = names.map(lambda n: n.get("primary") if isinstance(n, dict) else None) if names is not None else None
    parks = parks.rename(columns={"id": "park_id"})
    return parks[["park_id", "name", "subtype", "class", "geometry"]].reset_index(drop=True)


def fetch_overture_parks(polygon_4326: Polygon | MultiPolygon, park_land_use: list[str], crs: str) -> gpd.GeoDataFrame:
    """Fetch Overture land_use polygons matching park_land_use as parks (canonical column `park_id`)."""
    logger.info(f"Fetching Overture Maps parks (land_use in {park_land_use})")
    parks = select_parks(_read_overture("land_use", polygon_4326), park_land_use)
    if parks.empty:
        raise ValueError(f"Overture returned no land_use polygons matching {park_land_use} for the study area")
    logger.info(f"Fetched {len(parks)} Overture parks")
    return parks.to_crs(crs)
