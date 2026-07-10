"""
Building footprints from Overture Maps (data.buildings: overture).

Reads the buildings theme GeoParquet directly from Overture's public S3
release via the `overturemaps` package. Coverage is global, but the `height`
attribute is sparse — many footprints (especially outside major cities) carry
no height, and the Visibility module drops those buildings.
"""

import pandas as pd
import geopandas as gpd
from loguru import logger
from shapely.geometry import Polygon, MultiPolygon


def fetch_overture_buildings(polygon_4326: Polygon | MultiPolygon, crs: str) -> gpd.GeoDataFrame:
    """Fetch Overture building footprints with canonical columns `building_id`/`building_height`."""
    from overturemaps import core

    logger.info("Fetching Overture Maps buildings (GeoParquet from S3 — this can take a few minutes)")
    reader = core.record_batch_reader("building", polygon_4326.bounds)
    if reader is None:
        raise ValueError("Could not open the Overture Maps buildings dataset — check network access to S3")
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
    gdf = gdf[gdf.geometry.intersects(polygon_4326)]
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
