"""
OSM data sources. Fetches buildings, roads, parks and park access points from
OpenStreetMap (via osmnx/Overpass) for any layer whose config path is "osm",
producing GeoDataFrames with the same canonical columns as user-provided files.
"""

import geopandas as gpd
import osmnx as ox
import pandas as pd
from loguru import logger
from osmnx._errors import InsufficientResponseError
from shapely.geometry import MultiPolygon, Polygon

# Residential building=* values fetched by default; override with osm.building_types.
DEFAULT_BUILDING_TYPES = [
    "residential", "house", "apartments", "detached", "semidetached_house",
    "terrace", "bungalow", "dormitory", "static_caravan", "houseboat",
]

# Public green spaces fetched by default; override with osm.park_tags.
# leisure=garden is excluded on purpose (mostly private gardens), as are
# pitches/courts (never queried).
DEFAULT_PARK_TAGS = {
    "leisure": ["park", "recreation_ground"],
    "landuse": ["village_green", "recreation_ground"],
}

# access=* values that mark a feature as not publicly accessible
PRIVATE_ACCESS_VALUES = ("private", "no", "customers")

# Max distance (m) from a park boundary for an OSM entrance/gate node to count
# as an access point to that park
ENTRANCE_SNAP_DISTANCE = 15

# Min spacing (m) between derived boundary/road intersection points per park
ACCESS_POINT_SPACING = 1


def build_query_polygon(census_gdf: gpd.GeoDataFrame, buffer_m: int = 0) -> Polygon | MultiPolygon:
    """Union of the census boundaries in EPSG:4326, optionally buffered.

    The buffer is applied in the projected CRS (metres) before reprojecting.
    """
    geom = census_gdf.geometry
    if buffer_m:
        geom = geom.buffer(buffer_m)
    return geom.to_crs("EPSG:4326").union_all()


def _stringify_list_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Join list-valued OSM tag values into strings so the frame can be written to parquet."""
    for col in gdf.columns:
        if col == gdf.geometry.name:
            continue
        if gdf[col].map(lambda v: isinstance(v, list)).any():
            gdf[col] = gdf[col].map(lambda v: ", ".join(map(str, v)) if isinstance(v, list) else v)
    return gdf


def _osm_ids(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Build 'element/id' identifiers (e.g. 'way/123456') from a reset features index."""
    return gdf["element"].astype(str) + "/" + gdf["id"].astype(str)


def fetch_osm_buildings(polygon_4326: Polygon | MultiPolygon, building_types: list[str] | None, crs: str) -> gpd.GeoDataFrame:
    """Fetch building footprints from OSM with canonical column `building_id`.

    building_types are values of the OSM building=* tag; ["all"] fetches every
    building regardless of type.
    """
    types = building_types or DEFAULT_BUILDING_TYPES
    tags = {"building": True if "all" in types else types}
    logger.info(f"Fetching OSM buildings (types: {'all' if tags['building'] is True else types})")
    try:
        gdf = ox.features_from_polygon(polygon_4326, tags).reset_index()
    except InsufficientResponseError:
        raise ValueError("OSM returned no buildings for the study area — check the boundary or osm.building_types")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        raise ValueError("OSM returned no building polygons for the study area — check the boundary or osm.building_types")
    gdf["building_id"] = _osm_ids(gdf)
    gdf = gdf[[c for c in ("building_id", "building", "geometry") if c in gdf.columns]]
    logger.info(f"Fetched {len(gdf)} OSM buildings")
    return _stringify_list_columns(gdf.to_crs(crs).reset_index(drop=True))


def fetch_osm_roads(polygon_4326: Polygon | MultiPolygon, network_type: str, crs: str) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Fetch the OSM road network as canonical (edges, nodes) GeoDataFrames.

    Edge lengths come from osmnx in metres, matching the Dijkstra weight used
    by T300. Only the largest connected component is kept (osmnx default) —
    disconnected fragments (clipped stubs, courtyard paths) would otherwise
    capture nearest-node snaps and make buildings unroutable.
    """
    logger.info(f"Fetching OSM road network (network_type: {network_type})")
    graph = ox.graph_from_polygon(polygon_4326, network_type=network_type, simplify=True)
    nodes_gdf, edges_gdf = ox.graph_to_gdfs(graph)

    edges_gdf = edges_gdf.reset_index().rename(
        columns={"u": "road_edge_start", "v": "road_edge_end", "length": "road_edge_length"}
    )
    keep = ("road_edge_start", "road_edge_end", "road_edge_length", "highway", "name", "geometry")
    edges_gdf = edges_gdf[[c for c in keep if c in edges_gdf.columns]]
    edges_gdf = _stringify_list_columns(edges_gdf)

    nodes_gdf = nodes_gdf.reset_index().rename(columns={"osmid": "road_node_id"})[["road_node_id", "geometry"]]

    logger.info(f"Fetched OSM road network: {len(edges_gdf)} edges, {len(nodes_gdf)} nodes")
    return edges_gdf.to_crs(crs), nodes_gdf.to_crs(crs)


def fetch_osm_parks(polygon_4326: Polygon | MultiPolygon, park_tags: dict | None, exclude_private: bool, crs: str) -> gpd.GeoDataFrame:
    """Fetch public park polygons from OSM with canonical column `park_id`.

    Features tagged access=private/no/customers are dropped when
    exclude_private is set.
    """
    tags = park_tags or DEFAULT_PARK_TAGS
    logger.info(f"Fetching OSM parks (tags: {tags})")
    try:
        gdf = ox.features_from_polygon(polygon_4326, tags).reset_index()
    except InsufficientResponseError:
        raise ValueError("OSM returned no parks for the study area — check the boundary or osm.park_tags")
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    if exclude_private and "access" in gdf.columns:
        gdf = gdf[~gdf["access"].isin(PRIVATE_ACCESS_VALUES)]
    if gdf.empty:
        raise ValueError("OSM returned no public park polygons for the study area — check the boundary or osm.park_tags")
    gdf["park_id"] = _osm_ids(gdf)
    keep = ["park_id", "name", *tags.keys(), "access", "geometry"]
    gdf = gdf[[c for c in dict.fromkeys(keep) if c in gdf.columns]]
    logger.info(f"Fetched {len(gdf)} OSM parks")
    return _stringify_list_columns(gdf.to_crs(crs).reset_index(drop=True))


def derive_park_access_points(
    parks_gdf: gpd.GeoDataFrame,
    road_edges_gdf: gpd.GeoDataFrame,
    polygon_4326: Polygon | MultiPolygon,
    crs: str,
) -> gpd.GeoDataFrame:
    """Derive park access points, preferring mapped OSM entrances.

    Per park: OSM entrance/gate nodes within ENTRANCE_SNAP_DISTANCE of its
    boundary; else intersection points of its boundary with road edges; else
    its representative point. Output columns: park_id (unique per access
    point), park_access_ref (the park's park_id), Point geometry.
    """
    logger.info(f"Deriving park access points for {len(parks_gdf)} parks")
    points: dict[str, list] = {pid: [] for pid in parks_gdf["park_id"]}
    boundaries = parks_gdf[["park_id", "geometry"]].copy()
    boundaries["geometry"] = boundaries.geometry.boundary

    entrances_gdf = _fetch_entrance_nodes(polygon_4326, crs)
    if entrances_gdf is not None and not entrances_gdf.empty:
        snapped = gpd.sjoin_nearest(
            entrances_gdf, boundaries, how="inner", max_distance=ENTRANCE_SNAP_DISTANCE
        )
        for row in snapped.itertuples():
            points[row.park_id].append(row.geometry)
        logger.info(f"Matched {len(snapped)} OSM entrance nodes to {snapped['park_id'].nunique()} parks")

    remaining = boundaries[boundaries["park_id"].map(lambda pid: not points[pid])]
    if not remaining.empty:
        edges = road_edges_gdf[["geometry"]].reset_index(drop=True)
        hits = gpd.sjoin(remaining, edges, predicate="intersects")
        for row in hits.itertuples():
            crossing = row.geometry.intersection(edges.geometry.iloc[row.index_right])
            points[row.park_id].extend(_extract_points(crossing))
        for pid in remaining["park_id"]:
            points[pid] = _thin_points(points[pid])

    records = []
    for park in parks_gdf.itertuples():
        park_points = points[park.park_id] or [park.geometry.representative_point()]
        for i, pt in enumerate(park_points):
            records.append({"park_id": f"{park.park_id}_acc{i}", "park_access_ref": park.park_id, "geometry": pt})

    access_gdf = gpd.GeoDataFrame(records, crs=crs)
    logger.info(f"Derived {len(access_gdf)} park access points")
    return access_gdf


def _fetch_entrance_nodes(polygon_4326: Polygon | MultiPolygon, crs: str) -> gpd.GeoDataFrame | None:
    """Fetch OSM entrance/gate nodes; None when the area has none mapped."""
    try:
        gdf = ox.features_from_polygon(polygon_4326, {"entrance": True, "barrier": ["gate", "entrance"]})
    except InsufficientResponseError:
        logger.info("No OSM entrance/gate nodes in the study area — using boundary/road intersections")
        return None
    gdf = gdf.reset_index()
    gdf = gdf[gdf.geometry.geom_type == "Point"]
    return gdf[["geometry"]].to_crs(crs)


def _extract_points(geom) -> list:
    """Reduce a boundary/edge intersection to representative Points.

    Collinear overlaps come back as LineStrings — use their midpoints.
    """
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [geom]
    if geom.geom_type == "LineString":
        return [geom.interpolate(0.5, normalized=True)]
    if geom.geom_type in ("MultiPoint", "MultiLineString", "GeometryCollection"):
        return [pt for part in geom.geoms for pt in _extract_points(part)]
    return []


def _thin_points(pts: list) -> list:
    """Drop near-duplicate points (within ACCESS_POINT_SPACING metres)."""
    seen = set()
    thinned = []
    for pt in pts:
        key = (round(pt.x / ACCESS_POINT_SPACING), round(pt.y / ACCESS_POINT_SPACING))
        if key not in seen:
            seen.add(key)
            thinned.append(pt)
    return thinned
