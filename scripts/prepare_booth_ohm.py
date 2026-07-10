#!/usr/bin/env python3
"""Prepare Booth buildings + OHM parks/roads (London 1880-1900) as greenpy inputs.

Writes parquet inputs for examples/booth_london.yaml to
/maps/acz25/phd-thesis-data/output/greenpy/Booth/input/:

- booth_buildings.parquet  Booth gpkg with a generated building_id
- ohm_road_edges.parquet   roads noded (split at intersections) with road_edge_length
- ohm_parks_sites.parquet  parks with park_id (from osm_id)
- ohm_parks_access.parquet access points = park boundary x road intersections
                           (no OSM entrance fetch: modern entrances would be
                           anachronistic for 1880-1900)
- booth_boundary.parquet   single study-area polygon (city = LONDON)
"""

from pathlib import Path

import geopandas as gpd
import shapely
from loguru import logger

from greenpy.osm import _extract_points, _thin_points

CRS = "EPSG:27700"
BUILDINGS_GPKG = "/maps/acz25/phd-thesis-data/input/Booth_maps/Booth_buildings.gpkg"
PARKS_PARQUET = "/maps/acz25/phd-thesis-data/input/OHM/ohm_parks_london_1880_1900.parquet"
ROADS_PARQUET = "/maps/acz25/phd-thesis-data/input/OHM/ohm_roads_london_1880_1900.parquet"
OUT_DIR = Path("/maps/acz25/phd-thesis-data/output/greenpy/Booth/input")


def prepare_buildings() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(BUILDINGS_GPKG, layer="Booth").to_crs(CRS)
    gdf["building_id"] = [f"B{i:05d}" for i in range(len(gdf))]
    gdf.to_parquet(OUT_DIR / "booth_buildings.parquet", index=False)
    logger.info(f"buildings: {len(gdf)} features")
    return gdf


def prepare_roads() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(ROADS_PARQUET).to_crs(CRS)
    # Split ways at every intersection so junctions become segment endpoints;
    # greenpy derives the road graph nodes from segment endpoints only.
    collection = shapely.GeometryCollection(list(gdf.geometry.values))
    try:
        noded = shapely.node(collection)
    except shapely.errors.GEOSException:
        # snap coordinates to a 1 cm grid to clear side-location conflicts
        noded = shapely.node(shapely.set_precision(collection, 0.01))
    segments = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([noded], crs=CRS).explode(ignore_index=True)
    )
    segments = segments[segments.geometry.geom_type == "LineString"].reset_index(drop=True)

    # Keep only the largest connected component (endpoints matched the same way
    # greenpy derives nodes: rounded to 0.1 m) so every building snaps to a
    # routable network instead of an isolated fragment.
    import networkx as nx

    starts = [(round(g.coords[0][0], 1), round(g.coords[0][1], 1)) for g in segments.geometry]
    ends = [(round(g.coords[-1][0], 1), round(g.coords[-1][1], 1)) for g in segments.geometry]
    graph = nx.Graph(zip(starts, ends))
    giant = max(nx.connected_components(graph), key=len)
    keep = [s in giant and e in giant for s, e in zip(starts, ends)]
    dropped = len(segments) - sum(keep)
    segments = segments[keep].reset_index(drop=True)

    segments["road_edge_length"] = segments.geometry.length
    segments.to_parquet(OUT_DIR / "ohm_road_edges.parquet", index=False)
    logger.info(f"roads: {len(gdf)} ways -> {len(segments)} noded segments ({dropped} in disconnected fragments dropped)")
    return segments


def prepare_parks() -> gpd.GeoDataFrame:
    gdf = gpd.read_parquet(PARKS_PARQUET).to_crs(CRS)
    gdf = gdf.rename(columns={"osm_id": "park_id"})
    gdf["park_id"] = gdf["park_id"].astype(str)
    gdf = gdf.drop(columns=["all_tags"], errors="ignore")
    gdf.to_parquet(OUT_DIR / "ohm_parks_sites.parquet", index=False)
    logger.info(f"parks: {len(gdf)} features")
    return gdf


def prepare_park_access(parks_gdf: gpd.GeoDataFrame, edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Same derivation as osm.derive_park_access_points minus the entrance fetch,
    # but with the park boundary buffered 20 m: OHM roads are centerlines that
    # rarely touch the park polygon, so we take the pieces of road running
    # along/through the edge ribbon (their midpoints land on the road itself,
    # which snaps cleanly to the network). Parks with no road within 20 m fall
    # back to their representative point.
    points: dict[str, list] = {pid: [] for pid in parks_gdf["park_id"]}
    boundaries = parks_gdf[["park_id", "geometry"]].copy()
    boundaries["geometry"] = boundaries.geometry.boundary.buffer(20)

    edges = edges_gdf[["geometry"]].reset_index(drop=True)
    hits = gpd.sjoin(boundaries, edges, predicate="intersects")
    for row in hits.itertuples():
        crossing = row.geometry.intersection(edges.geometry.iloc[row.index_right])
        points[row.park_id].extend(_extract_points(crossing))
    for pid in points:
        points[pid] = _thin_points(points[pid])

    records = []
    for park in parks_gdf.itertuples():
        park_points = points[park.park_id] or [park.geometry.representative_point()]
        for i, pt in enumerate(park_points):
            records.append({"park_id": f"{park.park_id}_acc{i}", "park_access_ref": park.park_id, "geometry": pt})

    access_gdf = gpd.GeoDataFrame(records, crs=CRS)
    access_gdf.to_parquet(OUT_DIR / "ohm_parks_access.parquet", index=False)
    no_road = sum(1 for pid in points if not points[pid])
    logger.info(f"park access: {len(access_gdf)} points ({no_road} parks fell back to representative point)")
    return access_gdf


def prepare_boundary(buildings_gdf: gpd.GeoDataFrame) -> None:
    hull = shapely.convex_hull(shapely.GeometryCollection(list(buildings_gdf.geometry.values))).buffer(100)
    boundary = gpd.GeoDataFrame({"city": ["LONDON"]}, geometry=[hull], crs=CRS)
    boundary.to_parquet(OUT_DIR / "booth_boundary.parquet", index=False)
    logger.info(f"boundary: {hull.area / 1e6:.1f} km2 convex hull of buildings")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = prepare_buildings()
    roads = prepare_roads()
    parks = prepare_parks()
    prepare_park_access(parks, roads)
    prepare_boundary(buildings)
    logger.info(f"All inputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
