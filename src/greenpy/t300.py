from pathlib import Path

import time
import logging
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from tqdm import tqdm
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import filter_buffer_geometries, get_geometries


def filter_features(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    road_nodes_gdf: gpd.GeoDataFrame,
    road_edges_gdf: gpd.GeoDataFrame,
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    search_buffer: int = 2000,
) -> tuple:
    """Spatially filter roads, parks, and buildings to the geo_code boundary.

    Buildings are filtered to the exact OA boundary. Roads and parks use a
    buffered boundary so the network can route to parks outside the OA.
    """
    logging.debug("Filtering GeoDataFrames by spatial join")

    buffered = geo_boundary_gdf.copy()
    buffered["geometry"] = geo_boundary_gdf.geometry.buffer(search_buffer)

    _edges = road_edges_gdf.drop(columns=[c for c in road_edges_gdf.columns if c in ("index_right", "index_left")], errors="ignore")
    _buffered = buffered.drop(columns=[c for c in buffered.columns if c in ("index_right", "index_left")], errors="ignore")
    geo_road_edges_gdf = gpd.sjoin(_edges, _buffered).rename(columns={"road_edge_start": "u", "road_edge_end": "v"})
    geo_road_edges_gdf["key"] = geo_road_edges_gdf.groupby(["u", "v"]).cumcount()
    geo_road_edges_gdf = geo_road_edges_gdf.set_index(["u", "v", "key"])

    node_id_col = "road_node_id"
    geo_road_nodes_gdf = road_nodes_gdf[
        road_nodes_gdf[node_id_col].isin(geo_road_edges_gdf.index.get_level_values(0))
        | road_nodes_gdf[node_id_col].isin(geo_road_edges_gdf.index.get_level_values(1))
    ].set_index(node_id_col)
    geo_road_nodes_gdf["x"] = geo_road_nodes_gdf.geometry.x
    geo_road_nodes_gdf["y"] = geo_road_nodes_gdf.geometry.y

    # Buildings: exact OA boundary only
    geo_buildings_sdf = filter_buffer_geometries(sedona, geo_level, geo_code, "buildings", id_col="building_id")
    geo_buildings_gdf = gpd.GeoDataFrame(geo_buildings_sdf.toPandas(), geometry="geometry", crs=cfg.crs)

    # Parks and access points: use buffered boundary so nearby parks are reachable
    geo_park_sites_gdf = gpd.sjoin(
        gpd.read_parquet(Path(cfg.output.base_dir) / "database" / "parks_sites.parquet"),
        buffered.drop(columns=[c for c in buffered.columns if c not in ("geometry",)]),
        how="inner",
    ).drop(columns=["index_right"], errors="ignore")

    geo_park_access_gdf = gpd.sjoin(
        gpd.read_parquet(Path(cfg.output.base_dir) / "database" / "parks_access.parquet"),
        buffered.drop(columns=[c for c in buffered.columns if c not in ("geometry",)]),
        how="inner",
    ).drop(columns=["index_right"], errors="ignore")

    return geo_road_nodes_gdf, geo_road_edges_gdf, geo_park_sites_gdf, geo_park_access_gdf, geo_buildings_gdf


def get_road_graph_distances(
    geo_road_nodes_gdf: gpd.GeoDataFrame,
    geo_road_edges_gdf: gpd.GeoDataFrame,
    geo_park_access_gdf: gpd.GeoDataFrame,
    geo_buildings_gdf: gpd.GeoDataFrame,
) -> tuple:
    """Build road network graph and snap buildings + park accesses to nearest nodes."""
    logging.debug("Generating road graph")

    geo_graph = ox.graph_from_gdfs(geo_road_nodes_gdf, geo_road_edges_gdf).to_undirected()

    park_node_ids, park_node_dists = ox.distance.nearest_nodes(
        geo_graph,
        geo_park_access_gdf.geometry.centroid.x,
        geo_park_access_gdf.geometry.centroid.y,
        return_dist=True,
    )
    geo_park_access_gdf = geo_park_access_gdf.copy()
    geo_park_access_gdf["nearest_road_node"] = park_node_ids
    geo_park_access_gdf["nearest_road_node_distance"] = park_node_dists

    building_node_ids, building_node_dists = ox.distance.nearest_nodes(
        geo_graph,
        geo_buildings_gdf.geometry.centroid.x,
        geo_buildings_gdf.geometry.centroid.y,
        return_dist=True,
    )
    geo_buildings_gdf = geo_buildings_gdf.copy()
    geo_buildings_gdf["nearest_road_node"] = building_node_ids
    geo_buildings_gdf["nearest_road_node_distance"] = building_node_dists

    return geo_graph, geo_park_access_gdf, geo_buildings_gdf


def get_closest_park_manhattan(
    geo_graph: nx.MultiGraph,
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_access_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Shortest-path (Dijkstra) distance from each building to nearest park access point."""
    logging.debug(f"Computing park distances for {len(geo_buildings_gdf)} buildings, {len(geo_park_access_gdf)} access points")

    park_access_nodes = geo_park_access_gdf["nearest_road_node"].unique()
    shortest_paths = {}
    for node in tqdm(park_access_nodes, desc="Park access nodes"):
        shortest_paths[node] = nx.single_source_dijkstra_path_length(geo_graph, node, weight="road_edge_length")

    distances = []
    for building in tqdm(geo_buildings_gdf.itertuples(), desc="Buildings processed"):
        building_node = building.nearest_road_node
        building_id = building.building_id
        building_road_dist = building.nearest_road_node_distance
        min_distance = float("inf")
        closest_park_access_id = None

        for park_access in geo_park_access_gdf.itertuples():
            park_node = park_access.nearest_road_node
            park_road_dist = park_access.nearest_road_node_distance
            try:
                d = shortest_paths[park_node][building_node] + building_road_dist + park_road_dist
                if d < min_distance:
                    min_distance = d
                    closest_park_access_id = park_access.park_id
            except Exception:
                pass

        distances.append((building_id, closest_park_access_id, None if min_distance == float("inf") else round(min_distance, 1)))

    return pd.DataFrame(distances, columns=["building_id", "closest_park_access_id", "distance_manhattan"])


def get_closest_park_euclidean(
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_sites_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Euclidean (straight-line) distance from each building to nearest park site polygon."""
    result = gpd.sjoin_nearest(geo_buildings_gdf, geo_park_sites_gdf, distance_col="distance_euclidean")
    result["distance_euclidean"] = result["distance_euclidean"].round(1)
    return result[["building_id", "park_id", "distance_euclidean"]].rename(columns={"park_id": "closest_park_site_id"})


def get_closest_park(
    sedona: SparkSession,
    geo_graph: nx.MultiGraph,
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_access_gdf: gpd.GeoDataFrame,
    geo_park_sites_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    manhattan_df = get_closest_park_manhattan(geo_graph, geo_buildings_gdf, geo_park_access_gdf)
    euclidean_df = get_closest_park_euclidean(geo_buildings_gdf, geo_park_sites_gdf)
    return pd.merge(manhattan_df, euclidean_df, on="building_id")


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    road_nodes_gdf: gpd.GeoDataFrame,
    road_edges_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    output_dir: Path,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    start_time = time.time()
    logging.info(f"T300: processing {geo_code}")

    out_path = output_dir / f"T300_{geo_code}.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, dissolve=True)
        geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry="geometry", crs=cfg.crs)

        geo_road_nodes_gdf, geo_road_edges_gdf, geo_park_sites_gdf, geo_park_access_gdf, geo_buildings_gdf = filter_features(
            sedona, geo_level, geo_code, road_nodes_gdf, road_edges_gdf, geo_boundary_gdf, cfg
        )
        geo_graph, geo_park_access_gdf, geo_buildings_gdf = get_road_graph_distances(
            geo_road_nodes_gdf, geo_road_edges_gdf, geo_park_access_gdf, geo_buildings_gdf
        )
        geo_park_distance_df = get_closest_park(
            sedona, geo_graph, geo_buildings_gdf, geo_park_access_gdf, geo_park_sites_gdf
        )
        geo_park_distance_df.to_csv(out_path, index=False)

        end_time = time.time()
        logging.info(f"T300: {geo_code} — {len(geo_park_distance_df)} records in {end_time - start_time:.2f}s")
        return geo_park_distance_df

    except Exception as e:
        logging.error(f"T300: error processing {geo_code}: {e}")
