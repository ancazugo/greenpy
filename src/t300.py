
from utils.paths import T300_dir
from utils.constants import PROJECT_CRS
from utils.data_processing import filter_buffer_geometries, get_geometries

import time
import logging
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from tqdm import tqdm
from pyspark.sql.session import SparkSession

def filter_features(sedona: SparkSession, geo_level: str, geo_code: str, road_nodes_gdf: gpd.GeoDataFrame, 
                    road_edges_gdf: gpd.GeoDataFrame, geo_boundary_gdf: gpd.GeoDataFrame) -> dict:
    """
    Filters various GeoDataFrames by performing spatial joins with a given geographic boundary.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo level.
        geo_code (str): The geo code.
        road_nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing road nodes.
        road_edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing road edges.
        geo_boundary_gdf (gpd.GeoDataFrame): GeoDataFrame containing the geographic boundary for filtering.

    Returns:
        dict: A dictionary containing filtered GeoDataFrames with the following keys:
            - 'nodes': Filtered road nodes GeoDataFrame.
            - 'edges': Filtered road edges GeoDataFrame.
            - 'public_park_sites': Filtered public park sites GeoDataFrame.
            - 'public_park_access': Filtered public park access points GeoDataFrame.
            - 'buildings': Filtered buildings GeoDataFrame.
    """
    
    logging.debug("Filtering GeoDataFrames by spatial join")

    geo_road_edges_gdf = gpd.sjoin(road_edges_gdf, geo_boundary_gdf)\
        .rename(columns={'start_node': 'u', 'end_node': 'v', 'id': 'key'})\
        .set_index(['u', 'v', 'key'])
    geo_road_nodes_gdf = road_nodes_gdf[(road_nodes_gdf['id'].isin(geo_road_edges_gdf.index.get_level_values(0))) | (road_nodes_gdf['id'].isin(geo_road_edges_gdf.index.get_level_values(1)))].set_index('id')

    geo_road_nodes_gdf['x'] = geo_road_nodes_gdf.geometry.x
    geo_road_nodes_gdf['y'] = geo_road_nodes_gdf.geometry.y

    geo_buildings_sdf = filter_buffer_geometries(sedona, geo_level, geo_code, 'buildings')
    geo_buildings_gdf = gpd.GeoDataFrame(geo_buildings_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
    geo_public_park_sites_sdf = filter_buffer_geometries(sedona, geo_level, geo_code, 'public_park_sites')
    geo_public_park_sites_gdf = gpd.GeoDataFrame(geo_public_park_sites_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
    geo_public_park_accesses_sdf = filter_buffer_geometries(sedona, geo_level, geo_code, 'public_park_accesses')
    geo_public_park_accesses_gdf = gpd.GeoDataFrame(geo_public_park_accesses_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)

    return geo_road_nodes_gdf, geo_road_edges_gdf, geo_public_park_sites_gdf, geo_public_park_accesses_gdf, geo_buildings_gdf

def get_road_graph_distances(geo_road_nodes_gdf: gpd.GeoDataFrame, geo_road_edges_gdf: gpd.GeoDataFrame, 
                             geo_public_park_accesses_gdf: gpd.GeoDataFrame, geo_buildings_gdf: gpd.GeoDataFrame) -> tuple:
    """
    Generates a graph of the road network and calculates the distances to the nearest park access point and building.

    Args:
        geo_road_nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing road nodes.
        geo_road_edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing road edges.
        geo_public_park_accesses_gdf (gpd.GeoDataFrame): GeoDataFrame containing public park access points.
        geo_buildings_gdf (gpd.GeoDataFrame): GeoDataFrame containing building information.

    Returns:
        tuple: A tuple containing the graph of the road network, the public park access points GeoDataFrame, and the buildings GeoDataFrame.
    """
    
    logging.debug("Generating graph of the road network")

    geo_graph = ox.graph_from_gdfs(geo_road_nodes_gdf, geo_road_edges_gdf).to_undirected()

    park_near_road_node_id, park_near_road_node_dist = ox.distance.nearest_nodes(geo_graph, geo_public_park_accesses_gdf.geometry.centroid.x, geo_public_park_accesses_gdf.geometry.centroid.y, return_dist=True)
    geo_public_park_accesses_gdf['nearest_road_node'] = park_near_road_node_id
    geo_public_park_accesses_gdf['nearest_road_node_distance'] = park_near_road_node_dist
    building_near_road_node_id, building_near_road_node_dist = ox.distance.nearest_nodes(geo_graph, geo_buildings_gdf.geometry.centroid.x, geo_buildings_gdf.geometry.centroid.y, return_dist=True)
    geo_buildings_gdf['nearest_road_node'] = building_near_road_node_id
    geo_buildings_gdf['nearest_road_node_distance'] = building_near_road_node_dist

    return geo_graph, geo_public_park_accesses_gdf, geo_buildings_gdf

def get_closest_park_manhattan(geo_graph: nx.MultiGraph, geo_buildings_gdf: gpd.GeoDataFrame, geo_public_park_accesses_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculate the closest park access point for each building in the given GeoDataFrame.

    Parameters:
        geo_graph (nx.MultiGraph): A graph representing the road network.
        geo_buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame containing building information, including the nearest road node.
        geo_public_park_accesses_gdf (gpd.GeoDataFrame): A GeoDataFrame containing public park access points, including the nearest road node.

    Returns:
        pd.DataFrame: A DataFrame with columns 'verisk_premise_id', 'closest_park_access_id', and 'distance', representing the building ID, the closest park access point ID, and the distance to the closest park access point, respectively.
    """

    logging.debug(f"Getting closest park (n: {len(geo_public_park_accesses_gdf)}) to each building (n: {len(geo_buildings_gdf)})")

    park_access_nodes = geo_public_park_accesses_gdf['nearest_road_node'].unique()
    shortest_paths = {}
    for park_access_node in tqdm(park_access_nodes):
        lengths = nx.single_source_dijkstra_path_length(geo_graph, park_access_node, weight='length')
        shortest_paths[park_access_node] = lengths

    # Create a list to store the distances
    distances = []

    # Iterate over each building
    for building in tqdm(geo_buildings_gdf.itertuples(), desc='Buildings processed'):
        building_node = building.nearest_road_node
        building_id = building.verisk_premise_id
        building_road_node_dist = building.nearest_road_node_distance
        min_distance = float('inf')
        closest_park_access_id = None

        # Iterate over each park access point
        for park_access in tqdm(geo_public_park_accesses_gdf.itertuples(), desc='Parks checked', leave=False):
            park_access_node = park_access.nearest_road_node
            park_road_node_dist = park_access.nearest_road_node_distance

            # Lookup the precomputed shortest path distance
            try:
                distance = shortest_paths[park_access_node][building_node] + building_road_node_dist + park_road_node_dist
                if distance < min_distance:
                    min_distance = distance
                    closest_park_access_id = park_access.id
            except Exception as e:
                logging.debug(f"Error with building/park pair: {building_node}/{park_access_node} - {e}")

        min_distance = None if min_distance == float('inf') else min_distance

        distances.append((building_id, closest_park_access_id, min_distance))

    # Convert the distances to a DataFrame
    manhattan_distances_df = pd.DataFrame(distances, columns=['verisk_premise_id', 'closest_park_access_id', 'distance_manhattan'])
    manhattan_distances_df['distance_manhattan'] = manhattan_distances_df.distance_manhattan.apply(lambda x: round(x, 1))

    return manhattan_distances_df

def get_closest_park_euclidean(sedona: SparkSession, geo_buildings_gdf: gpd.GeoDataFrame, geo_public_park_site_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates the Euclidean distance to the closest park site for each building in the given GeoDataFrame.

    Args:
        sedona (SparkSession): The Spark session.
        geo_buildings_gdf (gpd.GeoDataFrame): GeoDataFrame containing building information.
        geo_public_park_site_gdf (gpd.GeoDataFrame): GeoDataFrame containing public park sites.

    Returns:
        pd.DataFrame: A DataFrame with columns 'verisk_premise_id', 'closest_park_site_id', and 'distance_euclidean', representing the building ID, the closest park site ID, and the Euclidean distance to the closest park site, respectively.
    """
    
    euclidean_distances_df = gpd.sjoin_nearest(geo_buildings_gdf, geo_public_park_site_gdf, distance_col='distance_euclidean')
    euclidean_distances_df['distance_euclidean'] = euclidean_distances_df.distance_euclidean.apply(lambda x: round(x, 1))
    euclidean_distances_df = euclidean_distances_df[['verisk_premise_id', 'id', 'distance_euclidean']].rename(columns={'id': 'closest_park_site_id'})

    # TODO: Figure out why the query doesn't work
    # euclidean_distances_sdf = sedona.sql(
    #     """
    #     SELECT b.verisk_premise_id, p.id
    #     FROM geo_buildings b
    #     JOIN geo_public_park_sites p
    #     ON ST_KNN(b.geometry, p.geometry, 1, TRUE) 
    #     """)
    # euclidean_distances_df = euclidean_distances_sdf.toPandas()
    
    return euclidean_distances_df

def get_closest_park(sedona: SparkSession, geo_graph: nx.MultiGraph, geo_buildings_gdf: gpd.GeoDataFrame, 
                     geo_public_park_accesses_gdf: gpd.GeoDataFrame, geo_public_park_sites_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """
    Calculates the closest park access point and Euclidean distance to the closest park site for each building in the given GeoDataFrame.

    Args:
        sedona (SparkSession): The Spark session.
        geo_graph (nx.MultiGraph): A graph representing the road network.
        geo_buildings_gdf (gpd.GeoDataFrame): A GeoDataFrame containing building information, including the nearest road node.
        geo_public_park_accesses_gdf (gpd.GeoDataFrame): A GeoDataFrame containing public park access points, including the nearest road node.
        geo_public_park_sites_gdf (gpd.GeoDataFrame): A GeoDataFrame containing public park sites.

    Returns:
        pd.DataFrame: A DataFrame with columns 'verisk_premise_id', 'closest_park_access_id', 'distance_manhattan', 'closest_park_site_id', and 'distance_euclidean', representing the building ID, the closest park access point ID, the Manhattan distance to the closest park access point, the closest park site ID, and the Euclidean distance to the closest park site, respectively.
    """
    
    manhattan_distances_df = get_closest_park_manhattan(geo_graph, geo_buildings_gdf, geo_public_park_accesses_gdf)
    euclidean_distances_df = get_closest_park_euclidean(sedona, geo_buildings_gdf, geo_public_park_sites_gdf)

    geo_park_distance_df = pd.merge(manhattan_distances_df, euclidean_distances_df, on='verisk_premise_id')

    return geo_park_distance_df

def process_geo_code(sedona: SparkSession, geo_level: str, geo_code: str, road_nodes_gdf: gpd.GeoDataFrame, 
                     road_edges_gdf: gpd.GeoDataFrame, overwrite: bool=True) -> pd.DataFrame:
    """
    Processes a given geo_code for T300.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo level.
        geo_code (str): The geo code.
        road_nodes_gdf (gpd.GeoDataFrame): GeoDataFrame containing road nodes.
        road_edges_gdf (gpd.GeoDataFrame): GeoDataFrame containing road edges.
        overwrite (bool): Whether to overwrite the existing file.
        
    Returns:
        pd.DataFrame: A DataFrame with columns 'verisk_premise_id', 'closest_park_access_id', 'distance_manhattan', 'closest_park_site_id', and 'distance_euclidean', representing the building ID, the closest park access point ID, the Manhattan distance to the closest park access point, the closest park site ID, and the Euclidean distance to the closest park site, respectively.
    """
    
    start_time = time.time()
    logging.info(f"Processing data for {geo_code}")

    geo_park_distance_path = T300_dir / f"T300_{geo_code}.csv"

    if not geo_park_distance_path.exists() or overwrite:
        try:

            geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, True)
            geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
            geo_road_nodes_gdf, geo_road_edges_gdf, geo_public_park_sites_gdf, geo_public_park_accesses_gdf, geo_buildings_gdf = filter_features(sedona, geo_level, geo_code, 
                                                                                                                                                road_nodes_gdf, road_edges_gdf, 
                                                                                                                                                geo_boundary_gdf)

            geo_graph, geo_public_park_accesses_gdf, geo_buildings_gdf = get_road_graph_distances(geo_road_nodes_gdf, geo_road_edges_gdf, geo_public_park_accesses_gdf, geo_buildings_gdf)
            geo_park_distance_df = get_closest_park(sedona, geo_graph, geo_buildings_gdf, geo_public_park_accesses_gdf, geo_public_park_sites_gdf)

            geo_park_distance_df.to_csv(geo_park_distance_path, index=False)

            end_time = time.time()
            logging.info(f"Processing for {geo_code} with {len(geo_park_distance_df)} records took {end_time - start_time:.2f} seconds")

            return geo_park_distance_df

        except Exception as e:
            logging.error(f"Error processing {geo_code}: {e}")
        