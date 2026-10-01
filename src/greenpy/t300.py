from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from loguru import logger
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .pipeline import _filter_parks, _filter_park_access
from .utils.data_processing import filter_buffer_geometries, get_geometries, drop_geo_views


def filter_features(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    road_nodes_gdf: gpd.GeoDataFrame,
    road_edges_gdf: gpd.GeoDataFrame,
    geo_boundary_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    search_buffer: int | None = None,
) -> tuple:
    """Spatially filter roads, parks, and buildings to the geo_code.

    Buildings are those owned by geo_code in the buildings overlay. Roads and
    parks use the boundary buffered by search_buffer metres (default
    cfg.osm.fetch_buffer, 2000 m) so the network can route to parks outside it.
    """
    if search_buffer is None:
        search_buffer = cfg.osm.fetch_buffer
    logger.debug("Filtering GeoDataFrames by spatial join")

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

    # Parks and access points: use buffered boundary so nearby parks are reachable.
    # Apply the same function/linkage filters that load_tables() applies.
    db_dir = Path(cfg.output.base_dir) / "database"
    park_sites_all = _filter_parks(gpd.read_parquet(db_dir / "parks_sites.parquet"), cfg)
    park_access_all = _filter_park_access(gpd.read_parquet(db_dir / "parks_access.parquet"), park_sites_all, cfg)

    geo_park_sites_gdf = gpd.sjoin(
        park_sites_all,
        buffered.drop(columns=[c for c in buffered.columns if c not in ("geometry",)]),
        how="inner",
    ).drop(columns=["index_right"], errors="ignore")

    geo_park_access_gdf = gpd.sjoin(
        park_access_all,
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
    logger.debug("Generating road graph")

    if "road_edge_length" not in geo_road_edges_gdf.columns or geo_road_edges_gdf["road_edge_length"].isna().any():
        # networkx would silently weight missing lengths as 1 (hop counts)
        raise ValueError(
            "Road edges lack road_edge_length — set columns.road_edge_length, or delete the "
            "database/ cache so it is recomputed from edge geometry"
        )
    geo_graph = ox.graph_from_gdfs(geo_road_nodes_gdf, geo_road_edges_gdf).to_undirected()

    def _snap(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        gdf = gdf.copy()
        if gdf.empty:
            gdf["nearest_road_node"] = pd.Series(dtype=object)
            gdf["nearest_road_node_distance"] = pd.Series(dtype=float)
            return gdf
        centroids = gdf.geometry.centroid
        ids, dists = ox.distance.nearest_nodes(geo_graph, centroids.x, centroids.y, return_dist=True)
        gdf["nearest_road_node"] = ids
        gdf["nearest_road_node_distance"] = dists
        return gdf

    return geo_graph, _snap(geo_park_access_gdf), _snap(geo_buildings_gdf)


def get_closest_park_manhattan(
    geo_graph: nx.MultiGraph,
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_access_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Network shortest-path distance from each building to its nearest park access point.

    Despite the historical name and output column (`distance_manhattan`, kept
    for compatibility), this is road-network distance weighted by
    road_edge_length, plus the straight-line snap distances of building and
    access point to their nearest road nodes. Computed with a single
    multi-source Dijkstra from a virtual source linked to every access node
    (weighted by its snap distance), which equals the minimum over all access
    points. Unreachable buildings get None.
    """
    logger.debug(f"Computing park distances for {len(geo_buildings_gdf)} buildings, {len(geo_park_access_gdf)} access points")

    # per road node: the closest access point snapped to it
    best_access = (
        geo_park_access_gdf.sort_values("nearest_road_node_distance")
        .drop_duplicates("nearest_road_node")
        .set_index("nearest_road_node")
    )
    source = object()  # virtual node, cannot collide with road node ids
    graph = nx.Graph()
    for u, v, w in geo_graph.edges(data="road_edge_length"):
        if not graph.has_edge(u, v) or w < graph[u][v]["w"]:
            graph.add_edge(u, v, w=w)
    for node, row in best_access.iterrows():
        graph.add_edge(source, node, w=row["nearest_road_node_distance"])

    dist, paths = (
        nx.single_source_dijkstra(graph, source, weight="w") if len(best_access) else ({}, {})
    )

    distances = []
    for building in geo_buildings_gdf.itertuples():
        node = building.nearest_road_node
        if node in dist:
            access_node = paths[node][1]
            d = round(dist[node] + building.nearest_road_node_distance, 1)
            distances.append((building.building_id, best_access.at[access_node, "park_id"], d))
        else:
            distances.append((building.building_id, None, None))

    return pd.DataFrame(distances, columns=["building_id", "closest_park_access_id", "distance_manhattan"])


def get_closest_park_euclidean(
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_sites_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Euclidean (straight-line) distance from each building to nearest park site polygon."""
    if geo_park_sites_gdf.empty:
        return pd.DataFrame({
            "building_id": geo_buildings_gdf["building_id"],
            "closest_park_site_id": None,
            "distance_euclidean": float("nan"),
        })
    result = gpd.sjoin_nearest(geo_buildings_gdf, geo_park_sites_gdf, distance_col="distance_euclidean")
    result["distance_euclidean"] = result["distance_euclidean"].round(1)
    # sjoin_nearest returns one row per tied nearest park — keep one per building
    result = result.drop_duplicates(subset="building_id")
    return result[["building_id", "park_id", "distance_euclidean"]].rename(columns={"park_id": "closest_park_site_id"})


def get_closest_park(
    sedona: SparkSession,
    geo_graph: nx.MultiGraph,
    geo_buildings_gdf: gpd.GeoDataFrame,
    geo_park_access_gdf: gpd.GeoDataFrame,
    geo_park_sites_gdf: gpd.GeoDataFrame,
) -> pd.DataFrame:
    """Combine network and Euclidean nearest-park distances into one row per building.

    Every building is kept: with no reachable park (or none within the search
    buffer) its distances are null.
    """
    manhattan_df = get_closest_park_manhattan(geo_graph, geo_buildings_gdf, geo_park_access_gdf)
    euclidean_df = get_closest_park_euclidean(geo_buildings_gdf, geo_park_sites_gdf)
    return pd.merge(manhattan_df, euclidean_df, on="building_id", how="left")


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    sub_geo_level: str,
    road_nodes_gdf: gpd.GeoDataFrame,
    road_edges_gdf: gpd.GeoDataFrame,
    cfg: GreenPyConfig,
    output_dir: Path,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    """Compute T300 (distance from each building to its nearest park) for one geo_code.

    Writes `T300_<geo_code>.csv` with columns building_id,
    closest_park_access_id, distance_manhattan (network distance),
    closest_park_site_id, distance_euclidean, <sub_geo_level>. Returns the
    DataFrame, the cached CSV when overwrite is False, or None on error.
    """
    start_time = time.time()
    logger.info(f"T300: processing {geo_code}")

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
        # Overlay lookup so the assignment always matches the one Merge aggregates by
        buildings_with_level = sedona.sql(
            f"""
            SELECT building_id, {sub_geo_level}
            FROM boundaries_buildings_overlay
            WHERE {geo_level} = '{geo_code}'
            """
        ).toPandas()
        # ids may be numeric in file-backed sources while views return strings
        buildings_with_level["building_id"] = buildings_with_level["building_id"].astype(str)
        geo_park_distance_df = geo_park_distance_df.assign(building_id=geo_park_distance_df["building_id"].astype(str))
        geo_park_distance_df = geo_park_distance_df.merge(buildings_with_level, on="building_id", how="left")

        geo_park_distance_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"T300: {geo_code} — {len(geo_park_distance_df)} records in {end_time - start_time:.2f}s")
        return geo_park_distance_df

    except Exception:
        logger.exception(f"T300: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
