"""Park distances: multi-source network Dijkstra vs brute force, and empty inputs."""

import random

import geopandas as gpd
import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString, Point, box

from greenpy.pipeline import _derive_road_nodes
from greenpy.t300 import (
    get_closest_park,
    get_closest_park_manhattan,
    get_road_graph_distances,
)

CRS = "EPSG:32632"


def _grid_network(n=6, step=100.0, seed=0):
    """Noded n x n street grid with jittered lengths, as canonical edges/nodes."""
    rng = random.Random(seed)
    lines = []
    for i in range(n):
        for j in range(n - 1):
            lines.append(LineString([(j * step, i * step), ((j + 1) * step, i * step)]))
            lines.append(LineString([(i * step, j * step), (i * step, (j + 1) * step)]))
    edges = gpd.GeoDataFrame(geometry=lines, crs=CRS)
    # lengths differ from geometry so the test would catch a hop-count / geometry fallback
    edges["road_edge_length"] = [line.length * rng.uniform(0.8, 1.6) for line in lines]
    edges, nodes = _derive_road_nodes(edges)
    edges = edges.rename(columns={"road_edge_start": "u", "road_edge_end": "v"})
    edges["key"] = edges.groupby(["u", "v"]).cumcount()
    edges = edges.set_index(["u", "v", "key"])
    nodes = nodes.set_index("road_node_id")
    nodes["x"], nodes["y"] = nodes.geometry.x, nodes.geometry.y
    return nodes, edges


def _buildings(points):
    return gpd.GeoDataFrame(
        {"building_id": [f"B{i}" for i in range(len(points))]},
        geometry=[Point(x, y).buffer(4) for x, y in points], crs=CRS,
    )


def _access(points):
    return gpd.GeoDataFrame(
        {"park_id": [f"A{i}" for i in range(len(points))]},
        geometry=[Point(x, y) for x, y in points], crs=CRS,
    )


def _brute_force(graph, buildings, access):
    rows = []
    for b in buildings.itertuples():
        best, best_id = float("inf"), None
        for a in access.itertuples():
            try:
                d = nx.shortest_path_length(graph, a.nearest_road_node, b.nearest_road_node, weight="road_edge_length")
            except nx.NetworkXNoPath:
                continue
            d += a.nearest_road_node_distance + b.nearest_road_node_distance
            if d < best:
                best, best_id = d, a.park_id
        rows.append((b.building_id, best_id, None if best == float("inf") else round(best, 1)))
    return pd.DataFrame(rows, columns=["building_id", "closest_park_access_id", "distance_manhattan"])


def test_multi_source_matches_brute_force():
    rng = random.Random(1)
    nodes, edges = _grid_network()
    buildings = _buildings([(rng.uniform(0, 500), rng.uniform(0, 500)) for _ in range(25)])
    access = _access([(rng.uniform(0, 500), rng.uniform(0, 500)) for _ in range(6)])
    graph, access, buildings = get_road_graph_distances(nodes, edges, access, buildings)

    got = get_closest_park_manhattan(graph, buildings, access)
    expected = _brute_force(graph, buildings, access)
    pd.testing.assert_frame_equal(got[["building_id", "distance_manhattan"]], expected[["building_id", "distance_manhattan"]])
    # ties aside, the chosen access point must realise the reported distance
    assert got["closest_park_access_id"].notna().all()


def test_missing_edge_length_raises():
    nodes, edges = _grid_network(n=3)
    edges = edges.drop(columns="road_edge_length")
    with pytest.raises(ValueError, match="road_edge_length"):
        get_road_graph_distances(nodes, edges, _access([(0, 0)]), _buildings([(50, 50)]))


def test_no_parks_keeps_every_building_with_null_distances():
    nodes, edges = _grid_network(n=3)
    buildings = _buildings([(10, 10), (150, 90)])
    access = _access([])
    graph, access, buildings = get_road_graph_distances(nodes, edges, access, buildings)
    parks = gpd.GeoDataFrame({"park_id": []}, geometry=[], crs=CRS)

    result = get_closest_park(None, graph, buildings, access, parks)
    assert list(result["building_id"]) == ["B0", "B1"]
    assert result["distance_manhattan"].isna().all()
    assert result["distance_euclidean"].isna().all()


def test_euclidean_to_park_polygon():
    nodes, edges = _grid_network(n=3)
    buildings = _buildings([(10, 10)])
    access = _access([(100, 100)])
    graph, access, buildings = get_road_graph_distances(nodes, edges, access, buildings)
    parks = gpd.GeoDataFrame({"park_id": ["P0"]}, geometry=[box(100, 0, 150, 50)], crs=CRS)

    result = get_closest_park(None, graph, buildings, access, parks)
    # building disc edge (x = 14) to park edge (x = 100)
    assert result.loc[0, "distance_euclidean"] == pytest.approx(86.0, abs=0.1)
    assert result.loc[0, "closest_park_site_id"] == "P0"
