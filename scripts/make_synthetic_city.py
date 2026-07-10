#!/usr/bin/env python3
"""
Generate a tiny synthetic city for end-to-end smoke-testing greenpy.

Creates buildings, trees, parks, roads and a 2-level census hierarchy
(2x2 districts of 2x2 tracts, 200 m tracts) plus a matching YAML config.

Usage: python scripts/make_synthetic_city.py <target_dir>
"""

import sys
import random
from pathlib import Path

import rasterio
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from shapely.geometry import Point, LineString, box

CRS = "EPSG:32632"
ORIGIN_X, ORIGIN_Y = 500_000, 5_000_000  # valid UTM 32N coordinates
TRACT = 200  # tract side in metres


def main(target: Path) -> None:
    random.seed(42)
    data_dir = target / "data"
    out_dir = target / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Census hierarchy: 2x2 districts, each 2x2 tracts
    records = []
    for di in range(2):
        for dj in range(2):
            for ti in range(2):
                for tj in range(2):
                    x0 = ORIGIN_X + (di * 2 + ti) * TRACT
                    y0 = ORIGIN_Y + (dj * 2 + tj) * TRACT
                    records.append({
                        "district_code": f"D{di}{dj}",
                        "tract_code": f"T{di}{dj}{ti}{tj}",
                        "geometry": box(x0, y0, x0 + TRACT, y0 + TRACT),
                    })
    census = gpd.GeoDataFrame(records, crs=CRS)
    census.to_file(data_dir / "census.gpkg", driver="GPKG")

    # Tract T1111 is reserved for a deterministic Visibility line-of-sight
    # scenario: observer building -> 8 m wall -> tree, colinear at y = +700.
    # No random feature may come within Visibility's buffer of it.
    reserved_tract = box(ORIGIN_X + 600, ORIGIN_Y + 600, ORIGIN_X + 800, ORIGIN_Y + 800)
    reserved_clear_zone = reserved_tract.buffer(60)

    # Buildings: 3 random 10x10 m squares per tract (with height bh in metres)
    buildings = []
    bid = 0
    for rec in records:
        if rec["tract_code"] == "T1111":
            continue
        minx, miny, maxx, maxy = rec["geometry"].bounds
        for _ in range(3):
            x = random.uniform(minx + 15, maxx - 25)
            y = random.uniform(miny + 15, maxy - 25)
            buildings.append({"bid": f"B{bid:04d}", "bh": round(random.uniform(4.0, 30.0), 1), "geometry": box(x, y, x + 10, y + 10)})
            bid += 1
    # one building without height to exercise the Visibility skip-warning path
    buildings[0]["bh"] = None
    # Deterministic trio: 20 m observer, 8 m wall 10 m in front of its facade,
    # 10 m tree 30 m from the facade. Visible from middle/top only.
    buildings.append({"bid": "B_OBS", "bh": 20.0, "geometry": box(ORIGIN_X + 640, ORIGIN_Y + 695, ORIGIN_X + 650, ORIGIN_Y + 705)})
    buildings.append({"bid": "B_WALL", "bh": 8.0, "geometry": box(ORIGIN_X + 660, ORIGIN_Y + 680, ORIGIN_X + 664, ORIGIN_Y + 720)})
    gpd.GeoDataFrame(buildings, crs=CRS).to_file(data_dir / "buildings.gpkg", driver="GPKG")

    # Trees: random canopy circles across the city, varying size and height
    trees_dir = data_dir / "trees"
    trees_dir.mkdir(exist_ok=True)
    trees = []
    for i in range(400):
        x = random.uniform(ORIGIN_X, ORIGIN_X + 4 * TRACT)
        y = random.uniform(ORIGIN_Y, ORIGIN_Y + 4 * TRACT)
        radius = random.uniform(1.0, 6.0)
        geom = Point(x, y).buffer(radius)
        if geom.intersects(reserved_clear_zone):
            continue
        trees.append({"h": round(random.uniform(1.5, 20.0), 1), "a": round(geom.area, 1), "geometry": geom})
    # the deterministic scenario's tree
    scenario_tree = Point(ORIGIN_X + 680, ORIGIN_Y + 700).buffer(2.0)
    trees.append({"h": 10.0, "a": round(scenario_tree.area, 1), "geometry": scenario_tree})
    gpd.GeoDataFrame(trees, crs=CRS).to_file(trees_dir / "trees_tile_1.gpkg", driver="GPKG")

    # CHM raster tiles: the same tree canopies rasterised at 1 m resolution
    # (pixel value = tree height, ground = 0) as two adjacent non-overlapping
    # tiles, so building buffers spanning the seam exercise cross-tile
    # aggregation in T30_buildings. A 120 m margin covers buffered buildings.
    chm_dir = data_dir / "chm"
    chm_dir.mkdir(exist_ok=True)
    margin = 120
    chm_miny, chm_maxy = ORIGIN_Y - margin, ORIGIN_Y + 4 * TRACT + margin
    mid_x = ORIGIN_X + 2 * TRACT
    shapes = [(t["geometry"], t["h"]) for t in trees]
    for name, x0, x1 in [
        ("chm_west", ORIGIN_X - margin, mid_x),
        ("chm_east", mid_x, ORIGIN_X + 4 * TRACT + margin),
    ]:
        width, height = int(x1 - x0), int(chm_maxy - chm_miny)
        transform = from_origin(x0, chm_maxy, 1.0, 1.0)
        arr = rasterize(shapes, out_shape=(height, width), transform=transform, fill=0.0, dtype="float32")
        with rasterio.open(
            chm_dir / f"{name}.tif", "w", driver="GTiff", height=height, width=width,
            count=1, dtype="float32", crs=CRS, transform=transform,
        ) as dst:
            dst.write(arr, 1)

    # Parks: two 80x80 m squares, with access points at their corners
    parks, accesses = [], []
    for pid, (px, py) in enumerate([(ORIGIN_X + 150, ORIGIN_Y + 150), (ORIGIN_X + 550, ORIGIN_Y + 550)]):
        parks.append({"pid": f"P{pid}", "geometry": box(px, py, px + 80, py + 80)})
        for cx, cy in [(px, py), (px + 80, py + 80)]:
            accesses.append({"pid": f"P{pid}", "geometry": Point(cx, cy)})
    gpd.GeoDataFrame(parks, crs=CRS).to_file(data_dir / "parks.gpkg", driver="GPKG")
    gpd.GeoDataFrame(accesses, crs=CRS).to_file(data_dir / "parks_access.gpkg", driver="GPKG")

    # Roads: grid every 100 m (edges only; nodes derived from endpoints)
    edges = []
    n_lines = 4 * TRACT // 100
    for i in range(n_lines + 1):
        offset = i * 100
        for j in range(n_lines):
            seg = j * 100
            edges.append(LineString([(ORIGIN_X + seg, ORIGIN_Y + offset), (ORIGIN_X + seg + 100, ORIGIN_Y + offset)]))
            edges.append(LineString([(ORIGIN_X + offset, ORIGIN_Y + seg), (ORIGIN_X + offset, ORIGIN_Y + seg + 100)]))
    roads = gpd.GeoDataFrame({"len_m": [e.length for e in edges]}, geometry=edges, crs=CRS)
    roads.to_file(data_dir / "roads.gpkg", driver="GPKG", layer="edges")

    config = f"""\
study_area_name: SynthCity
crs: {CRS}

data:
  buildings: {data_dir / "buildings.gpkg"}
  trees_dir: {trees_dir}/
  parks_sites: {data_dir / "parks.gpkg"}
  parks_access: {data_dir / "parks_access.gpkg"}
  roads: {data_dir / "roads.gpkg"}
  census_boundaries: {data_dir / "census.gpkg"}
  chm_tiles_dir: null

columns:
  building_id: bid
  building_layer: null
  building_height_col: bh
  road_edge_length: len_m
  road_edge_layer: edges
  road_node_layer: null
  park_id: pid
  park_function_col: null
  park_function_value: null
  park_access_ref_col: pid
  tree_height_col: h
  tree_area_col: a
  geo_levels:
    - district_code
    - tract_code

output:
  base_dir: {out_dir}/
"""
    (target / "config.yaml").write_text(config)

    # Variant config using the rasterised CHM tiles as canopy source (takes
    # priority over trees_dir), with its own output tree to avoid mixing runs.
    out_dir_chm = target / "output_chm"
    out_dir_chm.mkdir(parents=True, exist_ok=True)
    config_chm = config.replace("chm_tiles_dir: null", f"chm_tiles_dir: {chm_dir}/")
    config_chm = config_chm.replace(f"base_dir: {out_dir}/", f"base_dir: {out_dir_chm}/")
    (target / "config_chm.yaml").write_text(config_chm)

    print(f"Synthetic city written to {target} ({len(buildings)} buildings, {len(trees)} trees)")


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/greenpy_synth"))
