#!/usr/bin/env python3
import argparse
from pathlib import Path
import geopandas as gpd

from utils.constants import PROJECT_CRS
from utils.paths import T3_30_300_DIR
from tables_setup import *
from utils.logging_config import setup_logger
from utils.data_processing import get_geometries, get_overlapping_grid_tiles
from utils.sedona_config import get_spark
from t3 import read_vom_trees_unique
from t300 import filter_features

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--process', type=str, required=False, choices=['clip_data'], help='Name of the component of the module to run')
    parser.add_argument('--log_level', type=str, required=False, default='INFO', help='Logging level')
    parser.add_argument('--geo_level', type=str, required=True, default='LAD22CD', choices=['RGN22CD', 'MSOA21CD', 'LAD22CD', 'LSOA21CD'], help='Name/Code of the desired geography')
    parser.add_argument('--geo_code', type=str, required=True, help='Geographical variable name')
    parser.add_argument('--tile_level', type=str, required=False, default='TILE_NAME_5KM_int', help='Name/Code of the desired geography')
    parser.add_argument("--buildings", action='store_true')
    parser.add_argument("--trees", action='store_true')
    parser.add_argument("--green_spaces", action='store_true')
    parser.add_argument("--roads", action='store_true')
    parser.add_argument("--boundaries", action='store_true')
    
    args = parser.parse_args()

    args_dict = vars(args)
    geo_level = args_dict['geo_level']
    geo_code = args_dict['geo_code']
    tile_level = args_dict['tile_level']
    
    sedona = get_spark()

    tables = load_tables(sedona)
    
    log_path = Path(f"logs/{args_dict['process']}_processing.log")
    setup_logger(log_path=log_path, log_level=args_dict['log_level'])

    output_areas_boundaries_gdf = tables['output_areas_boundaries_gdf'] 
    os_tile_boundaries_gdf = tables['os_tile_boundaries_gdf'] 
    output_areas_os_tile_overlay_df = tables['output_areas_os_tile_overlay_df']
    output_areas_buildings_overlay_sdf = tables['output_areas_buildings_overlay_sdf']
    vom_raster_paths_df = tables['vom_raster_paths_df'] 
    tree_vector_paths_df = tables['tree_vector_paths_df']

    output_areas_boundaries_sdf = sedona.createDataFrame(output_areas_boundaries_gdf)
    output_areas_boundaries_sdf.createOrReplaceTempView('boundaries')
    output_areas_buildings_overlay_sdf.createOrReplaceTempView('output_areas_buildings_overlay')

    buildings_sdf = tables['buildings_sdf']
    buildings_sdf.createOrReplaceTempView('buildings')
    green_space_site_gdf = tables['green_space_site_gdf']
    green_space_access_gdf = tables['green_space_access_gdf']
    public_park_site_gdf = green_space_site_gdf.copy()[green_space_site_gdf['function'] == 'Public Park Or Garden'].reset_index(drop=True)
    public_park_site_sdf = sedona.createDataFrame(public_park_site_gdf)
    public_park_site_sdf.createOrReplaceTempView('public_park_sites')
    public_park_access_gdf = green_space_access_gdf.copy()[green_space_access_gdf['ref_to_greenspace_site'].isin(public_park_site_gdf.id)].reset_index(drop=True)
    public_park_access_sdf = sedona.createDataFrame(public_park_access_gdf)
    public_park_access_sdf.createOrReplaceTempView('public_park_accesses')
    road_nodes_gdf = tables['road_nodes_gdf'] 
    road_edges_gdf = tables['road_edges_gdf']
    road_nodes_sdf = sedona.createDataFrame(road_nodes_gdf)
    road_nodes_sdf.createOrReplaceTempView('road_nodes')
    road_edges_sdf = sedona.createDataFrame(road_edges_gdf)
    road_edges_sdf.createOrReplaceTempView('road_edges')

    geo_boundary_sdf = get_geometries(sedona, geo_level, geo_code, True)
    geo_boundary_sdf.createOrReplaceTempView('geo_boundary')
    geo_boundary_gdf = gpd.GeoDataFrame(geo_boundary_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)

    clipped_data_dir = T3_30_300_DIR / "clipped_data"
    geo_code_dir = clipped_data_dir / geo_code
    geo_code_dir.mkdir(parents=True, exist_ok=True)

    if (args_dict['buildings'] or args_dict['green_spaces'] or args_dict['roads']):
        geo_road_nodes_gdf, geo_road_edges_gdf, geo_public_park_sites_gdf, geo_public_park_accesses_gdf, geo_buildings_gdf = filter_features(sedona, geo_level, geo_code, 
                                                                                                                                            road_nodes_gdf, road_edges_gdf, 
                                                                                                                                            geo_boundary_gdf)

    if args_dict['boundaries']:
        geo_boundaries_sdf = get_geometries(sedona, geo_level, geo_code, False)
        geo_boundaries_gdf = gpd.GeoDataFrame(geo_boundaries_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
        geo_boundaries_gdf.to_parquet(geo_code_dir / f"{geo_code}_boundaries.parquet")

    if args_dict['buildings']:
        geo_buildings_gdf.to_parquet(geo_code_dir / f"{geo_code}_buildings.parquet")

    if args_dict['green_spaces']:
        geo_public_park_sites_gdf.to_parquet(geo_code_dir / f"{geo_code}_public_park_sites.parquet")
        geo_public_park_accesses_gdf.to_parquet(geo_code_dir / f"{geo_code}_public_park_accesses.parquet")

    if args_dict['roads']:
        geo_road_nodes_gdf.to_parquet(geo_code_dir / f"{geo_code}_road_nodes.parquet")
        geo_road_edges_gdf.to_parquet(geo_code_dir / f"{geo_code}_road_edges.parquet")

    if args_dict['trees']:
        overlapping_tiles_lst = get_overlapping_grid_tiles(output_areas_boundaries_gdf, os_tile_boundaries_gdf, geo_level, geo_code, tile_level)
        overlapping_tiles_lst = [tile_name.upper() for tile_name in overlapping_tiles_lst]
        geo_trees_sdf = read_vom_trees_unique(sedona, overlapping_tiles_lst, 0, 0)

        geo_trees_filtered_sdf = sedona.sql("""SELECT t.* FROM geo_boundary b
               LEFT JOIN geo_trees t
               ON ST_Intersects(b.geometry, t.geometry)
               """)
        
        geo_trees_filtered_gdf = gpd.GeoDataFrame(geo_trees_filtered_sdf.toPandas(), geometry='geometry', crs=PROJECT_CRS)
        geo_trees_filtered_gdf.to_parquet(geo_code_dir / f"{geo_code}_trees.parquet")