
from utils.constants import INPUT_DIR, OUTPUT_DIR

# IN paths
vom_dir = INPUT_DIR / "Defra" / "VOM"
vom_unzipped_dir = vom_dir / "unzipped_tiles"

# # CDRC
imd_england_2019_path = INPUT_DIR / "CDRC" / "IMD" / "English IMD 2019" / "IMD_2019.shp"

# # Office for National Statistics (ONS)
oa_2021_lookup_path = INPUT_DIR / "ONS" / "Output_Area_to_Lower_layer_Super_Output_Area_to_Middle_layer_Super_Output_Area_to_Local_Authority_District_(December_2021)_Lookup_in_England_and_Wales_v3.csv"
oa_2021_boundaries_path = INPUT_DIR / "ONS" / "Output_Areas_(December_2021)_Boundaries_EW_BFE_(V9).geojson"
oa_rgn_lookup_path = INPUT_DIR / "ONS" / "OA21_RGN22_LU.csv"
lsoa_2011_2021_lookup_path = INPUT_DIR / "ONS" / "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Best_Fit_Lookup_for_EW_(V2).csv"
population_estimates_path = INPUT_DIR / "ONS" / "sapelsoabroadage20112022.xlsx"

# # Ordnance Survey (OS)
os_5km_boundaries_path = INPUT_DIR / "OS" / "National_Grid" / "5km_grid_region.shp"
green_space_path = INPUT_DIR / "OS" / "Green_Spaces" / "opgrsp_gb.gpkg"
roads_path = INPUT_DIR / "OS" / "Roads" / "oproad_gb.gpkg"

# # Verisk
buildings_path = INPUT_DIR / "Verisk" / "Buildings_6183" / "edition_17_0_new_format.gpkg"

# Google Earth Engine
output_areas_boundaries_ee_path = "projects/ee-phd-thesis/assets/output_areas_boundaries"

# OUT paths
T3_30_300_DIR = OUTPUT_DIR / "3-30-300"
T3_dir = T3_30_300_DIR / "T3"
T30_dir = T3_30_300_DIR / "T30"
T300_dir = T3_30_300_DIR / "T300"
Spectral_dir = T3_30_300_DIR / "Spectral"
trees_dir = T3_30_300_DIR / "VOM_Trees"
trees_unique_dir = T3_30_300_DIR / "VOM_Trees_unique"
trees_unique_parquet_dir = T3_30_300_DIR / "VOM_Trees_parquet"
tree_count_dir = T3_30_300_DIR / "Tree_count"
database_dir = T3_30_300_DIR / "database"

## Parquet files
vom_raster_paths_parquet = database_dir / "vom_raster_paths.parquet"
tree_vector_paths_parquet = database_dir / "tree_vector_paths.parquet"
os_tile_boundaries_parquet = database_dir / "os_tile_boundaries.parquet"
output_areas_boundaries_parquet = database_dir / "output_areas_boundaries.parquet"
output_areas_os_tile_overlay_parquet = database_dir / "output_areas_os_tile_overlay.parquet"
output_areas_buildings_parquet = database_dir / "output_areas_buildings.parquet"
imd_lsoa_parquet = database_dir / "imd_england_2019.parquet"
std_population_estimates_parquet = database_dir / "population_estimates.parquet"
green_space_access_parquet = database_dir / "green_space_access.parquet"
green_space_site_parquet = database_dir / "green_space_site.parquet"
road_edges_parquet = database_dir / "road_edges.parquet"
road_nodes_parquet = database_dir / "road_nodes.parquet"
buildings_parquet = database_dir / "verisk_buildings.parquet"
t30_parquet = database_dir / "T30.parquet"
t300_parquet = database_dir / "T300.parquet"
t3_300_parquet = database_dir / "T3_300.parquet"
t3_300_metrics_parquet = database_dir / "T3_300_metrics.parquet"
spectral_parquet = database_dir / "Spectral.parquet"
tree_count_parquet = database_dir / "Tree_count.parquet"
t3_30_300_spectral_parquet = database_dir / "T3_30_300_spectral.parquet"
imd_lsoa_bua_boundaries_path = OUTPUT_DIR / "IMD" / "English_IMD_2019_BUA_filtered_boundaries.geojson"