
from utils.constants import PROJECT_CRS, INPUT_DIR, OUTPUT_DIR
from utils.paths import imd_england_2019_path, lsoa_2011_2021_lookup_path, oa_2021_lookup_path, oa_2021_boundaries_path, oa_rgn_lookup_path 
from utils.paths import population_estimates_path, os_5km_boundaries_path, green_space_path, roads_path, buildings_path
from utils.paths import output_areas_boundaries_parquet, output_areas_buildings_parquet, std_population_estimates_parquet, imd_lsoa_parquet 
from utils.paths import os_tile_boundaries_parquet, green_space_access_parquet, green_space_site_parquet, road_edges_parquet, road_nodes_parquet
from utils.paths import buildings_parquet, vom_raster_paths_parquet, tree_vector_paths_parquet, output_areas_os_tile_overlay_parquet
from utils.paths import T3_30_300_DIR, T3_dir, T30_dir, T300_dir, trees_dir, Spectral_dir, vom_unzipped_dir, vom_dir, database_dir
from utils.data_processing import translate_tile_name

import logging
import pandas as pd
import geopandas as gpd
from pyspark.sql.session import SparkSession

def setup_parquet_files() -> None:
    """
    Sets up the parquet files for the project using the paths defined in utils.paths.
    """

    logging.debug("Setting up parquet files")

    imd_england_columns = ['lsoa11cd', 'IMD_Rank', 'IMD_Decile', 'IMDScore', 'IncScore', 
                           'IncRank', 'IncDec', 'EmpScore', 'EmpRank', 'EmpDec', 'EduScore',
                           'EduRank', 'EduDec', 'HDDScore', 'HDDRank', 'HDDDec', 'CriScore', 
                           'CriRank', 'CriDec', 'BHSScore', 'BHSRank', 'BHSDec', 'EnvScore', 
                           'EnvRank', 'EnvDec']
    
    logging.debug("Loading data from shapefiles and CSVs")

    imd_england_2019_gdf = gpd.read_file(imd_england_2019_path)[imd_england_columns].rename(columns={'lsoa11cd': 'LSOA11CD'})
    lsoa_2011_2021_lookup_df = pd.read_csv(lsoa_2011_2021_lookup_path)
    oa_2021_lookup_df = pd.read_csv(oa_2021_lookup_path)
    oa_2021_boundaries_gdf = gpd.read_file(oa_2021_boundaries_path).to_crs(PROJECT_CRS)
    oa_rgn_lookup_df = pd.read_csv(oa_rgn_lookup_path)
    oa_rgn_lookup_df.columns = oa_rgn_lookup_df.columns.str.upper()
    population_estimates_df = pd.read_excel(population_estimates_path, sheet_name='Mid-2022 LSOA 2021', skiprows=3)
    os_5km_boundaries_gdf = gpd.read_file(os_5km_boundaries_path).to_crs(PROJECT_CRS)
    green_space_access_gdf = gpd.read_file(green_space_path, layer='access_point')
    green_space_site_gdf = gpd.read_file(green_space_path, layer='greenspace_site').drop(columns=["distinctive_name_3", "distinctive_name_4"])
    road_edges_gdf = gpd.read_file(roads_path, layer='road_link')
    road_nodes_gdf = gpd.read_file(roads_path, layer='road_node')
    buildings_columns = ['verisk_building_id', 'verisk_premise_id', 'premise_year', 'premise_use', 
                        'premise_type', 'premise_floor_count', 'height', 'building_area', 
                        'distance_building', 'distance_water', 'map_use', 'map_simple_use', 'geometry']
    buildings_gdf = gpd.read_file(buildings_path, layer='edition_17_0_new_format', columns=buildings_columns)

    logging.debug("Processing data")

    output_areas_boundaries_columns = ['OA21CD', 'LSOA21CD', 'LSOA21NM', 'MSOA21CD', 'MSOA21NM', 
                                   'LAD22CD', 'LAD22NM', "RGN22CD", "RGN22NM", "area", "geometry"]
    output_areas_boundaries_gdf = oa_2021_boundaries_gdf.merge(oa_2021_lookup_df, on=["OA21CD", "LSOA21CD", "LSOA21NM"]) \
        .merge(oa_rgn_lookup_df, on="OA21CD") 
    output_areas_boundaries_gdf = output_areas_boundaries_gdf[output_areas_boundaries_gdf.RGN22CD != 'W92000004']
    output_areas_boundaries_gdf['area'] = output_areas_boundaries_gdf.geometry.area / 1_000_000
    output_areas_boundaries_gdf = output_areas_boundaries_gdf[output_areas_boundaries_columns]
    output_areas_boundaries_gdf
    std_population_estimates_df = process_population_data(population_estimates_df)
    imd_lsoa_gdf = imd_england_2019_gdf.merge(lsoa_2011_2021_lookup_df[["LSOA11CD", "LSOA21CD"]], on="LSOA11CD")
    imd_lsoa_df = imd_lsoa_gdf[["LSOA11CD", "LSOA21CD"] + imd_lsoa_gdf.columns[1:-1].tolist()]
    os_tile_boundaries_gdf = expand_national_grid(os_5km_boundaries_gdf)

    logging.debug("Saving data to parquet files")

    output_areas_boundaries_gdf.to_parquet(output_areas_boundaries_parquet, index=False)
    std_population_estimates_df.to_parquet(std_population_estimates_parquet, index=False)
    imd_lsoa_df.to_parquet(imd_lsoa_parquet, index=False)
    os_tile_boundaries_gdf.to_parquet(os_tile_boundaries_parquet, index=False)
    green_space_access_gdf.to_parquet(green_space_access_parquet, index=False)
    green_space_site_gdf.to_parquet(green_space_site_parquet, index=False)
    road_edges_gdf.to_parquet(road_edges_parquet, index=False)
    road_nodes_gdf.to_parquet(road_nodes_parquet, index=False)
    buildings_gdf.to_parquet(buildings_parquet, index=False)

    logging.debug("Parquet files created successfully")

def create_in_out_folders() -> None:
    """
    Creates the input and output folders for the project using the paths defined in utils.paths.
    """
    
    logging.debug("Creating input and output folders")
    
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    T3_30_300_DIR.mkdir(parents=True, exist_ok=True)
    T3_dir.mkdir(parents=True, exist_ok=True)
    T30_dir.mkdir(parents=True, exist_ok=True)
    T300_dir.mkdir(parents=True, exist_ok=True)
    Spectral_dir.mkdir(parents=True, exist_ok=True)
    trees_dir.mkdir(parents=True, exist_ok=True)
    vom_unzipped_dir.mkdir(parents=True, exist_ok=True)
    vom_dir.mkdir(parents=True, exist_ok=True)
    database_dir.mkdir(parents=True, exist_ok=True)

def load_tables(sedona: SparkSession) -> dict:
    """
    Loads the tables from the parquet files into a dictionary.

    Args:
        sedona (SparkSession): The Spark session.

    Returns:
        dict: A dictionary containing the loaded tables.
    """
    logging.debug("Loading tables from parquet files")

    # TODO: Standardise output format: geopandas dataframe or spark dataframe

    tables = {
        "vom_raster_paths_df": pd.read_parquet(vom_raster_paths_parquet),
        "tree_vector_paths_df": pd.read_parquet(tree_vector_paths_parquet),
        "output_areas_boundaries_gdf": gpd.read_parquet(output_areas_boundaries_parquet),
        "output_areas_os_tile_overlay_df": pd.read_parquet(output_areas_os_tile_overlay_parquet),
        "output_areas_buildings_overlay_sdf": sedona.read.format("parquet").load(str(output_areas_buildings_parquet)),
        "std_population_estimates_df": pd.read_parquet(std_population_estimates_parquet),
        "imd_lsoa_gdf": pd.read_parquet(imd_lsoa_parquet),
        "os_tile_boundaries_gdf": gpd.read_parquet(os_tile_boundaries_parquet),
        "green_space_access_gdf": gpd.read_parquet(green_space_access_parquet),
        "green_space_site_gdf": gpd.read_parquet(green_space_site_parquet),
        "road_edges_gdf": gpd.read_parquet(road_edges_parquet),
        "road_nodes_gdf": gpd.read_parquet(road_nodes_parquet),
        "buildings_sdf": sedona.read.format("geoparquet").load(str(buildings_parquet))
    }

    output_areas_boundaries_sdf = sedona.createDataFrame(tables["output_areas_boundaries_gdf"])
    output_areas_boundaries_sdf.createOrReplaceTempView('boundaries')
    tables["output_areas_buildings_overlay_sdf"].createOrReplaceTempView("boundaries_buildings_overlay")
    tables["buildings_sdf"].createOrReplaceTempView("buildings")
    public_park_site_gdf = tables["green_space_site_gdf"].copy()[tables["green_space_site_gdf"]['function'] == 'Public Park Or Garden'].reset_index(drop=True)
    public_park_site_sdf = sedona.createDataFrame(public_park_site_gdf)
    public_park_site_sdf.createOrReplaceTempView('public_park_sites')
    public_park_access_gdf = tables["green_space_access_gdf"].copy()[tables["green_space_access_gdf"]['ref_to_greenspace_site'].isin(public_park_site_gdf.id)].reset_index(drop=True)
    public_park_access_sdf = sedona.createDataFrame(public_park_access_gdf)
    public_park_access_sdf.createOrReplaceTempView('public_park_accesses')

    return tables

def process_population_data(population_estimates_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the population data to calculate the ratio of each column compared to Total.

    Args:
        population_estimates_df (pd.DataFrame): The population estimates dataframe.

    Returns:
        pd.DataFrame: The processed population estimates dataframe.
    """

    logging.debug("Processing population data")

    population_estimates_df.columns = population_estimates_df.columns.str.replace(' ', '_')
    # Calculate the ratio of each column compared to Total
    columns_to_calculate = ['F0_to_15', 'F16_to_29', 'F30_to_44', 'F45_to_64', 'F65_and_over', 
                            'M0_to_15', 'M16_to_29', 'M30_to_44', 'M45_to_64', 'M65_and_over']
    for column in columns_to_calculate:
        population_estimates_df[f'{column}_ratio'] = population_estimates_df[column] / population_estimates_df['Total']

    # Calculate the ratio of total F and total M
    population_estimates_df['Total_F'] = population_estimates_df[['F0_to_15', 'F16_to_29', 'F30_to_44', 'F45_to_64', 'F65_and_over']].sum(axis=1)
    population_estimates_df['Total_M'] = population_estimates_df[['M0_to_15', 'M16_to_29', 'M30_to_44', 'M45_to_64', 'M65_and_over']].sum(axis=1)
    population_estimates_df['F_ratio'] = population_estimates_df['Total_F'] / population_estimates_df['Total']
    population_estimates_df['M_ratio'] = population_estimates_df['Total_M'] / population_estimates_df['Total']

    # Calculate the ratio per age range
    age_ranges = ['0_to_15', '16_to_29', '30_to_44', '45_to_64', '65_and_over']
    for age_range in age_ranges:
        population_estimates_df[f'{age_range}_ratio'] = population_estimates_df[f'F{age_range}'] + population_estimates_df[f'M{age_range}']
        population_estimates_df[f'{age_range}_ratio'] /= population_estimates_df['Total']

    # Keep only the required columns
    columns_to_keep = ['LSOA_2021_Code', 'Total'] + [col for col in population_estimates_df.columns if col.endswith('_ratio')]
    std_population_estimates_df = population_estimates_df.copy()[columns_to_keep]
    std_population_estimates_df.rename(columns={'LSOA_2021_Code': 'LSOA11CD', 'Total': 'total_pop'}, inplace=True)

    return std_population_estimates_df

def expand_national_grid(os_5km_boundaries_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Expands the national grid to include the 10km, 50km, and 100km tiles.

    Args:
        os_5km_boundaries_gdf (gpd.GeoDataFrame): The OS 5km boundaries dataframe.

    Returns:
        gpd.GeoDataFrame: The expanded national grid dataframe.
    """

    logging.debug("Expanding national grid")

    def translate_code(code):
        ew = 'W' if int(code[2]) < 5 else 'E'
        ns = 'S' if int(code[3]) < 5 else 'N'
        return code[:2] + ns + ew
    
    os_tile_boundaries_gdf = os_5km_boundaries_gdf.copy()
    
    os_tile_boundaries_gdf.rename(columns={'TILE_NAME': 'TILE_NAME_5KM'}, inplace=True)
    os_tile_boundaries_gdf['TILE_NAME_5KM_int'] = os_tile_boundaries_gdf['TILE_NAME_5KM'].apply(lambda x: translate_tile_name(x))
    os_tile_boundaries_gdf['TILE_NAME_10KM'] = os_tile_boundaries_gdf['TILE_NAME_5KM'].apply(lambda x: x[:4])
    os_tile_boundaries_gdf['TILE_NAME_50KM'] = os_tile_boundaries_gdf['TILE_NAME_5KM'].apply(translate_code)
    os_tile_boundaries_gdf['TILE_NAME_100KM'] = os_tile_boundaries_gdf['TILE_NAME_5KM'].apply(lambda x: x[:2])
    os_tile_boundaries_gdf = os_tile_boundaries_gdf[['TILE_NAME_5KM', 'TILE_NAME_5KM_int', 'TILE_NAME_10KM', 'TILE_NAME_50KM', 'TILE_NAME_100KM', 'geometry']]
    
    return os_tile_boundaries_gdf

def overlay_output_areas_with_os_tiles(output_areas_boundaries_gdf: gpd.GeoDataFrame, os_tile_boundaries_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Overlays output areas with OS tiles and returns the resulting GeoDataFrame.

    Parameters:
        output_areas_boundaries_gdf (gpd.GeoDataFrame): GeoDataFrame containing output area boundaries.
        os_tile_boundaries_gdf (gpd.GeoDataFrame): GeoDataFrame containing OS tile boundaries.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the overlayed data.
    """

    logging.warning("Overlaying output areas with OS tiles")

    # Perform spatial join
    overlay_columns = ['OA21CD', 'LSOA21CD', 'MSOA21CD', 'LAD22CD', 'RGN22CD', 'TILE_NAME_5KM_int',
                       'TILE_NAME_5KM', 'TILE_NAME_10KM', 'TILE_NAME_50KM', 'TILE_NAME_100KM']
    output_areas_os_tile_overlay_df = output_areas_boundaries_gdf.copy().sjoin(os_tile_boundaries_gdf, how='left')[overlay_columns]

    output_areas_os_tile_overlay_df.to_parquet(output_areas_os_tile_overlay_parquet, index=False)

    
    return output_areas_os_tile_overlay_df

def overlay_output_areas_with_buildings(sedona: SparkSession, output_areas_boundaries_sdf: gpd.GeoDataFrame, buildings_sdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Overlay output areas with buildings to get the matching geography for each building.

    Args:
        sedona (SparkSession): The Spark session.
        output_areas_boundaries_sdf (gpd.GeoDataFrame): The output areas boundaries dataframe.
        buildings_sdf (gpd.GeoDataFrame): The buildings dataframe.

    Returns:
        gpd.GeoDataFrame: The buildings boundaries dataframe.
    """

    # Perform spatial join
    buildings_boundaries_sdf = sedona.sql(
    """
    SELECT b.verisk_premise_id, l.OA21CD, l.LSOA21CD, l.MSOA21CD, l.LAD22CD, l.RGN22CD
    FROM buildings b
    JOIN boundaries l
    ON ST_Contains(l.geometry, b.geometry)
    """
    )
    buildings_boundaries_df = buildings_boundaries_sdf.toPandas()
    buildings_boundaries_df.drop_duplicates(subset=['verisk_premise_id', 'OA21CD'], keep='first', inplace=True)
    
    return buildings_boundaries_df
