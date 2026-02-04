
import logging
import glob
import shutil
import tempfile
import pandas as pd
import geopandas as gpd
from pathlib import Path
from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame

def translate_tile_name(tile_name: str) -> str:
    """
    Translates a tile name between two formats:

    - Format 1: TL0045 (where '0045' represents coordinates)
    - Format 2: TL04NW (where 'NW' represents directions)

    The function converts:
    - From Format 1 to Format 2 by interpreting the numeric coordinates and converting them to directional codes.
    - From Format 2 to Format 1 by interpreting the directional codes and converting them to numeric coordinates.

    Args:
        tile_name (str): The tile name to be translated. It should be a string of length 6.

    Returns:
        str: The translated tile name in the opposite format.

    Raises:
        AssertionError: If the input tile_name is not of length 6.
        ValueError: If the numeric part of the tile name cannot be converted to an integer when expected.
    """
    
    NS_dict = {'S': '0', 'N': '5'}
    EW_dict = {'W': '0', 'E': '5'} 

    assert len(tile_name) == 6
    
    code = tile_name[2:6].upper()
    try: # If input is like TL0045
        int(code)
        NS_dict = {v: k for k, v in NS_dict.items()}
        EW_dict = {v: k for k, v in EW_dict.items()}
        ns_id = code[3]
        ew_id = code[1]
        direction_code = code[0] + code[2] + NS_dict[ns_id] + EW_dict[ew_id]
        trans_tile_name = tile_name[:2].upper() + direction_code
    except ValueError: # If input is like TL04NW
        ns_id = code[2]
        ew_id = code[3]
        number_code = code[0] + EW_dict[ew_id] + code[1] + NS_dict[ns_id]
        trans_tile_name = tile_name[:2].lower() + number_code

    return trans_tile_name

def get_overlapping_grid_tiles(output_areas_boundaries_gdf: gpd.GeoDataFrame, os_tile_boundaries_gdf: gpd.GeoDataFrame, geo_level: str, geo_code: str, tile_level: str) -> list:
    """
    Gets the overlapping grid tiles for a given geo_code and tile_level.

    Args:
        output_areas_boundaries_gdf (gpd.GeoDataFrame): The output areas boundaries dataframe.
        os_tile_boundaries_gdf (gpd.GeoDataFrame): The OS tile boundaries dataframe.
        geo_level (str): The geo_level.
        geo_code (str): The geo_code.
        tile_level (str): The tile_level.

    Returns:
        list: The overlapping grid tiles.
    """

    logging.debug(f"Getting overlapping grid tiles for {geo_code} and {tile_level}")
    
    selected_feature = output_areas_boundaries_gdf[output_areas_boundaries_gdf[geo_level] == geo_code]

    # Get the overlapping features
    overlapping_tiles_lst = gpd.overlay(selected_feature, os_tile_boundaries_gdf, how='intersection')[tile_level].unique().tolist()

    return overlapping_tiles_lst

def generate_tile_paths(geo_level: str, geo_code: str, output_areas_os_tile_overlay_df: gpd.GeoDataFrame, vom_raster_paths_df: pd.DataFrame, tree_vector_paths_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates the tile paths for a given geo_code.

    Args:
        geo_level (str): The geo_level.
        geo_code (str): The geo_code.
        output_areas_os_tile_overlay_df (gpd.GeoDataFrame): The output areas OS tile overlay dataframe.
        vom_raster_paths_df (pd.DataFrame): The VOM raster paths dataframe.
        tree_vector_paths_df (pd.DataFrame): The tree vector paths dataframe.

    Returns:
        pd.DataFrame: The tile paths dataframe.
    """
    
    logging.debug(f"Generating tile (vom and tree) paths for {geo_code}")

    geo_output_areas_os_tile_overlay_df = output_areas_os_tile_overlay_df.copy()[output_areas_os_tile_overlay_df[geo_level] == geo_code]
    vom_raster_paths_df['TILE_NAME_5KM_int'] = vom_raster_paths_df.TILE_NAME.apply(lambda x: x.lower())
    tree_vector_paths_df['TILE_NAME_5KM_int'] = tree_vector_paths_df.TILE_NAME.apply(lambda x: x.lower())
    geo_vom_tiles_df = geo_output_areas_os_tile_overlay_df.merge(vom_raster_paths_df, on='TILE_NAME_5KM_int', how='left')[['TILE_NAME', 'year', 'path']].drop_duplicates().sort_values(['TILE_NAME', 'year'], ascending=[True, False]).reset_index(drop=True)
    geo_tree_tiles_df = geo_output_areas_os_tile_overlay_df.merge(tree_vector_paths_df, on='TILE_NAME_5KM_int', how='left')[['TILE_NAME', 'year', 'path']].drop_duplicates().sort_values(['TILE_NAME', 'year'], ascending=[True, False]).reset_index(drop=True)

    geo_tiles_df = geo_vom_tiles_df.merge(geo_tree_tiles_df, on=['TILE_NAME', 'year'], suffixes=['_vom', '_tree'])

    return geo_tiles_df

def filter_buffer_geometries(sedona: SparkSession, geo_level: str, geo_code: str, table_name: str, buffer: int=None) -> DataFrame:
    """
    Filters the buffer geometries for a given geo_code.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo_level.
        geo_code (str): The geo_code.
        table_name (str): The table name.
        buffer (int): The buffer size.

    Returns:
        DataFrame: The filtered buffer geometries dataframe.
    """
    
    logging.debug(f"Filtering {table_name} for {geo_code}")

    geo_buildings_sdf = sedona.sql(
        f"""
            SELECT b.* FROM {table_name} b, geo_boundary g 
            WHERE ST_Intersects(b.geometry, g.geometry)
        """)
    geo_buildings_sdf.createOrReplaceTempView(f"geo_{table_name}")

    if buffer:

        geo_buildings_buffer_sdf = sedona.sql(
        f"""
            SELECT ST_Buffer(b.geometry, {buffer}) AS geometry, b.verisk_premise_id
            FROM geo_{table_name} b
        """)
        geo_buildings_buffer_sdf.createOrReplaceTempView(f"{table_name}_buffers")

        return geo_buildings_buffer_sdf
    
    return geo_buildings_sdf

def get_geometries(sedona: SparkSession, geo_level: str, geo_code: str, dissolve: bool=True) -> DataFrame:
    """
    Gets the geometries for a given geo_code.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo_level.
        geo_code (str): The geo_code.
        dissolve (bool): Whether to dissolve the geometries.

    Returns:
        DataFrame: The geometries dataframe.
    """
    
    logging.debug(f"Dissolving geometries for {geo_level}:{geo_code}")

    query = "ST_Union_Aggr(geometry) AS geometry" if dissolve else "*"

    geo_boundary_sdf = sedona.sql(
        f"""
            SELECT {query}
            FROM boundaries
            WHERE {geo_level} = '{geo_code}'
        """)
    geo_boundary_sdf.createOrReplaceTempView("geo_boundary")

    return geo_boundary_sdf

def save_csv_as_parquet(in_directory: Path, path_pattern: str, out_path: Path) -> pd.DataFrame:
    """
    Saves the CSV files as a parquet file.

    Args:
        in_directory (Path): The input directory.
        path_pattern (str): The path pattern.
        out_path (Path): The output path.

    Returns:
        pd.DataFrame: The concatenated dataframe.
    """
    
    csv_files = list(in_directory.glob(path_pattern))
    dataframes_lst = [pd.read_csv(file) for file in csv_files]
    concatenated_df = pd.concat(dataframes_lst, ignore_index=True)
    concatenated_df.to_parquet(out_path, index=False)

    return concatenated_df

def save_temp_file(spark_df: DataFrame, output_path: Path, coalesce: int=1, file_format: str="csv") -> pd.DataFrame:
    """
    Saves a Spark DataFrame to a single named file and returns a Pandas DataFrame.

    This function is a workaround for the fact that Spark normally writes to a directory.
    It works by coalescing the DataFrame to a single partition, writing to a temporary
    directory, and then moving the single generated data file to the final destination.

    Args:
        spark_df (DataFrame): The Spark DataFrame to save.
        output_path (Path): The final, full path for the output file (e.g., Path("/data/my_file.parquet")).
        file_format (str): The format to save the file in ("parquet", "csv", etc.).

    Returns:
        pd.DataFrame: The saved data loaded into a Pandas DataFrame.
    """
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Write the coalesced DataFrame to the temporary directory.
        # Spark will create a directory with part-files and metadata inside.
        spark_df.coalesce(coalesce) \
            .write \
            .option("header", "true") \
            .mode("overwrite") \
            .format(file_format) \
            .save(str(temp_path))

        # Step 2: Find the single part-file Spark wrote.
        # It will have a name like 'part-00000-....c000.snappy.parquet'
        part_files = list(temp_path.glob(f"part-*.{file_format}*"))
        if not part_files:
            raise FileNotFoundError(f"No part file found with format '{file_format}' in {temp_dir}")
        
        temp_file = part_files[0]
        
        # Step 3: Move and rename the part-file to your target path.
        # This moves the actual data file to its final destination.
        shutil.move(temp_file, output_path)

    # Step 4: Read the final file into Pandas using the correct reader.
    if file_format == "parquet":
        pandas_df = pd.read_parquet(output_path)
    elif file_format == "csv":
        pandas_df = pd.read_csv(output_path)
    else:
        # Add other formats as needed or raise an error
        raise ValueError(f"Unsupported file format for reading into Pandas: {file_format}")

    return pandas_df
