
from utils.paths import tree_count_dir
from utils.constants import PROJECT_CRS
from utils.data_processing import save_temp_file
from utils.sedona_rdd import create_spatial_rdds, count_trees_rdd

import time
import logging
import pandas as pd
import geopandas as gpd
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.session import SparkSession

def concatenate_trees_for_boundary(sedona: SparkSession, geo_level: str, geo_code: str, output_areas_os_tile_overlay_df: pd.DataFrame, tree_vector_paths_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Concatenates the trees for a given boundary.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo level.
        geo_code (str): The geo code.
        output_areas_os_tile_overlay_df (pd.DataFrame): The output areas OS tile overlay dataframe.
        tree_vector_paths_df (pd.DataFrame): The tree vector paths dataframe.

    Returns:
        gpd.GeoDataFrame: The concatenated trees for the boundary.
    """

    logging.debug(f"Getting trees for {geo_code}")

    geo_tile_lst = output_areas_os_tile_overlay_df[output_areas_os_tile_overlay_df[geo_level] == geo_code]['TILE_NAME_5KM_int'].str.upper().unique().tolist()

    tree_paths_lst = tree_vector_paths_df.copy().drop_duplicates(subset=['TILE_NAME'])[tree_vector_paths_df['TILE_NAME'].isin(geo_tile_lst)]['path'].tolist()
    trees_gdf_lst = []
    for gpkg_path in tree_paths_lst:
        temp_gdf = gpd.read_file(gpkg_path)
        trees_gdf_lst.append(temp_gdf)
    
    geo_trees_gdf = gpd.GeoDataFrame(pd.concat(trees_gdf_lst, ignore_index=True), crs=PROJECT_CRS)
    geo_trees_sdf = sedona.createDataFrame(geo_trees_gdf)
    # TODO: Figure out why sedona can't read multiple geopackages at once
    # geo_trees_sdf = sedona.read.format("geopackage").option('tableName','trees').load(tree_paths_lst)
    geo_trees_sdf.createOrReplaceTempView("geo_trees")
    
    geo_trees_sdf.withColumn("treeID", monotonically_increasing_id()).createOrReplaceTempView("geo_trees")

    return geo_trees_sdf

def process_geo_code(sedona: SparkSession, geo_level: str, sub_geo_level: str, geo_code: str, 
                     output_areas_os_tile_overlay_df: pd.DataFrame, tree_vector_paths_df: pd.DataFrame, overwrite: bool=True) -> pd.DataFrame:
    """
    Processes a given geo_code for tree count.

    Args:
        sedona (SparkSession): The Spark session.
        geo_level (str): The geo level.
        sub_geo_level (str): The sub geo level.
        geo_code (str): The geo code.
        output_areas_os_tile_overlay_df (pd.DataFrame): The output areas OS tile overlay dataframe.
        tree_vector_paths_df (pd.DataFrame): The tree vector paths dataframe.
        overwrite (bool): Whether to overwrite the existing file.
        
    Returns:
        pd.DataFrame: The tree count dataframe.
    """

    start_time = time.time()
    logging.info(f"Processing data for {geo_code}")

    geo_tree_count_path = tree_count_dir / f"Tree_count_{geo_code}.csv"

    if not geo_tree_count_path.exists() or overwrite:
        try:
            sub_geo_code_sdf = sedona.sql(f"""SELECT {sub_geo_level}, geometry
                                            FROM boundaries
                                            WHERE {geo_level} = '{geo_code}'""")
            
            geo_trees_sdf = concatenate_trees_for_boundary(sedona, geo_level, geo_code, output_areas_os_tile_overlay_df, tree_vector_paths_df)
            sub_geo_code_rdd, geo_trees_rdd = create_spatial_rdds(sub_geo_code_sdf, geo_trees_sdf, build_on_spatial_partitioned_rdd = True)
            geo_tree_count_df = count_trees_rdd(sedona, sub_geo_code_rdd, geo_trees_rdd, sub_geo_level, using_index = True)

            # geo_tree_count_df.to_csv(geo_tree_count_path, index=False)
            # Create a temp output folder (Spark writes here)
            # temp_dir = tree_count_dir / "_temp_tree_count"

            # geo_tree_count_df.coalesce(1) \
            #     .write \
            #     .option("header", True) \
            #     .mode("overwrite") \
            #     .csv(str(temp_dir))
            
            # # Step 2: Find the part file Spark wrote
            # part_file = glob.glob(str(temp_dir / "part-*.csv"))[0]
            
            # # Step 3: Move and rename it to your target file
            # shutil.move(part_file, str(geo_tree_count_path))

            # # Step 4: Clean up temp folder
            # shutil.rmtree(temp_dir)
            geo_tree_count_df = save_temp_file(geo_tree_count_df, geo_tree_count_path)
            end_time = time.time()
            logging.info(f"Processing for {geo_code} with {sum(geo_tree_count_df['tree_count'])} records took {end_time - start_time:.2f} seconds")

            return geo_tree_count_df

        except Exception as e:
            logging.error(f"Error processing {geo_code}: {e}")