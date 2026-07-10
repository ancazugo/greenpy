#!/usr/bin/env python3
from utils.constants import PROJECT_CRS
from utils.paths import trees_unique_dir, trees_unique_parquet_dir, T3_30_300_DIR
import shutil
import os

from tables_setup import *
from utils.sedona_config import get_spark

from pyspark.sql.functions import monotonically_increasing_id

if __name__ == '__main__':
    
    sedona = get_spark()

    # tables = load_tables(sedona)
    # tree_vector_paths_df = tables['tree_vector_paths_df']
    # # Create target directory if it doesn't exist
    # os.makedirs(trees_unique_dir, exist_ok=True)
    # unique_tree_paths = tree_vector_paths_df.drop_duplicates(subset=['TILE_NAME'])['path'].tolist()
    # # Copy each uniquefile to the target directory
    # for path in unique_tree_paths:
    #     filename = os.path.basename(path)
    #     target_path = trees_unique_dir / filename
    #     shutil.copy2(path, target_path)
    # Save files as geoparquet in web mercator projection and centroided
    trees_sdf = sedona.read.format("geopackage").option("tableName", "trees").load(str(trees_unique_dir))
    trees_sdf.createOrReplaceTempView("trees")

    trees_sdf = sedona.sql(f"""SELECT treeID, height, area, ST_Transform(ST_Centroid(geom), "{PROJECT_CRS}", "EPSG:4326") AS geometry FROM trees""")
    trees_sdf =  trees_sdf.withColumn("treeID", monotonically_increasing_id())

    trees_sdf.coalesce(50).write.format("geoparquet").save(str(T3_30_300_DIR / "App_files" / "Trees"))
