
import logging

from sedona.utils.adapter import Adapter
from sedona.core.enums import GridType, IndexType
from sedona.core.spatialOperator import JoinQueryRaw
from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame
from sedona.core.SpatialRDD import SpatialRDD

def create_spatial_rdds(query_sdf: DataFrame, object_sdf: DataFrame, build_on_spatial_partitioned_rdd: bool = True) -> tuple:
    """
    Creates Spatial RDDs for two spatial dataframes.

    Args:
        query_sdf (DataFrame): The query dataframe.
        object_sdf (DataFrame): The object dataframe.
        build_on_spatial_partitioned_rdd (bool): Whether to build on spatial partitioned RDD.

    Returns:    
        tuple: The query and object RDDs.
    """
    
    logging.debug("Creating Spatial RDDs for two spatial dataframes")

    query_rdd  = Adapter.toSpatialRdd(query_sdf, 'geometry')
    object_rdd = Adapter.toSpatialRdd(object_sdf, 'geometry')
    
    query_rdd.analyze()
    object_rdd.analyze()

    object_rdd.spatialPartitioning(GridType.KDBTREE)
    query_rdd.spatialPartitioning(object_rdd.getPartitioner())
    
    query_rdd.buildIndex(IndexType.QUADTREE, build_on_spatial_partitioned_rdd)

    return query_rdd, object_rdd

def count_trees_rdd(sedona: SparkSession, query_rdd: SpatialRDD, object_rdd: SpatialRDD, query_column: str, using_index: bool = True) -> DataFrame:
    """
    Counts the trees for each area using RDD.

    Args:
        sedona (SparkSession): The Spark session.
        query_rdd (SpatialRDD): The query RDD.
        object_rdd (SpatialRDD): The object RDD.
        query_column (str): The query column.   
        using_index (bool): Whether to use the index.
        
    Returns:
        DataFrame: The dataframe with the tree counts.
    """
    
    logging.debug("Counting trees for each area using RDD")

    query_result = JoinQueryRaw.SpatialJoinQueryFlat(object_rdd, query_rdd, using_index, True)

    query_result_sdf = Adapter.toDf(query_result, [query_column], ["treeID"], sedona)

    # query_result_df = query_result_sdf.toPandas().sort_values(by=query_column)

    # geo_tree_count_df = query_result_df.groupby(query_column).size().reset_index(name='tree_count')

    geo_tree_count_df = (
        query_result_sdf
        .groupBy(query_column)
        .count()
        .withColumnRenamed("count", "tree_count")
        .orderBy(query_column)
    )

    return geo_tree_count_df
