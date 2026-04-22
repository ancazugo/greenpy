from __future__ import annotations

from loguru import logger
from typing import TYPE_CHECKING

from sedona.utils.adapter import Adapter
from sedona.core.enums import GridType, IndexType
from sedona.core.spatialOperator import JoinQueryRaw
from pyspark.sql.session import SparkSession
from pyspark.sql.dataframe import DataFrame

if TYPE_CHECKING:
    from sedona.core.SpatialRDD import SpatialRDD


def create_spatial_rdds(query_sdf: DataFrame, object_sdf: DataFrame, build_on_spatial_partitioned_rdd: bool = True) -> tuple:
    logger.debug("Creating Spatial RDDs for two spatial dataframes")

    query_rdd = Adapter.toSpatialRdd(query_sdf, "geometry")
    object_rdd = Adapter.toSpatialRdd(object_sdf, "geometry")

    query_rdd.analyze()
    object_rdd.analyze()

    object_rdd.spatialPartitioning(GridType.KDBTREE)
    query_rdd.spatialPartitioning(object_rdd.getPartitioner())
    query_rdd.buildIndex(IndexType.QUADTREE, build_on_spatial_partitioned_rdd)

    return query_rdd, object_rdd


def count_trees_rdd(sedona: SparkSession, query_rdd: SpatialRDD, object_rdd: SpatialRDD, query_column: str, using_index: bool = True) -> DataFrame:
    logger.debug("Counting trees for each area using RDD")

    query_result = JoinQueryRaw.SpatialJoinQueryFlat(object_rdd, query_rdd, using_index, True)
    query_result_sdf = Adapter.toDf(query_result, [query_column], ["treeID"], sedona)

    geo_tree_count_df = (
        query_result_sdf
        .groupBy(query_column)
        .count()
        .withColumnRenamed("count", "tree_count")
        .orderBy(query_column)
    )

    return geo_tree_count_df
