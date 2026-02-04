
from utils.constants import JAVA_HOME

import os
import logging
from sedona.spark import SedonaContext
from pyspark.sql.session import SparkSession

def get_spark() -> SparkSession:
    """
    Sets up the Apache Sedona Spark session.
    
    Returns:
        SedonaContext (SparkSession): The Apache Sedona Spark session.
    """
    
    logging.debug("Setting up Apache Sedona")

    os.environ["JAVA_HOME"] = JAVA_HOME

    config = (
        SedonaContext.builder()
        .config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-spark-3.5_2.12:1.7.0,"
            "org.datasyslab:geotools-wrapper:1.7.0-28.5,"
            "net.postgis:postgis-jdbc:2021.1.0,"
            "net.postgis:postgis-geometry:2021.1.0,"
            "org.postgresql:postgresql:42.5.4,",
        )
        .config(
            "spark.jars.repositories",
            "https://artifacts.unidata.ucar.edu/repository/unidata-all",
        )
        # Shapefile WKT strings can get very long -- show more debugging info
        .config("spark.sql.debug.maxToStringFields", 10000)
        # No. of partitions given to the result of a shuffle by default.
        # (i.e. output of any rekeying of data)
        .config("spark.default.parallelism", 200)
        # Do not allow Spark to automatically coalesce partitions after shuffle
        # when it thinks that the data size is small enough to fit in one
        # partition. Often, we are using a large number of partitions to distribute large data resources that Spark does not know about.
        # Call coalese() explicitly after a shuffle when this behavior is
        # desired.
        # https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution
        .config("spark.sql.adaptive.coalescePartitions.enabled", False)
        .config("spark.executor.memory", "32g")
        .config("spark.driver.memory", "64g")
        .config("spark.driver.maxResultSize", "15g")  # Increase the max result size to 2g
        # Configure max number of concurrent tasks and allowable task failures
        # for spark in local mode.
        # https://spark.apache.org/docs/2.2.0/submitting-applications.html#master-urls
        # Format: local[num_tasks, num_failures]
        # N.b. Restrict number of concurrent tasks to
        #       1. Play nicely with other users on a shared machine
        #       2. Slow Spark down, esp. when tasks use resources spark doesn't
        #           know about. E.g. Querying a DB, making web requests,
        #           opening large files, etc.
        # N.b. Set max failures to 0 in development. Increase for deployment.
        .master("local[10,0]")
    ).getOrCreate()
    
    return SedonaContext.create(config)