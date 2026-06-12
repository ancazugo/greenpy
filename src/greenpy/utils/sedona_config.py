import os
from loguru import logger
from sedona.spark import SedonaContext
from pyspark.sql.session import SparkSession

from .constants import JAVA_HOME


def get_spark() -> SparkSession:
    """Create (or reuse) a local Sedona-enabled SparkSession with the project's JAR packages."""
    logger.debug("Setting up Apache Sedona")

    if JAVA_HOME:
        os.environ["JAVA_HOME"] = JAVA_HOME

    config = (
        SedonaContext.builder()
        .config(
            "spark.jars.packages",
            "org.apache.sedona:sedona-spark-3.5_2.12:1.7.1,"
            "org.datasyslab:geotools-wrapper:1.7.1-28.5,"
            "net.postgis:postgis-jdbc:2021.1.0,"
            "net.postgis:postgis-geometry:2021.1.0,"
            "org.postgresql:postgresql:42.5.4",
        )
        .config(
            "spark.jars.repositories",
            "https://artifacts.unidata.ucar.edu/repository/unidata-all",
        )
        .config("spark.sql.debug.maxToStringFields", 10000)
        .config("spark.default.parallelism", 200)
        .config("spark.sql.adaptive.coalescePartitions.enabled", False)
        .config("spark.executor.memory", "32g")
        .config("spark.driver.memory", "64g")
        .config("spark.driver.maxResultSize", "15g")
        .master("local[10,0]")
    ).getOrCreate()

    return SedonaContext.create(config)
