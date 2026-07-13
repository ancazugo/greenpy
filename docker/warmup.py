"""Build-time Sedona/Spark JAR warm-up.

Starting a Sedona session forces Ivy to resolve and download every jar listed
in `spark.jars.packages` (Sedona, GeoTools, PostGIS, PostgreSQL) into the Ivy
cache. Running this during `docker build` bakes those jars into the image so
the container starts fast and works with no network at runtime.

Uses the real `get_spark()` so the cached jars exactly match what runtime asks
for. Keep the driver/executor memory small at build time via the
SPARK_DRIVER_MEMORY / SPARK_EXECUTOR_MEMORY env vars (set in the Dockerfile).
"""

from greenpy.utils.sedona_config import get_spark


def main() -> None:
    sedona = get_spark()
    print(f"Sedona/Spark {sedona.sparkContext.version} session started for warm-up; jars cached.")
    sedona.stop()
    print("JAR warm-up complete.")


if __name__ == "__main__":
    main()
