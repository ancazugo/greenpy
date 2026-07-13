# syntax=docker/dockerfile:1
# greenpy — self-contained image with OpenJDK 17, Python 3.12, and the Sedona
# JARs baked in. Lets users run the 3-30-300 pipeline without installing Java
# or Spark natively.
#
#   docker build -t greenpy .
#   docker run --rm -v /path/to/data:/data -e DATA_DIR=/data \
#     -v "$PWD":/work greenpy run -c /data/config.yaml -p T3
#
FROM python:3.12-slim-bookworm

# --- System deps: OpenJDK 17 (Sedona 1.9 needs Java 11+, Spark 3.5 caps at 17)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        openjdk-17-jre-headless \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # Resolve the arch-specific JDK dir and expose it at a stable path.
    && ln -s "$(dirname "$(dirname "$(readlink -f "$(command -v java)")")")" /opt/java-17

ENV JAVA_HOME=/opt/java-17
ENV PATH="/app/.venv/bin:${JAVA_HOME}/bin:${PATH}"

# --- uv for dependency install
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_PREFERENCE=only-system \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_LINK_MODE=copy

# --- Non-root user so the baked Ivy cache lives at a predictable, writable path.
# /work is the writable runtime dir (logs/, spark-warehouse/, cache/, outputs).
RUN useradd --create-home --home-dir /home/greenpy --shell /bin/bash greenpy \
    && mkdir -p /work && chown greenpy:greenpy /work

WORKDIR /app

# Install dependencies first (better layer caching), then the project itself.
# `--no-cache` keeps uv's download cache out of the image layers (pyspark alone
# ships sdist-only and unpacks ~300 MB of Spark jars). Plain RUN (no BuildKit
# `--mount`) so this builds on both the legacy and BuildKit builders.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache

COPY src/ ./src/
COPY examples/ ./examples/
COPY docker/ ./docker/
RUN uv sync --frozen --no-dev --no-cache

# --- Pre-download the Sedona/GeoTools/PostGIS jars into the Ivy cache.
# Runs as the greenpy user (the root-owned venv is world-readable/executable, so
# no chown of /app is needed) with modest memory so the JVM starts cheaply.
# The jars land in /home/greenpy/.ivy2, baked into the image.
USER greenpy
RUN SPARK_DRIVER_MEMORY=2g SPARK_EXECUTOR_MEMORY=1g python /app/docker/warmup.py

# --- Runtime defaults (laptop-friendly; override at `docker run` for big hosts)
ENV SPARK_DRIVER_MEMORY=4g \
    SPARK_EXECUTOR_MEMORY=2g

# Writable working dir for logs/, spark-warehouse/, cache/ and outputs.
WORKDIR /work

ENTRYPOINT ["greenpy"]
CMD ["--help"]
