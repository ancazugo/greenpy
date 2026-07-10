"""Deprecated: H3 grid generation moved to greenpy.dggs (use get_system("h3"))."""

import warnings

import geopandas as gpd
from pyspark.sql.session import SparkSession

from ..dggs.h3 import H3System


def _warn(name: str) -> None:
    warnings.warn(
        f"greenpy.utils.h3_boundaries.{name} is deprecated; use greenpy.dggs.get_system('h3')",
        DeprecationWarning,
        stacklevel=3,
    )


def h3_column(resolution: int) -> str:
    """Column name holding the H3 cell id at a given resolution."""
    _warn("h3_column")
    return H3System().column_name(resolution)


def build_h3_boundaries(
    sedona: SparkSession, census_view: str, resolution: int, geo_levels: list[str], crs: str
) -> gpd.GeoDataFrame:
    """Generate full H3 hexagons covering the study area, tagged with parent census codes."""
    _warn("build_h3_boundaries")
    census_gdf = gpd.GeoDataFrame(
        sedona.table(census_view).toPandas(), geometry="geometry", crs=crs
    )
    return H3System().build_boundaries(sedona, census_gdf, resolution, geo_levels, crs)


def build_h3_buildings_overlay(
    sedona: SparkSession, buildings_view: str, h3_view: str, code_cols: list[str]
):
    """Return a building_id → hexagon/census-codes lookup as a pandas DataFrame."""
    _warn("build_h3_buildings_overlay")
    return H3System().build_buildings_overlay(sedona, buildings_view, h3_view, code_cols)
