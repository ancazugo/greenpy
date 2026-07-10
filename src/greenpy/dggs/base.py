"""Base classes for DGGS grid builders.

A system only has to say how cells are generated for a geometry; everything
downstream — deduplication, parent-census-unit assignment, overlap filtering,
area computation and the buildings overlay — is shared and system-agnostic.
Systems with native Sedona SQL support (H3, S2) extend SedonaSQLDGGS and run
fully distributed; the rest (geohash, A5, rHEALPix) extend PythonDGGS and
compute covering cells driver-side, which is fine at city scale but for
country-scale study areas the Sedona-backed systems are recommended.
"""

import abc
from typing import ClassVar, Iterable

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyspark.sql.session import SparkSession
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


def _finalize_boundaries(
    sedona: SparkSession, pairs_view: str, col_name: str, geo_levels: list[str], crs: str
) -> gpd.GeoDataFrame:
    """Turn a (census codes, unit_geometry, cell, cell_geometry) pairs view into the grid table.

    Every cell overlapping the study area is kept, so no feature inside the
    area falls outside the grid. Each cell belongs to the census unit
    containing its centre (the most-overlapping unit when the centre lies just
    outside). Cells are kept whole (not clipped), so cell areas stay uniform;
    stats near the border may include features slightly outside the study area.
    Candidate cells that don't actually touch any unit are dropped, so
    over-inclusive generators (e.g. bounding-box enumeration) are fine.
    """
    cols = ", ".join(geo_levels)
    p_cols = ", ".join(f"p.{c}" for c in geo_levels)
    sdf = sedona.sql(
        f"""
        WITH cells AS (
            SELECT cell, FIRST(cell_geometry) AS geometry
            FROM {pairs_view} GROUP BY cell
        ),
        ranked AS (
            SELECT c.cell, c.geometry, {p_cols},
                   ST_Intersects(c.geometry, p.unit_geometry) AS overlaps_unit,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.cell
                       ORDER BY ST_Contains(p.unit_geometry, ST_Centroid(c.geometry)) DESC,
                                ST_Area(ST_Intersection(c.geometry, p.unit_geometry)) DESC
                   ) AS rn
            FROM cells c JOIN {pairs_view} p ON c.cell = p.cell
        )
        SELECT cell AS {col_name}, {cols},
               ST_Area(geometry) / 1000000 AS area, geometry
        FROM ranked WHERE rn = 1 AND overlaps_unit
        """
    )
    pdf = sdf.toPandas()
    if pdf.empty:
        raise ValueError(f"No {col_name} cells overlap the study area.")
    grid_gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs=crs)
    logger.info(f"{col_name} grid built: {len(grid_gdf)} cells assigned to census units")
    return grid_gdf


class DGGS(abc.ABC):
    """A Discrete Global Grid System usable as the aggregation unit."""

    name: ClassVar[str]
    min_resolution: ClassVar[int]
    max_resolution: ClassVar[int]
    resolution_term: ClassVar[str] = "resolution"

    def column_name(self, resolution: int) -> str:
        """Column name holding the cell id at a given resolution."""
        return f"{self.name}_{resolution}"

    def validate_resolution(self, resolution: int) -> None:
        if not (isinstance(resolution, int) and self.min_resolution <= resolution <= self.max_resolution):
            raise ValueError(
                f"{self.name} {self.resolution_term} must be an integer between "
                f"{self.min_resolution} and {self.max_resolution}, got {resolution!r}"
            )

    @abc.abstractmethod
    def build_boundaries(
        self, sedona: SparkSession, census_gdf: gpd.GeoDataFrame,
        resolution: int, geo_levels: list[str], crs: str,
    ) -> gpd.GeoDataFrame:
        """Grid table covering the census area: cell column + geo_levels + area + geometry."""

    def build_buildings_overlay(
        self, sedona: SparkSession, buildings_view: str, grid_view: str, code_cols: list[str]
    ) -> pd.DataFrame:
        """Return a building_id → cell/census-codes lookup as a pandas DataFrame.

        Buildings are reduced to a point on their surface so each maps to exactly
        one cell (mirrors the representative-point logic of the census overlay).
        """
        sel = ", ".join(f"g.{c}" for c in code_cols)
        overlay_df = sedona.sql(
            f"""
            SELECT b.building_id, {sel}
            FROM {buildings_view} b
            JOIN {grid_view} g ON ST_Contains(g.geometry, ST_PointOnSurface(b.geometry))
            """
        ).toPandas()
        return overlay_df.drop_duplicates(subset="building_id")


class SedonaSQLDGGS(DGGS):
    """DGGS whose cells are generated with native Sedona SQL functions (distributed)."""

    @abc.abstractmethod
    def cell_ids_sql(self, geom_expr: str, resolution: int) -> str:
        """SQL expression: array of raw cell ids covering an EPSG:4326 geometry."""

    @abc.abstractmethod
    def cell_id_str_sql(self, cell_expr: str) -> str:
        """SQL expression: string form of a raw cell id."""

    @abc.abstractmethod
    def cell_geom_sql(self, cell_expr: str) -> str:
        """SQL expression: EPSG:4326 polygon of a raw cell id."""

    def build_boundaries(self, sedona, census_gdf, resolution, geo_levels, crs):
        self.validate_resolution(resolution)
        logger.info(f"Building {self.name} grid at {self.resolution_term} {resolution}")
        cols = ", ".join(geo_levels)
        sedona.createDataFrame(census_gdf[geo_levels + ["geometry"]]) \
            .createOrReplaceTempView("dggs_census_src")
        geom_4326 = f"ST_Transform(geometry, '{crs}', 'EPSG:4326')"
        sedona.sql(
            f"""
            SELECT {cols}, geometry AS unit_geometry,
                   {self.cell_id_str_sql('cell')} AS cell,
                   ST_Transform({self.cell_geom_sql('cell')}, 'EPSG:4326', '{crs}') AS cell_geometry
            FROM (
                SELECT {cols}, geometry,
                       explode({self.cell_ids_sql(geom_4326, resolution)}) AS cell
                FROM dggs_census_src
            )
            """
        ).createOrReplaceTempView("dggs_pairs")
        return _finalize_boundaries(sedona, "dggs_pairs", self.column_name(resolution), geo_levels, crs)


class PythonDGGS(DGGS):
    """DGGS whose cells are generated driver-side with a Python library."""

    @abc.abstractmethod
    def cells_covering(
        self, geom_wgs84: BaseGeometry, resolution: int
    ) -> Iterable[tuple[str, Polygon]]:
        """(cell id, EPSG:4326 polygon) pairs covering a geometry; may over-include."""

    def build_boundaries(self, sedona, census_gdf, resolution, geo_levels, crs):
        self.validate_resolution(resolution)
        logger.info(f"Building {self.name} grid at {self.resolution_term} {resolution}")
        units = census_gdf[geo_levels + ["geometry"]]
        units_4326 = units.to_crs("EPSG:4326")

        rows = []
        for (_, unit), geom_4326 in zip(units.iterrows(), units_4326.geometry):
            for cell_id, cell_poly in self.cells_covering(geom_4326, resolution):
                rows.append([unit[c] for c in geo_levels] + [cell_id, unit.geometry.wkb, cell_poly])
        pairs = pd.DataFrame(rows, columns=geo_levels + ["cell", "unit_wkb", "cell_poly"])
        # Reproject the cell polygons in one pass, then ship both geometries as WKB
        pairs["cell_wkb"] = gpd.GeoSeries(pairs.pop("cell_poly"), crs="EPSG:4326").to_crs(crs).to_wkb()

        sedona.createDataFrame(pairs).createOrReplaceTempView("dggs_pairs_raw")
        cols = ", ".join(geo_levels)
        sedona.sql(
            f"""
            SELECT {cols}, cell,
                   ST_GeomFromWKB(unit_wkb) AS unit_geometry,
                   ST_GeomFromWKB(cell_wkb) AS cell_geometry
            FROM dggs_pairs_raw
            """
        ).createOrReplaceTempView("dggs_pairs")
        return _finalize_boundaries(sedona, "dggs_pairs", self.column_name(resolution), geo_levels, crs)
