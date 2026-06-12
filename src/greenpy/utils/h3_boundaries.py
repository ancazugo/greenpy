"""H3 hexagon grid generation for uniform statistics aggregation.

Builds a hexagon table shaped like the census boundaries table (geo_levels
columns + geometry + area) so it can serve as the `boundaries` Spark view,
with one extra `h3_<resolution>` column to aggregate on. The grid is computed
with Sedona's native H3 functions (ST_H3CellIDs / ST_H3ToGeom), so generation
and parent assignment run distributed and scale to large study areas.
"""

import geopandas as gpd
from loguru import logger
from pyspark.sql.session import SparkSession


def h3_column(resolution: int) -> str:
    """Column name holding the H3 cell id at a given resolution."""
    return f"h3_{resolution}"


def build_h3_boundaries(
    sedona: SparkSession, census_view: str, resolution: int, geo_levels: list[str], crs: str
) -> gpd.GeoDataFrame:
    """Generate full H3 hexagons covering the study area, tagged with parent census codes.

    Every cell overlapping the study area is included, so no feature inside the
    area falls outside the grid. Each hexagon belongs to the census unit
    containing its centre (the most-overlapping unit when the centre lies just
    outside). Hexagons are kept whole (not clipped), so cell areas stay
    uniform; stats near the border may include features slightly outside the
    study area.

    Cell ids are stored as lowercase hex strings, matching the standard H3
    string representation used by the h3 Python package.
    """
    logger.info(f"Building H3 grid at resolution {resolution}")
    cols = ", ".join(geo_levels)
    p_cols = ", ".join(f"p.{c}" for c in geo_levels)
    sdf = sedona.sql(
        f"""
        WITH pairs AS (
            SELECT {cols}, geometry AS unit_geometry,
                   explode(ST_H3CellIDs(ST_Transform(geometry, '{crs}', 'EPSG:4326'), {resolution}, true)) AS cell
            FROM {census_view}
        ),
        cells AS (
            SELECT cell, ST_Transform(ST_H3ToGeom(array(cell))[0], 'EPSG:4326', '{crs}') AS geometry
            FROM (SELECT DISTINCT cell FROM pairs)
        ),
        ranked AS (
            SELECT c.cell, c.geometry, {p_cols},
                   ST_Intersects(c.geometry, p.unit_geometry) AS overlaps_unit,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.cell
                       ORDER BY ST_Contains(p.unit_geometry, ST_Centroid(c.geometry)) DESC,
                                ST_Area(ST_Intersection(c.geometry, p.unit_geometry)) DESC
                   ) AS rn
            FROM cells c JOIN pairs p ON c.cell = p.cell
        )
        SELECT LOWER(HEX(cell)) AS {h3_column(resolution)}, {cols},
               ST_Area(geometry) / 1000000 AS area, geometry
        FROM ranked WHERE rn = 1 AND overlaps_unit
        """
    )
    pdf = sdf.toPandas()
    if pdf.empty:
        raise ValueError(f"No H3 cells at resolution {resolution} overlap the study area.")
    h3_gdf = gpd.GeoDataFrame(pdf, geometry="geometry", crs=crs)
    logger.info(f"H3 grid built: {len(h3_gdf)} cells assigned to census units")
    return h3_gdf


def build_h3_buildings_overlay(
    sedona: SparkSession, buildings_view: str, h3_view: str, code_cols: list[str]
):
    """Return a building_id → hexagon/census-codes lookup as a pandas DataFrame.

    Buildings are reduced to a point on their surface so each maps to exactly
    one hexagon (mirrors the representative-point logic of the census overlay).
    """
    sel = ", ".join(f"h.{c}" for c in code_cols)
    overlay_df = sedona.sql(
        f"""
        SELECT b.building_id, {sel}
        FROM {buildings_view} b
        JOIN {h3_view} h ON ST_Contains(h.geometry, ST_PointOnSurface(b.geometry))
        """
    ).toPandas()
    return overlay_df.drop_duplicates(subset="building_id")
