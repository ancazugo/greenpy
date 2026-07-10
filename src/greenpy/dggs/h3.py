"""H3 hexagonal grid, generated with Sedona's native H3 functions.

Cell ids are stored as lowercase hex strings, matching the standard H3
string representation used by the h3 Python package.
"""

from .base import SedonaSQLDGGS


class H3System(SedonaSQLDGGS):
    name = "h3"
    min_resolution = 0
    max_resolution = 15

    def cell_ids_sql(self, geom_expr: str, resolution: int) -> str:
        return f"ST_H3CellIDs({geom_expr}, {resolution}, true)"

    def cell_id_str_sql(self, cell_expr: str) -> str:
        return f"LOWER(HEX({cell_expr}))"

    def cell_geom_sql(self, cell_expr: str) -> str:
        return f"ST_H3ToGeom(array({cell_expr}))[0]"
