"""S2 quadrilateral grid, generated with Sedona's native S2 functions.

Cell ids are stored as the full 16-char lowercase hex of the uint64 id (not
the trailing-zero-trimmed S2 "token"); ids only need internal consistency.
"""

from .base import SedonaSQLDGGS


class S2System(SedonaSQLDGGS):
    name = "s2"
    min_resolution = 0
    max_resolution = 30
    resolution_term = "level"

    def cell_ids_sql(self, geom_expr: str, resolution: int) -> str:
        return f"ST_S2CellIDs({geom_expr}, {resolution})"

    def cell_id_str_sql(self, cell_expr: str) -> str:
        return f"LOWER(HEX({cell_expr}))"

    def cell_geom_sql(self, cell_expr: str) -> str:
        return f"ST_S2ToGeom(array({cell_expr}))[0]"
