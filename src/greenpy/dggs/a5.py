"""A5 pentagonal equal-area grid, generated driver-side with the pya5 library.

Cell ids are stored as the 16-char hex of the uint64 id (a5.u64_to_hex).
a5.polygon_to_cells does the polyfill but is not a full cover (cells only
partially overlapping the polygon can be missed), so the result is expanded by
one neighbour ring with a5.grid_disk (matching H3's full-cover semantics) and
the shared finalizer trims the excess.
"""

from typing import Iterable

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .base import PythonDGGS


class A5System(PythonDGGS):
    name = "a5"
    min_resolution = 0
    max_resolution = 30

    def cells_covering(
        self, geom_wgs84: BaseGeometry, resolution: int
    ) -> Iterable[tuple[str, Polygon]]:
        import a5

        parts = geom_wgs84.geoms if geom_wgs84.geom_type == "MultiPolygon" else [geom_wgs84]
        base = set()
        for part in parts:
            rings = [list(part.exterior.coords)] + [list(r.coords) for r in part.interiors]
            base.update(a5.polygon_to_cells(rings, resolution))
        if not base:
            # geometry smaller than one cell — seed from its representative point
            point = geom_wgs84.representative_point()
            base.add(a5.lonlat_to_cell((point.x, point.y), resolution))
        cells = set()
        for cell in base:
            cells.update(a5.grid_disk(cell, 1))
        for cell in cells:
            yield a5.u64_to_hex(cell), Polygon(a5.cell_to_boundary(cell))
