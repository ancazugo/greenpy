"""rHEALPix equal-area grid, generated driver-side with the rhealpixdggs library.

Uses the predefined WGS84 ellipsoid instance with N_side=3. Cell ids are the
standard string form (e.g. "N24785"). cells_from_region covers the rectangle
between the *projected* corners, which can miss cells at the lon/lat corners,
so the bounding box is first padded by one cell's angular extent; the shared
finalizer trims cells not touching the unit.
"""

from typing import Iterable

from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from .base import PythonDGGS


class RHEALPixSystem(PythonDGGS):
    name = "rhealpix"
    min_resolution = 0
    max_resolution = 15

    def cells_covering(
        self, geom_wgs84: BaseGeometry, resolution: int
    ) -> Iterable[tuple[str, Polygon]]:
        from rhealpixdggs.dggs import WGS84_003

        minx, miny, maxx, maxy = geom_wgs84.bounds
        centre = geom_wgs84.representative_point()
        seed = WGS84_003.cell_from_point(resolution, (centre.x, centre.y), plane=False)
        seed_poly = Polygon(seed.vertices(plane=False, trim_dart=True))
        sminx, sminy, smaxx, smaxy = seed_poly.bounds
        pad_x, pad_y = smaxx - sminx, smaxy - sminy
        cell_rows = WGS84_003.cells_from_region(
            resolution,
            ul=(max(minx - pad_x, -180.0), min(maxy + pad_y, 90.0)),
            dr=(min(maxx + pad_x, 180.0), max(miny - pad_y, -90.0)),
            plane=False,
        )
        for row in cell_rows:
            for cell in row:
                yield str(cell), Polygon(cell.vertices(plane=False, trim_dart=True))
