"""Geohash rectangular grid, enumerated driver-side.

Sedona's ST_GeoHash hashes a single point and cannot polyfill a polygon, but
geohash cells at a given precision form a fixed global lon/lat lattice, so the
cells covering a census unit's bounding box can be enumerated exactly: cell
polygons come straight from the lattice and ids from a small pure-Python
encoder (no external dependency). Cells not touching the unit are dropped by
the shared finalizer, giving the same whole-area coverage semantics as H3.
"""

import math
from typing import Iterable

from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from .base import PythonDGGS

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_encode(lat: float, lon: float, precision: int) -> str:
    """Standard geohash of a point (interleaved lon/lat bisection, base32)."""
    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    chars = []
    bits = 0
    n_bits = 0
    even = True  # even bit positions encode longitude
    while len(chars) < precision:
        if even:
            mid = (lon_lo + lon_hi) / 2
            if lon >= mid:
                bits = (bits << 1) | 1
                lon_lo = mid
            else:
                bits <<= 1
                lon_hi = mid
        else:
            mid = (lat_lo + lat_hi) / 2
            if lat >= mid:
                bits = (bits << 1) | 1
                lat_lo = mid
            else:
                bits <<= 1
                lat_hi = mid
        even = not even
        n_bits += 1
        if n_bits == 5:
            chars.append(_BASE32[bits])
            bits = n_bits = 0
    return "".join(chars)


def cell_size(precision: int) -> tuple[float, float]:
    """(width, height) in degrees of a geohash cell at a given precision."""
    lon_bits = math.ceil(5 * precision / 2)
    lat_bits = 5 * precision // 2
    return 360.0 / (1 << lon_bits), 180.0 / (1 << lat_bits)


class GeohashSystem(PythonDGGS):
    name = "geohash"
    min_resolution = 1
    max_resolution = 12
    resolution_term = "precision"

    def cells_covering(
        self, geom_wgs84: BaseGeometry, resolution: int
    ) -> Iterable[tuple[str, Polygon]]:
        w, h = cell_size(resolution)
        minx, miny, maxx, maxy = geom_wgs84.bounds
        i0 = max(int((minx + 180.0) // w), 0)
        i1 = min(int((maxx + 180.0) // w), (1 << math.ceil(5 * resolution / 2)) - 1)
        j0 = max(int((miny + 90.0) // h), 0)
        j1 = min(int((maxy + 90.0) // h), (1 << (5 * resolution // 2)) - 1)
        for i in range(i0, i1 + 1):
            x0 = -180.0 + i * w
            for j in range(j0, j1 + 1):
                y0 = -90.0 + j * h
                yield geohash_encode(y0 + h / 2, x0 + w / 2, resolution), box(x0, y0, x0 + w, y0 + h)
