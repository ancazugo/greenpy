"""Discrete Global Grid Systems for uniform statistics aggregation.

Each system builds a cell grid shaped like the census boundaries table
(geo_levels columns + geometry + area) so it can serve as the `boundaries`
Spark view, with one extra `<system>_<resolution>` column to aggregate on.

Keep this module import-cheap: system modules (and their optional third-party
libraries) are only imported by get_system().
"""

import importlib

SYSTEM_NAMES = ("h3", "s2", "geohash", "a5", "rhealpix")

# (min, max) resolution per system, duplicated here as literals so config/CLI
# validation never triggers a system-module import.
RESOLUTION_RANGES = {
    "h3": (0, 15),
    "s2": (0, 30),
    "geohash": (1, 12),
    "a5": (0, 30),
    "rhealpix": (0, 15),
}

_CLASS_NAMES = {
    "h3": "H3System",
    "s2": "S2System",
    "geohash": "GeohashSystem",
    "a5": "A5System",
    "rhealpix": "RHEALPixSystem",
}


def get_system(name: str):
    """Return the DGGS implementation for a system name (h3, s2, geohash, a5, rhealpix)."""
    if name not in SYSTEM_NAMES:
        raise ValueError(f"Unknown DGGS '{name}'; expected one of {SYSTEM_NAMES}")
    module = importlib.import_module(f"greenpy.dggs.{name}")
    return getattr(module, _CLASS_NAMES[name])()
