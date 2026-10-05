"""Visibility: trees visible from each building's windows, with obstruction by buildings and trees.

Two engines share the definitions in observers / targets / pairs / params:
the raster engine (raster_engine, default) casts rays over a building +
vegetation surface model; the vector engine (vector_engine, Sedona SQL) uses
exact footprint and crown prisms, as a reference for small areas.
"""

from .vector_engine import process_geo_code  # noqa: F401  (Sedona engine, dispatched by cli._run_process)
