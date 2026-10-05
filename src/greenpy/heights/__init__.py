"""Building heights from an ordered chain of sources (the Heights process).

Each source attaches a height to the building footprints of data.buildings:
attributes already in the footprints (`native`), vector datasets matched by
footprint overlap (`gba`, `utglobus`, vector `file`) or raster datasets
summarised per footprint (`open_buildings_temporal`, `ghs_built_h`, raster
`file`). enrich.build_building_heights coalesces them in config order.

Keep this module import-cheap: source modules (and GEE) are only imported by
get_source(), so config validation can use the literals below.
"""

import importlib

SOURCE_NAMES = ("native", "gba", "utglobus", "open_buildings_temporal", "ghs_built_h", "file")

# Options each source accepts in heights.sources (besides `source` and `name`),
# and the ones it requires — literals so validation never imports a source module
SOURCE_OPTIONS = {
    "native": {"levels_col"},
    "gba": set(),
    "utglobus": {"city"},
    "open_buildings_temporal": {"year", "presence_threshold", "resolution"},
    "ghs_built_h": {"resolution"},
    "file": {"path", "column", "layer", "stat"},
}
REQUIRED_OPTIONS = {"utglobus": {"city"}, "file": {"path"}}

_MODULES = {
    "native": ("native", "NativeHeights"),
    "gba": ("gba", "GBAHeights"),
    "utglobus": ("utglobus", "UTGlobusHeights"),
    "open_buildings_temporal": ("open_buildings_temporal", "OpenBuildingsTemporalHeights"),
    "ghs_built_h": ("ghs", "GHSBuiltHeights"),
    "file": ("file", "FileHeights"),
}


def get_source(spec, cfg):
    """Instantiate the height source for a heights.sources entry (HeightSourceSpec)."""
    if spec.source not in SOURCE_NAMES:
        raise ValueError(f"Unknown height source '{spec.source}'; expected one of {SOURCE_NAMES}")
    module_name, class_name = _MODULES[spec.source]
    module = importlib.import_module(f"greenpy.heights.{module_name}")
    return getattr(module, class_name)(spec, cfg)
