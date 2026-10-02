from dataclasses import dataclass, field


# Remote sources accepted by data.buildings instead of a file path
BUILDING_SOURCE_SENTINELS = ("osm", "overture", "open_buildings")


def _sentinel(source: str | None) -> str | None:
    return source.strip().lower() if isinstance(source, str) else None


def is_osm(source: str | None) -> bool:
    """True when a config data path requests the OSM source instead of a file."""
    return _sentinel(source) == "osm"


def is_overture(source: str | None) -> bool:
    """True when data.buildings requests Overture Maps building footprints."""
    return _sentinel(source) == "overture"


def is_open_buildings(source: str | None) -> bool:
    """True when data.buildings requests Google Open Buildings v3 polygons (via GEE)."""
    return _sentinel(source) == "open_buildings"


def building_source(source: str | None) -> str | None:
    """Sentinel name when data.buildings names a remote source, else None (file path)."""
    s = _sentinel(source)
    return s if s in BUILDING_SOURCE_SENTINELS else None


@dataclass
class ColumnMapping:
    # Buildings (building_id only required when buildings come from a file)
    building_id: str | None = None
    building_layer: str | None = None
    # Building height in metres — required only by the Visibility module
    building_height_col: str | None = None
    # Keep only buildings whose building_use_col matches building_use_value (a
    # single value or a list). File-based sources only; OSM uses osm.building_types.
    building_use_col: str | None = None
    building_use_value: str | list[str] | None = None

    # Roads
    road_node_id: str = "id"
    road_edge_start: str = "start_node"
    road_edge_end: str = "end_node"
    road_edge_length: str = "length"
    road_edge_layer: str = "road_link"
    road_node_layer: str = "road_node"

    # Parks
    park_id: str = "id"
    park_function_col: str | None = None
    park_function_value: str | None = None
    park_access_ref_col: str | None = None

    # Trees
    tree_height_col: str = "height"
    tree_area_col: str = "area"
    tree_id_col: str = "treeID"
    tree_layer: str = "trees"

    # Census geographies — ordered coarsest → finest
    # e.g. ["RGN22CD", "LAD22CD", "LSOA21CD", "OA21CD"]
    geo_levels: list[str] = field(default_factory=list)


@dataclass
class DataPaths:
    buildings: str
    parks_sites: str
    parks_access: str
    roads: str
    census_boundaries: str
    road_nodes: str | None = None
    trees_dir: str | None = None
    chm_tiles_dir: str | None = None
    # Glob for CHM tiles under chm_tiles_dir (searched recursively), used by T30,
    # T30_buildings and Trees. Defra VOM hillshades (VOM_HS_*) are always skipped.
    chm_pattern: str = "*.tif"
    # Where CHM tiles overlap (e.g. survey years) — T30, T30_buildings and Trees:
    # "latest" = last path in sorted order wins (the latest year for <dir>/<year>/
    # layouts); "max" = per-pixel maximum across them (a tree seen in any survey counts)
    chm_overlap: str = "latest"
    # Downloaded CHM tiles, raster indexes, VRTs and overlap composites;
    # None = $GREENPY_CACHE_DIR, else <output.base_dir>/database/chm_cache
    chm_cache_dir: str | None = None
    # GEE canopy-height asset (e.g. projects/sat-io/open-datasets/facebook/meta-canopy-height)
    canopy_height_ee_path: str | None = None


@dataclass
class OSMConfig:
    """Options for layers sourced from OSM (data paths set to "osm")."""

    # osmnx network type for roads (walkable by default)
    network_type: str = "walk"
    # building=* tag values to fetch; None = residential defaults, ["all"] = every building
    building_types: list[str] | None = None
    # OSM tags for parks, e.g. {leisure: [park], landuse: [village_green]}; None = public-park defaults
    park_tags: dict[str, list[str]] | None = None
    # Drop parks tagged access=private/no/customers
    exclude_private: bool = True
    # Buffer (m) around the census boundary when fetching roads/parks/access,
    # so the network can route to parks just outside the study area
    fetch_buffer: int = 2000


@dataclass
class OpenBuildingsConfig:
    """Options for buildings sourced from Google Open Buildings v3 (data.buildings: open_buildings)."""

    # Min detection confidence to keep (dataset values roughly in [0.5, 1))
    confidence_threshold: float = 0.7


@dataclass
class OutputPaths:
    base_dir: str


@dataclass
class TileSystemConfig:
    """Legacy UK VOM tile settings, parsed so older configs still load.

    Tree and CHM tile directories are always searched by file extent, so these
    no longer change which tiles are read.
    """

    enabled: bool = False
    tile_name_pattern: str | None = None


@dataclass
class TreeSegmentationConfig:
    """Options for the Trees process, which segments tree crowns from a CHM into data.trees_dir."""

    # "chm_tiles" (data.chm_tiles_dir) or "meta" (Meta/WRI global 1 m CHM from AWS);
    # None uses chm_tiles when data.chm_tiles_dir is set, else meta
    source: str | None = None
    # SegmentationParams preset (legacy_vom = the lidR chm_processing.R settings) and per-field overrides
    preset: str = "legacy_vom"
    params: dict = field(default_factory=dict)
    # Block side in pixels; each block is processed with a halo, in parallel with --parallel --n_workers
    block_size: int = 2048
    # "polygon" crowns, or "point" crown centroids (faster; T3/Tree_count only use centroids)
    geometry: str = "polygon"


@dataclass
class GreenPyConfig:
    study_area_name: str
    crs: str
    data: DataPaths
    columns: ColumnMapping
    output: OutputPaths
    gee_project: str | None = None
    gee_boundaries_asset: str | None = None
    # Drop parks smaller than this (hectares) before T300; None keeps all.
    # The WHO guideline behind the 300 rule uses green spaces of >= 0.5-1 ha.
    park_min_area_ha: float | None = None
    # Aggregate to DGGS cells instead of the finest geo level
    dggs: str | None = None  # h3, s2, geohash, a5 or rhealpix
    dggs_resolution: int | None = None
    # Deprecated: use dggs: h3 + dggs_resolution (loader normalizes it there)
    h3_resolution: int | None = None
    tile_system: TileSystemConfig = field(default_factory=TileSystemConfig)
    # Options for layers with their data path set to "osm"
    osm: OSMConfig = field(default_factory=OSMConfig)
    # Options for buildings set to "open_buildings"
    open_buildings: OpenBuildingsConfig = field(default_factory=OpenBuildingsConfig)
    # Options for the Trees process
    tree_segmentation: TreeSegmentationConfig = field(default_factory=TreeSegmentationConfig)
