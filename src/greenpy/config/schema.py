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
    # Building height in metres (file sources) — read by the "native" height source
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
    # Optional display names for the levels (e.g. {ADM3_code: Ward}) and, per level,
    # a census_boundaries column holding each unit's name (e.g. {ADM3_code: ADM3_name}) — used by the viz
    geo_level_labels: dict[str, str] = field(default_factory=dict)
    geo_level_names: dict[str, str] = field(default_factory=dict)


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
class OvertureConfig:
    """Options for layers sourced from Overture Maps (buildings or parks_sites set to "overture")."""

    # land_use subtypes or classes kept as parks (an entry matches either). Protected
    # areas (national parks, reserves) are left out by default: they are usually
    # fenced or ticketed rather than open neighbourhood green space
    park_land_use: list[str] = field(default_factory=lambda: ["park", "recreation_ground"])


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
class HeightSourceSpec:
    """One entry of heights.sources: a source name (see greenpy.heights.SOURCE_NAMES) and its options."""

    source: str
    options: dict = field(default_factory=dict)
    # label written to height_source (default: the source name), e.g. "lidar" for a local nDSM
    name: str | None = None

    @property
    def label(self) -> str:
        return self.name or self.source


@dataclass
class HeightsConfig:
    """Options for the Heights process, which attaches a height to every building footprint."""

    # Ordered sources; for each building the first valid height wins
    sources: list[HeightSourceSpec] = field(default_factory=lambda: [HeightSourceSpec("native")])
    # Height given to buildings no source covers (height_source = "default")
    default_height: float = 6.0
    # Metres per storey: converts floor counts to heights and spaces Visibility's observer floors
    storey_height: float = 3.0
    # Vector sources: min share of a footprint's area a source polygon must cover to lend it its height
    min_overlap: float = 0.3
    # Heights outside [min_height, max_height] are treated as missing (falls through the chain)
    min_height: float = 2.0
    max_height: float = 300.0


@dataclass
class VisibilityConfig:
    """Options for the Visibility process (line of sight from building windows to trees)."""

    # "raster": DSM ray casting (default, scales); "vector": exact Sedona geometry (reference, small areas)
    engine: str = "raster"
    # Observer windows: a point every facade_spacing metres around each footprint, facade_offset
    # metres outside the wall, at eye_height above each floor (floors from heights.storey_height)
    facade_spacing: float = 5.0
    facade_offset: float = 0.5
    eye_height: float = 1.5
    # Targets: the treetop plus crown_points points on the crown at crown_point_height x tree height
    crown_points: int = 4
    crown_point_height: float = 2 / 3
    # Raster engine: DSM pixel size (m), vegetation surface ("auto": CHM when one is configured,
    # else rasterised crowns; "chm"; "crowns"), and whether CHM pixels on roofs are dropped
    resolution: float = 1.0
    vegetation: str = "auto"
    mask_chm_buildings: bool = True
    # Metres ignored at both ends of each sightline (None = resolution)
    end_skip: float | None = None
    # Raster engine: buildings are processed in square tiles of this side (m)
    tile_size: float = 2000.0


@dataclass
class TerrainConfig:
    """Ground elevation under buildings and trees for Visibility (raster engine)."""

    # "fabdem" (bare-earth Copernicus, buildings and forests removed; CC BY-NC-SA 4.0),
    # "copernicus" (GLO-30) or "nasadem" (both surface models: in cities they include
    # rooftops and canopy), a local DTM raster file/directory, or null for flat ground
    source: str | None = "fabdem"
    # Download resolution (m) for the GEE sources (native ~30 m)
    resolution: float = 30.0


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
    # Options for layers set to "overture"
    overture: OvertureConfig = field(default_factory=OvertureConfig)
    # Options for the Trees process
    tree_segmentation: TreeSegmentationConfig = field(default_factory=TreeSegmentationConfig)
    # Options for the Heights process (building heights) and the Visibility process
    heights: HeightsConfig = field(default_factory=HeightsConfig)
    visibility: VisibilityConfig = field(default_factory=VisibilityConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    # Metres around the study area from which context is gathered: buildings that block
    # views (Visibility) and trees that count for edge buildings (Trees -> T3, Visibility)
    context_buffer: float = 100.0
