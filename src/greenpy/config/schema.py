from dataclasses import dataclass, field


def is_osm(source: str | None) -> bool:
    """True when a config data path requests the OSM source instead of a file."""
    return isinstance(source, str) and source.strip().lower() == "osm"


@dataclass
class ColumnMapping:
    # Buildings (building_id only required when buildings come from a file)
    building_id: str | None = None
    building_layer: str | None = None

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
class OutputPaths:
    base_dir: str


@dataclass
class TileSystemConfig:
    enabled: bool = False
    tile_name_pattern: str | None = None


@dataclass
class GreenPyConfig:
    study_area_name: str
    crs: str
    data: DataPaths
    columns: ColumnMapping
    output: OutputPaths
    gee_project: str | None = None
    gee_boundaries_asset: str | None = None
    # Aggregate to H3 hexagons at this resolution instead of the finest geo level
    h3_resolution: int | None = None
    tile_system: TileSystemConfig = field(default_factory=TileSystemConfig)
    # Options for layers with their data path set to "osm"
    osm: OSMConfig = field(default_factory=OSMConfig)
