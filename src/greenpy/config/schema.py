from dataclasses import dataclass, field


@dataclass
class ColumnMapping:
    # Buildings
    building_id: str
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
    tile_system: TileSystemConfig = field(default_factory=TileSystemConfig)
