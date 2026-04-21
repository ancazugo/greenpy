from pathlib import Path

import yaml

from .schema import GreenPyConfig, ColumnMapping, DataPaths, OutputPaths, TileSystemConfig


def load_config(path: str | Path) -> GreenPyConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    _require(raw, "study_area_name", "crs", "data", "columns", "output")

    data_raw = raw["data"]
    _require_section(data_raw, "data", "buildings", "parks_sites", "parks_access", "roads", "census_boundaries")

    columns_raw = raw["columns"]
    _require_section(columns_raw, "columns", "building_id", "geo_levels")
    if not columns_raw.get("geo_levels"):
        raise ValueError("columns.geo_levels must be a non-empty list")

    output_raw = raw["output"]
    _require_section(output_raw, "output", "base_dir")

    tile_raw = raw.get("tile_system", {})

    try:
        from pyproj import CRS
        CRS.from_user_input(raw["crs"])
    except Exception as e:
        raise ValueError(f"Invalid CRS '{raw['crs']}': {e}")

    return GreenPyConfig(
        study_area_name=raw["study_area_name"],
        crs=raw["crs"],
        data=DataPaths(
            buildings=data_raw["buildings"],
            parks_sites=data_raw["parks_sites"],
            parks_access=data_raw["parks_access"],
            roads=data_raw["roads"],
            census_boundaries=data_raw["census_boundaries"],
            road_nodes=data_raw.get("road_nodes"),
            trees_dir=data_raw.get("trees_dir"),
            chm_tiles_dir=data_raw.get("chm_tiles_dir"),
        ),
        columns=ColumnMapping(
            building_id=columns_raw["building_id"],
            building_layer=columns_raw.get("building_layer"),
            road_node_id=columns_raw.get("road_node_id", "id"),
            road_edge_start=columns_raw.get("road_edge_start", "start_node"),
            road_edge_end=columns_raw.get("road_edge_end", "end_node"),
            road_edge_length=columns_raw.get("road_edge_length", "length"),
            road_edge_layer=columns_raw.get("road_edge_layer", "road_link"),
            road_node_layer=columns_raw.get("road_node_layer", "road_node"),
            park_id=columns_raw.get("park_id", "id"),
            park_function_col=columns_raw.get("park_function_col"),
            park_function_value=columns_raw.get("park_function_value"),
            park_access_ref_col=columns_raw.get("park_access_ref_col"),
            tree_height_col=columns_raw.get("tree_height_col", "height"),
            tree_area_col=columns_raw.get("tree_area_col", "area"),
            tree_id_col=columns_raw.get("tree_id_col", "treeID"),
            tree_layer=columns_raw.get("tree_layer", "trees"),
            geo_levels=columns_raw["geo_levels"],
        ),
        output=OutputPaths(base_dir=output_raw["base_dir"]),
        gee_project=raw.get("gee_project"),
        gee_boundaries_asset=raw.get("gee_boundaries_asset"),
        tile_system=TileSystemConfig(
            enabled=tile_raw.get("enabled", False),
            tile_name_pattern=tile_raw.get("tile_name_pattern"),
        ),
    )


def _require(d: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Config is missing required fields: {missing}")


def _require_section(d: dict, section: str, *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Config section '{section}' is missing required fields: {missing}")
