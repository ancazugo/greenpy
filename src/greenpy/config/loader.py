from pathlib import Path

import yaml
from loguru import logger

from .schema import GreenPyConfig, ColumnMapping, DataPaths, OSMConfig, OutputPaths, TileSystemConfig, is_osm

# osmnx 2.x network types
VALID_NETWORK_TYPES = {"walk", "bike", "drive", "drive_service", "all", "all_public"}


def load_config(path: str | Path) -> GreenPyConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    _require(raw, "study_area_name", "crs", "data", "columns", "output")

    data_raw = raw["data"]
    _require_section(data_raw, "data", "buildings", "parks_sites", "parks_access", "roads", "census_boundaries")

    columns_raw = raw["columns"]
    _require_section(columns_raw, "columns", "geo_levels")
    if not columns_raw.get("geo_levels"):
        raise ValueError("columns.geo_levels must be a non-empty list")
    if not is_osm(data_raw["buildings"]) and not columns_raw.get("building_id"):
        raise ValueError("columns.building_id is required when data.buildings is a file path")

    _validate_osm_sources(data_raw, columns_raw)
    osm_cfg = _parse_osm_section(raw.get("osm") or {})

    output_raw = raw["output"]
    _require_section(output_raw, "output", "base_dir")

    tile_raw = raw.get("tile_system", {})

    try:
        from pyproj import CRS
        CRS.from_user_input(raw["crs"])
    except Exception as e:
        raise ValueError(f"Invalid CRS '{raw['crs']}': {e}")

    h3_resolution = raw.get("h3_resolution")
    if h3_resolution is not None and not (isinstance(h3_resolution, int) and 0 <= h3_resolution <= 15):
        raise ValueError(f"h3_resolution must be an integer between 0 and 15, got {h3_resolution!r}")

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
            canopy_height_ee_path=data_raw.get("canopy_height_ee_path"),
        ),
        columns=ColumnMapping(
            building_id=columns_raw.get("building_id"),
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
        h3_resolution=h3_resolution,
        tile_system=TileSystemConfig(
            enabled=tile_raw.get("enabled", False),
            tile_name_pattern=tile_raw.get("tile_name_pattern"),
        ),
        osm=osm_cfg,
    )


def _validate_osm_sources(data_raw: dict, columns_raw: dict) -> None:
    """Check OSM-sourced layers are consistent with the rest of the config."""
    if is_osm(data_raw["census_boundaries"]):
        raise ValueError("data.census_boundaries cannot be 'osm' — it defines the study area and must be a file")
    if is_osm(data_raw["roads"]) and data_raw.get("road_nodes"):
        raise ValueError("data.road_nodes cannot be combined with roads: osm — nodes come from the OSM network")
    if is_osm(data_raw["parks_sites"]):
        if columns_raw.get("park_function_col") or columns_raw.get("park_function_value"):
            raise ValueError(
                "columns.park_function_col/value cannot be combined with parks_sites: osm — "
                "OSM parks are already filtered by tags (see osm.park_tags)"
            )
        if not is_osm(data_raw["parks_access"]) and columns_raw.get("park_access_ref_col"):
            logger.warning(
                "parks_sites is 'osm' but parks_access is a file with park_access_ref_col set — "
                "its references will not match OSM park ids, so all access points may be filtered out"
            )


def _parse_osm_section(osm_raw: dict) -> OSMConfig:
    network_type = osm_raw.get("network_type", "walk")
    if network_type not in VALID_NETWORK_TYPES:
        raise ValueError(f"osm.network_type must be one of {sorted(VALID_NETWORK_TYPES)}, got {network_type!r}")

    building_types = osm_raw.get("building_types")
    if building_types is not None and not (isinstance(building_types, list) and all(isinstance(t, str) for t in building_types)):
        raise ValueError("osm.building_types must be a list of strings (or 'all' as a list item)")

    park_tags = osm_raw.get("park_tags")
    if park_tags is not None and not isinstance(park_tags, dict):
        raise ValueError("osm.park_tags must be a mapping of OSM key -> list of values")

    fetch_buffer = osm_raw.get("fetch_buffer", 2000)
    if not isinstance(fetch_buffer, int) or fetch_buffer < 0:
        raise ValueError(f"osm.fetch_buffer must be a non-negative integer (metres), got {fetch_buffer!r}")

    return OSMConfig(
        network_type=network_type,
        building_types=building_types,
        park_tags=park_tags,
        exclude_private=osm_raw.get("exclude_private", True),
        fetch_buffer=fetch_buffer,
    )


def _require(d: dict, *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Config is missing required fields: {missing}")


def _require_section(d: dict, section: str, *keys: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise ValueError(f"Config section '{section}' is missing required fields: {missing}")
