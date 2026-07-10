"""
Data loading and setup pipeline. All paths and column names come from GreenPyConfig.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig, is_open_buildings, is_osm, is_overture
from .dggs import get_system


def _read_vector(path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector file regardless of format (parquet, gpkg, shp, geojson, etc.)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".geoparquet"):
        return gpd.read_parquet(p)
    kwargs = {"layer": layer} if layer else {}
    return gpd.read_file(p, **kwargs)


def _rename_columns(gdf: gpd.GeoDataFrame, mapping: dict[str | None, str]) -> gpd.GeoDataFrame:
    """Rename user-supplied column names to internal canonical names.

    Renames are applied per dataset (one mapping per file) because different
    config columns may share the same source name (e.g. road_node_id and
    park_id both defaulting to 'id').
    """
    return gdf.rename(columns={k: v for k, v in mapping.items() if k and k in gdf.columns})


def setup_output_dirs(cfg: GreenPyConfig) -> dict[str, Path]:
    """Create output directories and return a dict of named paths."""
    base = Path(cfg.output.base_dir)
    dirs = {
        "base": base,
        "t3": base / "T3",
        "t30": base / "T30",
        "t30_buildings": base / "T30_buildings",
        "t300": base / "T300",
        "spectral": base / "Spectral",
        "tree_count": base / "Tree_count",
        "visibility": base / "Visibility",
        "database": base / "database",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _derive_road_nodes(edges_gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Derive road nodes from edge endpoint geometries when no nodes dataset is provided."""
    from shapely.geometry import Point

    coord_to_id: dict[tuple, int] = {}
    node_records: list[dict] = []
    start_ids: list[int] = []
    end_ids: list[int] = []

    for geom in edges_gdf.geometry:
        if geom.geom_type == "MultiLineString":
            start_pt = geom.geoms[0].coords[0]
            end_pt = geom.geoms[-1].coords[-1]
        else:
            coords = list(geom.coords)
            start_pt, end_pt = coords[0], coords[-1]
        # round to nearest decimetre to merge near-duplicate endpoints
        s = (round(start_pt[0], 1), round(start_pt[1], 1))
        e = (round(end_pt[0], 1), round(end_pt[1], 1))
        for coord in (s, e):
            if coord not in coord_to_id:
                nid = len(coord_to_id)
                coord_to_id[coord] = nid
                node_records.append({"road_node_id": nid, "geometry": Point(coord)})
        start_ids.append(coord_to_id[s])
        end_ids.append(coord_to_id[e])

    edges_gdf = edges_gdf.copy()
    edges_gdf["road_edge_start"] = start_ids
    edges_gdf["road_edge_end"] = end_ids
    nodes_gdf = gpd.GeoDataFrame(node_records, crs=edges_gdf.crs)
    return edges_gdf, nodes_gdf


def _coerce_null_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Cast fully-null columns to string so Spark 4.x can infer their types."""
    for col in gdf.columns:
        if gdf[col].isna().all():
            gdf = gdf.copy()
            gdf[col] = gdf[col].astype(str).replace("nan", None)
    return gdf


def load_tables(
    sedona: SparkSession, cfg: GreenPyConfig,
    dggs: str | None = None, dggs_resolution: int | None = None,
) -> dict:
    """
    Load all datasets from paths in cfg, rename columns to canonical names,
    register Spark temp views, and return a dict of named GeoDataFrames.

    When a DGGS is given, the `boundaries` view (and the buildings overlay)
    contains grid cells tagged with parent census codes instead of the census
    units themselves.
    """
    logger.debug("Loading tables from config paths")

    col = cfg.columns
    dirs = setup_output_dirs(cfg)
    db_dir = dirs["database"]

    buildings_parquet = db_dir / "buildings.parquet"
    parks_sites_parquet = db_dir / "parks_sites.parquet"
    parks_access_parquet = db_dir / "parks_access.parquet"
    roads_edges_parquet = db_dir / "road_edges.parquet"
    roads_nodes_parquet = db_dir / "road_nodes.parquet"
    census_parquet = db_dir / "census_boundaries.parquet"
    buildings_overlay_parquet = db_dir / "census_buildings_overlay.parquet"

    if not buildings_parquet.exists():
        _setup_parquet_files(cfg, db_dir)

    if dggs is not None:
        census_parquet, buildings_overlay_parquet = ensure_dggs_files(sedona, db_dir, cfg, dggs, dggs_resolution)

    buildings_gdf = gpd.read_parquet(buildings_parquet)
    parks_sites_gdf = gpd.read_parquet(parks_sites_parquet)
    parks_access_gdf = gpd.read_parquet(parks_access_parquet)
    road_edges_gdf = gpd.read_parquet(roads_edges_parquet)
    road_nodes_gdf = gpd.read_parquet(roads_nodes_parquet)
    census_boundaries_gdf = gpd.read_parquet(census_parquet)

    census_boundaries_sdf = sedona.createDataFrame(_coerce_null_columns(census_boundaries_gdf))
    census_boundaries_sdf.createOrReplaceTempView("boundaries")

    buildings_sdf = sedona.read.format("geoparquet").load(str(buildings_parquet))
    buildings_sdf.createOrReplaceTempView("buildings")

    if not buildings_overlay_parquet.exists():
        # older parquet caches predate the overlay lookup — build it in place
        build_buildings_overlay(db_dir, cfg)
    overlay_sdf = sedona.read.format("parquet").load(str(buildings_overlay_parquet))
    overlay_sdf.createOrReplaceTempView("boundaries_buildings_overlay")

    park_sites_filtered = _filter_parks(parks_sites_gdf, cfg)
    park_access_filtered = _filter_park_access(parks_access_gdf, park_sites_filtered, cfg)

    park_sites_sdf = sedona.createDataFrame(_coerce_null_columns(park_sites_filtered))
    park_sites_sdf.createOrReplaceTempView("public_park_sites")
    park_access_sdf = sedona.createDataFrame(_coerce_null_columns(park_access_filtered))
    park_access_sdf.createOrReplaceTempView("public_park_accesses")

    return {
        "census_boundaries_gdf": census_boundaries_gdf,
        "buildings_gdf": buildings_gdf,
        "parks_sites_gdf": park_sites_filtered,
        "parks_access_gdf": park_access_filtered,
        "road_edges_gdf": road_edges_gdf,
        "road_nodes_gdf": road_nodes_gdf,
        "output_dirs": dirs,
    }


def _setup_parquet_files(cfg: GreenPyConfig, db_dir: Path) -> None:
    """Convert raw inputs to parquet, applying canonical column renames.

    Layers whose data path is "osm" are fetched from OpenStreetMap inside the
    census-boundary extent instead of read from file. The census boundaries are
    loaded first because they define the OSM query area.
    """
    logger.info("Setting up parquet cache from raw input files")
    col = cfg.columns

    census_gdf = _read_vector(cfg.data.census_boundaries).to_crs(cfg.crs)
    census_gdf["area"] = census_gdf.geometry.area / 1_000_000
    census_gdf.to_parquet(db_dir / "census_boundaries.parquet", index=False)

    osm_layers = [s for s in ("buildings", "parks_sites", "parks_access", "roads") if is_osm(getattr(cfg.data, s))]
    if osm_layers:
        from . import osm
        logger.info(f"Layers sourced from OSM: {osm_layers} (fetched once, then cached as parquet)")
        # buffered so the road network can route to parks just outside the study area
        query_polygon = osm.build_query_polygon(census_gdf, cfg.osm.fetch_buffer)

    if is_osm(cfg.data.buildings):
        buildings_gdf = osm.fetch_osm_buildings(osm.build_query_polygon(census_gdf), cfg.osm.building_types, cfg.crs)
    elif is_overture(cfg.data.buildings):
        from . import overture
        from .osm import build_query_polygon

        buildings_gdf = overture.fetch_overture_buildings(build_query_polygon(census_gdf), cfg.crs)
    elif is_open_buildings(cfg.data.buildings):
        from . import open_buildings
        from .osm import build_query_polygon

        buildings_gdf = open_buildings.fetch_open_buildings(
            build_query_polygon(census_gdf), cfg.open_buildings.confidence_threshold, cfg.crs, cfg.gee_project
        )
    else:
        buildings_gdf = _read_vector(cfg.data.buildings, layer=col.building_layer).to_crs(cfg.crs)
        buildings_gdf = _rename_columns(
            buildings_gdf, {col.building_id: "building_id", col.building_height_col: "building_height"}
        )
        if "building_height" in buildings_gdf.columns:
            # non-numeric values become NaN and are skipped by the Visibility module
            buildings_gdf["building_height"] = pd.to_numeric(buildings_gdf["building_height"], errors="coerce")
    buildings_gdf.to_parquet(db_dir / "buildings.parquet", index=False)

    if is_osm(cfg.data.parks_sites):
        parks_sites_gdf = osm.fetch_osm_parks(query_polygon, cfg.osm.park_tags, cfg.osm.exclude_private, cfg.crs)
    else:
        parks_sites_gdf = _read_vector(cfg.data.parks_sites).to_crs(cfg.crs)
        parks_sites_gdf = parks_sites_gdf.rename(columns={col.park_id: "park_id"})
    parks_sites_gdf.to_parquet(db_dir / "parks_sites.parquet", index=False)

    if is_osm(cfg.data.roads):
        road_edges_gdf, road_nodes_gdf = osm.fetch_osm_roads(query_polygon, cfg.osm.network_type, cfg.crs)
    else:
        edge_renames = {
            col.road_edge_start: "road_edge_start",
            col.road_edge_end: "road_edge_end",
            col.road_edge_length: "road_edge_length",
        }
        node_renames = {col.road_node_id: "road_node_id"}

        road_edges_gdf = _read_vector(cfg.data.roads, layer=col.road_edge_layer).to_crs(cfg.crs)
        road_edges_gdf = _rename_columns(road_edges_gdf, edge_renames)

        roads_path = Path(cfg.data.roads)
        if cfg.data.road_nodes:
            road_nodes_gdf = _read_vector(cfg.data.road_nodes).to_crs(cfg.crs)
            road_nodes_gdf = _rename_columns(road_nodes_gdf, node_renames)
        elif roads_path.suffix.lower() not in (".parquet", ".geoparquet") and col.road_node_layer:
            road_nodes_gdf = _read_vector(cfg.data.roads, layer=col.road_node_layer).to_crs(cfg.crs)
            road_nodes_gdf = _rename_columns(road_nodes_gdf, node_renames)
        else:
            logger.info("No road nodes file — deriving nodes from edge endpoints")
            road_edges_gdf, road_nodes_gdf = _derive_road_nodes(road_edges_gdf)

    road_edges_gdf.to_parquet(db_dir / "road_edges.parquet", index=False)
    road_nodes_gdf.to_parquet(db_dir / "road_nodes.parquet", index=False)

    # Access points last: OSM derivation needs the parks and roads from above
    if is_osm(cfg.data.parks_access):
        parks_access_gdf = osm.derive_park_access_points(parks_sites_gdf, road_edges_gdf, query_polygon, cfg.crs)
    else:
        parks_access_gdf = _read_vector(cfg.data.parks_access).to_crs(cfg.crs)
        parks_access_gdf = parks_access_gdf.rename(columns={col.park_id: "park_id"})
        if col.park_access_ref_col and col.park_access_ref_col in parks_access_gdf.columns:
            parks_access_gdf = parks_access_gdf.rename(columns={col.park_access_ref_col: "park_access_ref"})
    parks_access_gdf.to_parquet(db_dir / "parks_access.parquet", index=False)

    build_buildings_overlay(db_dir, cfg)

    logger.info("Parquet cache created successfully")


def _build_overlay(buildings_gdf: gpd.GeoDataFrame, boundaries_gdf: gpd.GeoDataFrame, code_cols: list[str], out_path: Path) -> None:
    """Write a building_id → boundary-codes lookup parquet.

    Used by the Merge step as the `boundaries_buildings_overlay` view. Buildings
    are reduced to representative points so each maps to exactly one unit.
    """
    points_gdf = buildings_gdf[["building_id", "geometry"]].copy()
    points_gdf["geometry"] = points_gdf.representative_point()
    overlay_gdf = gpd.sjoin(points_gdf, boundaries_gdf[code_cols + ["geometry"]], how="inner", predicate="within")
    overlay_df = pd.DataFrame(overlay_gdf[["building_id"] + code_cols]).drop_duplicates(subset="building_id")
    overlay_df.to_parquet(out_path, index=False)


def build_buildings_overlay(db_dir: Path, cfg: GreenPyConfig) -> None:
    """Create census_buildings_overlay.parquet mapping each building to its census units."""
    logger.info("Building census/buildings overlay lookup")
    buildings_gdf = gpd.read_parquet(db_dir / "buildings.parquet")
    census_gdf = gpd.read_parquet(db_dir / "census_boundaries.parquet")
    _build_overlay(buildings_gdf, census_gdf, cfg.columns.geo_levels, db_dir / "census_buildings_overlay.parquet")


def ensure_dggs_files(
    sedona: SparkSession, db_dir: Path, cfg: GreenPyConfig, system_name: str, resolution: int
) -> tuple[Path, Path]:
    """Build (if missing) and return the DGGS boundaries and buildings-overlay parquets.

    Files are system- and resolution-suffixed so multiple grids can coexist in
    the same database directory (for h3 the names match the pre-DGGS caches,
    which stay valid).
    """
    system = get_system(system_name)
    system.validate_resolution(resolution)
    grid_parquet = db_dir / f"{system.name}_boundaries_res{resolution}.parquet"
    grid_overlay_parquet = db_dir / f"{system.name}_buildings_overlay_res{resolution}.parquet"

    if not grid_parquet.exists():
        census_gdf = gpd.read_parquet(db_dir / "census_boundaries.parquet")
        grid_gdf = system.build_boundaries(sedona, census_gdf, resolution, cfg.columns.geo_levels, cfg.crs)
        grid_gdf.to_parquet(grid_parquet, index=False)

    if not grid_overlay_parquet.exists():
        logger.info(f"Building {system.name}/buildings overlay lookup")
        sedona.read.format("geoparquet").load(str(grid_parquet)).createOrReplaceTempView("dggs_grid_src")
        sedona.read.format("geoparquet").load(str(db_dir / "buildings.parquet")) \
            .createOrReplaceTempView("dggs_buildings_src")
        code_cols = cfg.columns.geo_levels + [system.column_name(resolution)]
        overlay_df = system.build_buildings_overlay(sedona, "dggs_buildings_src", "dggs_grid_src", code_cols)
        overlay_df.to_parquet(grid_overlay_parquet, index=False)

    return grid_parquet, grid_overlay_parquet


def ensure_h3_files(sedona: SparkSession, db_dir: Path, cfg: GreenPyConfig, resolution: int) -> tuple[Path, Path]:
    """Deprecated: use ensure_dggs_files(..., "h3", resolution)."""
    return ensure_dggs_files(sedona, db_dir, cfg, "h3", resolution)


def _filter_parks(parks_sites_gdf: gpd.GeoDataFrame, cfg: GreenPyConfig) -> gpd.GeoDataFrame:
    """Filter park sites by function if configured; otherwise use all features."""
    col = cfg.columns
    if col.park_function_col and col.park_function_value:
        func_col = col.park_function_col
        func_val = col.park_function_value
        return parks_sites_gdf[parks_sites_gdf[func_col] == func_val].reset_index(drop=True)
    return parks_sites_gdf.reset_index(drop=True)


def _filter_park_access(parks_access_gdf: gpd.GeoDataFrame, parks_sites_gdf: gpd.GeoDataFrame, cfg: GreenPyConfig) -> gpd.GeoDataFrame:
    """Filter park access points to those linked to filtered park sites."""
    if "park_access_ref" in parks_access_gdf.columns:
        return parks_access_gdf[parks_access_gdf["park_access_ref"].isin(parks_sites_gdf["park_id"])].reset_index(drop=True)
    return parks_access_gdf.reset_index(drop=True)
