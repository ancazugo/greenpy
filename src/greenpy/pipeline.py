"""
Data loading and setup pipeline. All paths and column names come from GreenPyConfig.
"""

from pathlib import Path

import pandas as pd
import geopandas as gpd
from loguru import logger
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig


def _read_vector(path: str, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a vector file regardless of format (parquet, gpkg, shp, geojson, etc.)."""
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".geoparquet"):
        return gpd.read_parquet(p)
    kwargs = {"layer": layer} if layer else {}
    return gpd.read_file(p, **kwargs)


def _rename_to_canonical(gdf: gpd.GeoDataFrame, cfg: GreenPyConfig) -> gpd.GeoDataFrame:
    """Rename user-supplied column names to internal canonical names."""
    col = cfg.columns
    rename = {
        col.building_id: "building_id",
        col.road_node_id: "road_node_id",
        col.road_edge_start: "road_edge_start",
        col.road_edge_end: "road_edge_end",
        col.road_edge_length: "road_edge_length",
        col.park_id: "park_id",
        col.tree_height_col: "tree_height",
        col.tree_area_col: "tree_area",
        col.tree_id_col: "tree_id",
    }
    if col.park_access_ref_col:
        rename[col.park_access_ref_col] = "park_access_ref"
    return gdf.rename(columns={k: v for k, v in rename.items() if k in gdf.columns})


def setup_output_dirs(cfg: GreenPyConfig) -> dict[str, Path]:
    """Create output directories and return a dict of named paths."""
    base = Path(cfg.output.base_dir)
    dirs = {
        "base": base,
        "t3": base / "T3",
        "t30": base / "T30",
        "t300": base / "T300",
        "spectral": base / "Spectral",
        "tree_count": base / "Tree_count",
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
        coords = list(geom.coords)
        # round to nearest decimetre to merge near-duplicate endpoints
        s = (round(coords[0][0], 1), round(coords[0][1], 1))
        e = (round(coords[-1][0], 1), round(coords[-1][1], 1))
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


def load_tables(sedona: SparkSession, cfg: GreenPyConfig) -> dict:
    """
    Load all datasets from paths in cfg, rename columns to canonical names,
    register Spark temp views, and return a dict of named GeoDataFrames.
    """
    logger.debug("Loading tables from config paths")

    col = cfg.columns
    db_dir = Path(cfg.output.base_dir) / "database"

    buildings_parquet = db_dir / "buildings.parquet"
    parks_sites_parquet = db_dir / "parks_sites.parquet"
    parks_access_parquet = db_dir / "parks_access.parquet"
    roads_edges_parquet = db_dir / "road_edges.parquet"
    roads_nodes_parquet = db_dir / "road_nodes.parquet"
    census_parquet = db_dir / "census_boundaries.parquet"
    buildings_overlay_parquet = db_dir / "census_buildings_overlay.parquet"

    if not buildings_parquet.exists():
        _setup_parquet_files(cfg, db_dir)

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

    if buildings_overlay_parquet.exists():
        overlay_sdf = sedona.read.format("parquet").load(str(buildings_overlay_parquet))
        overlay_sdf.createOrReplaceTempView("boundaries_buildings_overlay")

    park_sites_filtered = _filter_parks(parks_sites_gdf, cfg)
    park_access_filtered = _filter_park_access(parks_access_gdf, park_sites_filtered, cfg)

    park_sites_sdf = sedona.createDataFrame(_coerce_null_columns(park_sites_filtered))
    park_sites_sdf.createOrReplaceTempView("public_park_sites")
    park_access_sdf = sedona.createDataFrame(_coerce_null_columns(park_access_filtered))
    park_access_sdf.createOrReplaceTempView("public_park_accesses")

    dirs = setup_output_dirs(cfg)

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
    """Convert raw input files to parquet, applying canonical column renames."""
    logger.info("Setting up parquet cache from raw input files")
    col = cfg.columns

    buildings_gdf = _read_vector(cfg.data.buildings, layer=col.building_layer).to_crs(cfg.crs)
    buildings_gdf = _rename_to_canonical(buildings_gdf, cfg)
    buildings_gdf.to_parquet(db_dir / "buildings.parquet", index=False)

    parks_sites_gdf = _read_vector(cfg.data.parks_sites).to_crs(cfg.crs)
    parks_sites_gdf = parks_sites_gdf.rename(columns={col.park_id: "park_id"})
    parks_sites_gdf.to_parquet(db_dir / "parks_sites.parquet", index=False)

    parks_access_gdf = _read_vector(cfg.data.parks_access).to_crs(cfg.crs)
    parks_access_gdf = parks_access_gdf.rename(columns={col.park_id: "park_id"})
    if col.park_access_ref_col and col.park_access_ref_col in parks_access_gdf.columns:
        parks_access_gdf = parks_access_gdf.rename(columns={col.park_access_ref_col: "park_access_ref"})
    parks_access_gdf.to_parquet(db_dir / "parks_access.parquet", index=False)

    road_edges_gdf = _read_vector(cfg.data.roads, layer=col.road_edge_layer).to_crs(cfg.crs)
    road_edges_gdf = _rename_to_canonical(road_edges_gdf, cfg)

    roads_path = Path(cfg.data.roads)
    if cfg.data.road_nodes:
        road_nodes_gdf = _read_vector(cfg.data.road_nodes).to_crs(cfg.crs)
        road_nodes_gdf = _rename_to_canonical(road_nodes_gdf, cfg)
    elif roads_path.suffix.lower() not in (".parquet", ".geoparquet") and col.road_node_layer:
        road_nodes_gdf = _read_vector(cfg.data.roads, layer=col.road_node_layer).to_crs(cfg.crs)
        road_nodes_gdf = _rename_to_canonical(road_nodes_gdf, cfg)
    else:
        logger.info("No road nodes file — deriving nodes from edge endpoints")
        road_edges_gdf, road_nodes_gdf = _derive_road_nodes(road_edges_gdf)

    road_edges_gdf.to_parquet(db_dir / "road_edges.parquet", index=False)
    road_nodes_gdf.to_parquet(db_dir / "road_nodes.parquet", index=False)

    census_gdf = _read_vector(cfg.data.census_boundaries).to_crs(cfg.crs)
    census_gdf["area"] = census_gdf.geometry.area / 1_000_000
    census_gdf.to_parquet(db_dir / "census_boundaries.parquet", index=False)

    logger.info("Parquet cache created successfully")


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
