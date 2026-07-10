#!/usr/bin/env python3
"""
greenpy CLI — measure the 3-30-300 rule of urban greening.
"""

import inspect
import concurrent.futures
from pathlib import Path
from typing import Optional

import typer
from loguru import logger
from tqdm import tqdm

from .config.loader import load_config
from .pipeline import load_tables, setup_output_dirs
from .utils.h3_boundaries import h3_column
from .utils.logging_config import setup_logger
from .utils.sedona_config import get_spark
from . import t3 as t3_module
from . import t30 as t30_module
from . import t30_buildings as t30_buildings_module
from . import t300 as t300_module
from . import tree_count as tree_count_module
from .optional import visibility as visibility_module

app = typer.Typer(help="greenpy: 3-30-300 urban greening metrics")


@app.callback()
def _callback():
    """greenpy: 3-30-300 urban greening metrics."""
    # Forces typer to keep `run` as a named subcommand even while it is the only one.


def _run_process(process: str, args_dict: dict, geo_code: str) -> object:
    """Dispatch one geo_code to the module's process_geo_code, passing only the kwargs it accepts."""
    process_fns = {
        "T3": t3_module.process_geo_code,
        "T30": t30_module.process_geo_code,
        "T30_buildings": t30_buildings_module.process_geo_code,
        "T300": t300_module.process_geo_code,
        "Tree_count": tree_count_module.process_geo_code,
        "Visibility": visibility_module.process_geo_code,
    }
    fn = process_fns[process]
    valid_params = set(inspect.signature(fn).parameters.keys())
    filtered = {k: v for k, v in args_dict.items() if k in valid_params}
    filtered["geo_code"] = geo_code
    return fn(**filtered)


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to study-area YAML config file"),
    process: str = typer.Option(..., "--process", "-p", help="Module to run: T3, T30, T30_buildings, T300, Tree_count, Visibility, Spectral, Merge"),
    geo_level: str = typer.Option(None, "--geo_level", help="Geography column to process (must be in config.columns.geo_levels)"),
    sub_geo_level: str = typer.Option(None, "--sub_geo_level", help="Sub-geography column (for Tree_count and Merge)"),
    h3_resolution: Optional[int] = typer.Option(None, "--h3_resolution", help="Aggregate to H3 hexagons at this resolution (0-15) instead of the finest census level; overrides config"),
    geo_code: Optional[str] = typer.Option(None, "--geo_code", help="Single geography code to process; omit to process all"),
    query_method: str = typer.Option("rdd", "--query_method", help="Sedona query method: sql or rdd"),
    buffer: int = typer.Option(100, "--buffer", help="Buffer radius in metres around each building (T3, T30_buildings; 0 = footprint only)"),
    tree_area: int = typer.Option(10, "--tree_area", help="Minimum tree canopy area (m²)"),
    tree_height: int = typer.Option(3, "--tree_height", help="Minimum tree height (m)"),
    observer_mode: str = typer.Option("facade", "--observer_mode", help="Visibility observer point: facade (nearest footprint boundary point to the tree) or centroid"),
    low_threshold: int = typer.Option(3, "--low_threshold", help="Min canopy height in metres for T30 binarisation"),
    high_threshold: int = typer.Option(60, "--high_threshold", help="Max canopy height in metres for T30 binarisation"),
    gee_scale: float = typer.Option(1.0, "--gee_scale", help="Download scale in metres for the T30 GEE canopy source"),
    start_date: str = typer.Option("2024-01-01", "--start_date", help="GEE imagery start date"),
    end_date: str = typer.Option("2024-12-31", "--end_date", help="GEE imagery end date"),
    imagery_ee_path: str = typer.Option("COPERNICUS/S2_HARMONIZED", "--imagery_ee_path", help="GEE imagery collection path"),
    cloud_coverage: float = typer.Option(10.0, "--cloud_coverage", help="Max cloud coverage percentage"),
    spectral_indexes: list[str] = typer.Option(["NDVI", "NDWI", "NDBI"], "--spectral_indexes", help="Spectral indices to compute"),
    t3_buffers: list[int] = typer.Option([10, 25, 50, 75, 100], "--t3_buffers", help="T3 buffer sizes for Merge step"),
    parallel: bool = typer.Option(False, "--parallel", is_flag=True, help="Run geo codes in parallel"),
    n_workers: int = typer.Option(2, "--n_workers", help="Number of parallel workers"),
    log_level: str = typer.Option("INFO", "--log_level", help="Logging level"),
    overwrite: bool = typer.Option(True, "--overwrite/--no-overwrite", help="Overwrite existing output files (--no-overwrite resumes, returning cached CSVs)"),
):
    """Run one greenpy module over a study area.

    T3, T30, T30_buildings, T300, Tree_count and Visibility iterate over every
    code of --geo_level (default: the coarsest level in config.columns.geo_levels)
    and write one CSV per code, aggregated at --sub_geo_level (default: the
    finest level; T30_buildings reports per building instead).
    Spectral does the same via Google Earth Engine. Merge consolidates all
    module CSVs into `<output.base_dir>/database/T3_30_300_spectral.parquet`
    and must run last.
    """
    cfg = load_config(config)
    geo_levels = cfg.columns.geo_levels

    valid_processes = {"T3", "T30", "T30_buildings", "T300", "Tree_count", "Visibility", "Spectral", "Merge"}
    if process not in valid_processes:
        typer.echo(f"Error: --process must be one of {sorted(valid_processes)}", err=True)
        raise typer.Exit(1)

    if observer_mode not in ("facade", "centroid"):
        typer.echo(f"Error: --observer_mode must be 'facade' or 'centroid', got '{observer_mode}'", err=True)
        raise typer.Exit(1)

    if geo_level and geo_level not in geo_levels:
        typer.echo(f"Error: --geo_level '{geo_level}' not in config.columns.geo_levels: {geo_levels}", err=True)
        raise typer.Exit(1)

    h3_from_cli = h3_resolution is not None
    if h3_resolution is None:
        h3_resolution = cfg.h3_resolution
    if h3_resolution is not None and process == "Spectral":
        # Spectral aggregates server-side on a GEE boundaries asset, so hexagons don't apply
        if h3_from_cli:
            typer.echo("Error: Spectral uses a GEE boundaries asset and does not support --h3_resolution", err=True)
            raise typer.Exit(1)
        logger.warning("Spectral does not support H3 aggregation — ignoring h3_resolution from config")
        h3_resolution = None
    if h3_resolution is not None:
        if not 0 <= h3_resolution <= 15:
            typer.echo(f"Error: --h3_resolution must be between 0 and 15, got {h3_resolution}", err=True)
            raise typer.Exit(1)
        if sub_geo_level:
            typer.echo("Error: --sub_geo_level cannot be combined with --h3_resolution (hexagons replace the sub-geography)", err=True)
            raise typer.Exit(1)
        sub_geo_level = h3_column(h3_resolution)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir / f"{process}_processing.log", log_level)

    if process == "Merge":
        from .merge import merge_output_csv, process_data
        sedona = get_spark()
        merge_output_csv(sedona, cfg, t3_buffers)
        if h3_resolution is not None:
            # one output row per hexagon unless the user aggregates up to a census level
            merge_geo_level = geo_level or sub_geo_level
        else:
            merge_geo_level = geo_level or (geo_levels[-2] if len(geo_levels) > 1 else geo_levels[0])
        process_data(sedona, cfg, merge_geo_level, sub_geo_level or geo_levels[-1], t3_buffers, h3_resolution=h3_resolution)
        return

    if process == "Spectral":
        from .optional.spectral import setup_gee, process_geo_code as process_spectral
        if not cfg.gee_boundaries_asset:
            typer.echo("Error: Spectral requires gee_boundaries_asset in the config", err=True)
            raise typer.Exit(1)
        sedona = get_spark()
        tables = load_tables(sedona, cfg)
        setup_gee(cfg.gee_project)
        spectral_geo_level = geo_level or geo_levels[0]
        codes = [geo_code] if geo_code else tables["census_boundaries_gdf"][spectral_geo_level].unique()
        for code in tqdm(codes, desc="Regions"):
            process_spectral(
                code, spectral_geo_level, sub_geo_level or geo_levels[-1],
                imagery_ee_path, start_date, end_date, cloud_coverage,
                spectral_indexes,
                output_dir=tables["output_dirs"]["spectral"],
                gee_boundaries_asset=cfg.gee_boundaries_asset,
                overwrite=overwrite,
            )
        return

    sedona = get_spark()
    tables = load_tables(sedona, cfg, h3_resolution=h3_resolution)
    dirs = tables["output_dirs"]

    output_dir_map = {"T3": dirs["t3"], "T30": dirs["t30"], "T30_buildings": dirs["t30_buildings"], "T300": dirs["t300"], "Tree_count": dirs["tree_count"], "Visibility": dirs["visibility"]}
    output_dir = output_dir_map[process]

    args_dict = {
        "sedona": sedona,
        "geo_level": geo_level or geo_levels[0],
        "sub_geo_level": sub_geo_level or (geo_levels[-1] if len(geo_levels) > 1 else geo_levels[0]),
        "cfg": cfg,
        "output_dir": output_dir,
        "query_method": query_method,
        "buffer": buffer,
        "tree_area": tree_area,
        "tree_height": tree_height,
        "observer_mode": observer_mode,
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
        "gee_scale": gee_scale,
        "overwrite": overwrite,
        "road_nodes_gdf": tables.get("road_nodes_gdf"),
        "road_edges_gdf": tables.get("road_edges_gdf"),
    }

    codes = [geo_code] if geo_code else tables["census_boundaries_gdf"][geo_level or geo_levels[0]].unique()
    logger.info(f"Running {process} for {len(codes)} regions")

    if parallel:
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_run_process, process, args_dict, code) for code in codes]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Regions"):
                future.result()
    else:
        for code in tqdm(codes, desc="Regions"):
            _run_process(process, args_dict, code)

    logger.info(f"{process} completed for {len(codes)} regions")


def main():
    app()


if __name__ == "__main__":
    main()
