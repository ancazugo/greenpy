from pathlib import Path

import time
import pandas as pd
import geopandas as gpd
import rioxarray as rxr
from loguru import logger
from pyproj import CRS
from pyspark.sql.session import SparkSession

from .config.schema import GreenPyConfig
from .utils.data_processing import (
    get_geometries,
    filter_buffer_geometries,
    load_trees_gdf,
    find_overlapping_rasters,
    view_suffix,
    drop_geo_views,
)

RESULT_COLUMNS = ["building_id", "tree_pixels", "total_pixels", "canopy_cover"]


def get_canopy_cover_buildings_raster(
    sedona: SparkSession,
    chm_paths: list,
    geo_code: str,
    epsg: int,
    low_threshold: float = 3,
    high_threshold: float = 60,
    already_binary: bool = False,
) -> pd.DataFrame:
    """Per-building canopy cover (%) via distributed Sedona RS_ZonalStats.

    Requires the `buildings_buffers_<geo_code>` temp view to be registered
    (done by filter_buffer_geometries()). All tiles are binarised and loaded
    into one Sedona DataFrame so Spark distributes the (building x tile) join
    in a single job. SUM across tiles assumes non-overlapping tiles: buffers
    spanning a tile boundary aggregate correctly, but overlapping tiles would
    double-count pixels.

    With already_binary=True the tiles are assumed to hold a 1/0/nodata canopy
    mask already (e.g. the cached GEE download) and the height-threshold
    binarisation is skipped.

    Returns a DataFrame with building_id, tree_pixels, total_pixels and
    canopy_cover (0-100, 3 dp); buildings falling entirely in nodata are dropped.
    """
    sfx = view_suffix(geo_code)

    # Pre-filter: skip tiles that can't be decoded (corrupt compression or
    # invalid format), sampling rows at 25/50/75% of tile height to catch
    # mid-file corruption cheaply, and fail loudly on CRS mismatches — a tile
    # in another CRS would silently miss every building in RS_Intersects.
    valid_paths = []
    for path in chm_paths:
        try:
            r = rxr.open_rasterio(path)
            tile_epsg = r.rio.crs.to_epsg() if r.rio.crs is not None else None
            if tile_epsg != epsg:
                raise ValueError(
                    f"Tile {path} is in EPSG:{tile_epsg} but the config CRS is "
                    f"EPSG:{epsg} — reproject the tiles or fix cfg.crs"
                )
            n_rows = r.shape[1]
            for frac in [0.25, 0.50, 0.75]:
                r.isel(y=slice(int(n_rows * frac), int(n_rows * frac) + 1)).values
            valid_paths.append(str(path))
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Skipping unreadable tile {Path(path).name}: {e}")

    if not valid_paths:
        logger.warning(f"No valid CHM tiles found for {geo_code}, returning empty result")
        return pd.DataFrame(columns=RESULT_COLUMNS)

    logger.debug(f"Loading {len(valid_paths)}/{len(chm_paths)} valid CHM tiles into Sedona")

    raster_sdf = (
        sedona.read.format("binaryFile")
        .load(valid_paths)
        .selectExpr("RS_FromGeoTiff(content) AS raster")
    )
    raster_sdf.createOrReplaceTempView(f"raw_tiles_{sfx}")

    # Tile-explode large rasters into 512×512 chunks, then binarise in-SQL:
    # pixels with CHM height in [low_threshold, high_threshold] → 1.0 (tree), else → 0.0
    tile_expr = (
        "tile"
        if already_binary
        else (
            f"RS_MapAlgebra(tile, 'D', "
            f"'out = (rast[0] >= {low_threshold} && rast[0] <= {high_threshold}) ? 1.0 : 0.0;')"
        )
    )
    binary_sdf = sedona.sql(f"""
        SELECT {tile_expr} AS tile
        FROM (
            SELECT RS_TileExplode(raster, 512, 512) AS (x_idx, y_idx, tile)
            FROM raw_tiles_{sfx}
        )
    """)
    binary_sdf.createOrReplaceTempView(f"binary_tiles_{sfx}")

    from sedona.spark.utils.adapter import Adapter
    from sedona.spark.core.enums import GridType, IndexType

    buildings_sdf = sedona.sql(
        f"SELECT building_id, ST_SetSRID(geometry, {epsg}) AS geometry FROM buildings_buffers_{sfx}"
    )
    buildings_rdd = Adapter.toSpatialRdd(buildings_sdf, "geometry")
    buildings_rdd.analyze()
    # KDB-tree partitioning requires partitions <= records/2; cap for small datasets
    default_parallelism = sedona.sparkContext.defaultParallelism
    total = buildings_rdd.approximateTotalCount or 1
    num_partitions = max(1, min(default_parallelism, total // 2))
    buildings_rdd.spatialPartitioning(GridType.KDBTREE, num_partitions)
    buildings_rdd.buildIndex(IndexType.QUADTREE, True)
    buildings_partitioned_sdf = Adapter.toDf(buildings_rdd, ["building_id"], sedona)
    # Adapter.toDf() strips SRID; re-apply so RS_Intersects matches the raster's CRS
    buildings_partitioned_sdf = buildings_partitioned_sdf.selectExpr(
        "building_id",
        f"ST_SetSRID(geometry, {epsg}) AS geometry",
    )
    buildings_partitioned_sdf.createOrReplaceTempView(f"buildings_partitioned_{sfx}")

    # On the binary raster: 'sum' = tree pixels (1.0 values), 'count' = all valid pixels.
    result_sdf = sedona.sql(f"""
        SELECT
            b.building_id,
            SUM(RS_ZonalStats(bt.tile, b.geometry, 1, 'sum',   true)) AS tree_pixels,
            SUM(RS_ZonalStats(bt.tile, b.geometry, 1, 'count', true)) AS total_pixels
        FROM buildings_partitioned_{sfx} b, binary_tiles_{sfx} bt
        WHERE RS_Intersects(bt.tile, b.geometry)
        GROUP BY b.building_id
    """)

    result_df = result_sdf.toPandas()
    logger.debug(f"RS_ZonalStats returned {len(result_df)} buildings before filtering")

    if result_df.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    # Drop buildings whose geometry fell entirely in nodata regions of the raster
    result_df = result_df[result_df["total_pixels"] > 0].copy()
    result_df["tree_pixels"] = result_df["tree_pixels"].fillna(0)
    result_df["canopy_cover"] = (100.0 * result_df["tree_pixels"] / result_df["total_pixels"]).round(3)
    return result_df


def get_canopy_cover_buildings_vector(
    sedona: SparkSession, geo_code: str, trees_gdf: gpd.GeoDataFrame
) -> pd.DataFrame:
    """Per-building canopy cover (%) from tree vectors via a Sedona spatial join.

    For polygon trees the clipped intersection area is used; for points with a
    stored `tree_area` the whole tree area counts (mirroring
    t30.get_canopy_cover_vector, so overlapping canopies can exceed 100%).
    `total_pixels` holds the buffer area in m² (1 m² ≈ one pixel) for
    consistency with the raster path. Buildings with no trees get 0.
    """
    sfx = view_suffix(geo_code)

    polygonal = set(trees_gdf.geometry.geom_type.unique()) <= {"Polygon", "MultiPolygon"}
    cols = ["geometry"] + (["tree_area"] if "tree_area" in trees_gdf.columns else [])
    if not polygonal and "tree_area" not in cols:
        raise ValueError(
            "Non-polygon trees need a tree_area column (columns.tree_area_col) "
            "to compute canopy cover per building"
        )
    sedona.createDataFrame(trees_gdf[cols]).createOrReplaceTempView(f"t30b_trees_{sfx}")

    canopy_expr = (
        "ST_Area(ST_Intersection(t.geometry, b.geometry))" if polygonal else "t.tree_area"
    )
    result_df = sedona.sql(f"""
        SELECT
            b.building_id,
            ROUND(SUM(COALESCE({canopy_expr}, 0)), 3) AS tree_pixels,
            ROUND(MAX(ST_Area(b.geometry))) AS total_pixels
        FROM buildings_buffers_{sfx} b
        LEFT JOIN t30b_trees_{sfx} t ON ST_Intersects(b.geometry, t.geometry)
        GROUP BY b.building_id
    """).toPandas()

    result_df["canopy_cover"] = (100.0 * result_df["tree_pixels"] / result_df["total_pixels"]).round(3)
    return result_df


def process_geo_code(
    sedona: SparkSession,
    geo_level: str,
    geo_code: str,
    cfg: GreenPyConfig,
    output_dir: Path,
    buffer: int = 100,
    low_threshold: int = 3,
    high_threshold: int = 60,
    gee_scale: float = 1.0,
    overwrite: bool = True,
) -> pd.DataFrame | None:
    """Compute canopy cover (%) within `buffer` metres of each building in one geo_code.

    Canopy source, in T30's priority order: local CHM raster tiles
    (cfg.data.chm_tiles_dir) > a GEE canopy-height asset binarised server-side
    (cfg.data.canopy_height_ee_path) > tree polygon areas (cfg.data.trees_dir).
    Writes `T30_buildings_<geo_code>_<buffer>m.csv` with columns building_id,
    tree_pixels, total_pixels, canopy_cover. Returns the DataFrame, the cached
    CSV when overwrite is False, or None on error.
    """
    start_time = time.time()
    logger.info(f"T30_buildings: processing {geo_code} with buffer {buffer}m")

    out_path = output_dir / f"T30_buildings_{geo_code}_{buffer}m.csv"

    if out_path.exists() and not overwrite:
        return pd.read_csv(out_path)

    try:
        epsg = CRS.from_user_input(cfg.crs).to_epsg()
        if epsg is None:
            raise ValueError(
                f"cfg.crs '{cfg.crs}' has no EPSG code — T30_buildings needs one for ST_SetSRID"
            )

        get_geometries(sedona, geo_level, geo_code, dissolve=True)
        filter_buffer_geometries(sedona, geo_level, geo_code, "buildings", buffer, id_col="building_id")
        sfx = view_suffix(geo_code)
        if not buffer:
            # buffer 0 = footprint only; filter_buffer_geometries skips the buffers view
            sedona.sql(
                f"SELECT geometry, building_id FROM geo_buildings_{sfx}"
            ).createOrReplaceTempView(f"buildings_buffers_{sfx}")

        # Buffered buildings extend up to `buffer` metres past the geo boundary,
        # so canopy tiles/trees must be selected against the widened boundary too
        boundary_gdf = gpd.GeoDataFrame(
            sedona.table(f"geo_boundary_{sfx}").toPandas(), geometry="geometry", crs=cfg.crs
        )
        search_gdf = boundary_gdf.copy()
        search_gdf["geometry"] = search_gdf.buffer(buffer or 0)

        if cfg.data.chm_tiles_dir:
            chm_paths = find_overlapping_rasters(search_gdf, Path(cfg.data.chm_tiles_dir), pattern="*.tif")
            result_df = get_canopy_cover_buildings_raster(
                sedona, chm_paths, geo_code, epsg, low_threshold, high_threshold
            )

        elif cfg.data.canopy_height_ee_path:
            from .optional.canopy_gee import download_binary_canopy  # lazy: keeps ee/xee optional
            # buffer-suffixed cache: T30's gee_canopy/<geo_code>.tif only covers
            # the unbuffered boundary bounds
            cache = Path(cfg.output.base_dir) / "database" / "gee_canopy" / f"{geo_code}_b{buffer}m.tif"
            download_binary_canopy(
                search_gdf, cfg, low_threshold, high_threshold,
                scale=gee_scale, cache_path=cache, overwrite=overwrite,
            )
            result_df = get_canopy_cover_buildings_raster(
                sedona, [cache], geo_code, epsg, already_binary=True
            )

        elif cfg.data.trees_dir:
            trees_gdf = load_trees_gdf(Path(cfg.data.trees_dir), search_gdf, cfg)
            result_df = get_canopy_cover_buildings_vector(sedona, geo_code, trees_gdf)

        else:
            raise ValueError(
                "No canopy source configured — set one of chm_tiles_dir, "
                "canopy_height_ee_path, or trees_dir to compute T30_buildings"
            )

        result_df = result_df[RESULT_COLUMNS]
        result_df.to_csv(out_path, index=False)

        end_time = time.time()
        logger.info(f"T30_buildings: {geo_code} — {len(result_df)} records in {end_time - start_time:.2f}s")
        return result_df

    except Exception:
        logger.exception(f"T30_buildings: error processing {geo_code}")
        return None
    finally:
        drop_geo_views(sedona, geo_code)
