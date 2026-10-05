"""GEE downloads for the remote height sources: tiled rasters (xee) and chunked feature collections.

Rasters are pulled onto a pinned grid in the projected study CRS, in square
tiles aligned to multiples of the tile size, so a rerun (or a neighbouring
study area) reuses every tile already on disk. Only tiles holding footprints
are downloaded. Features are fetched in small lon/lat chunks, each cached as
its own parquet so an interrupted fetch resumes where it stopped.
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import affine
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from loguru import logger
from tqdm import tqdm

from ..utils.gee import ensure_gee, write_raster
from .match import grid_chunks

# Pixels per tile side for raster downloads (4096 px x 4 B = 64 MB per tile)
TILE_PX = 4096


def tag(payload: dict) -> str:
    """Short stable hash naming a cache directory for a download configuration."""
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]


def tiles_with_footprints(buildings: gpd.GeoDataFrame, res: float, tile_px: int = TILE_PX) -> list[tuple[int, int]]:
    """(row, col) of the tile_px*res grid tiles (aligned to multiples of the tile side) touched by footprints."""
    side = res * tile_px
    b = shapely.bounds(buildings.geometry.values)
    tiles = set()
    for c0, c1, r0, r1 in zip(
        np.floor(b[:, 0] / side).astype(int), np.floor(b[:, 2] / side).astype(int),
        np.floor(b[:, 1] / side).astype(int), np.floor(b[:, 3] / side).astype(int),
    ):
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                tiles.add((r, c))
    return sorted(tiles)


def tiles_for_extent(bounds: tuple, res: float, tile_px: int = TILE_PX) -> list[tuple[int, int]]:
    """(row, col) of the tile_px*res grid tiles (aligned to multiples of the tile side) covering bounds."""
    side = res * tile_px
    minx, miny, maxx, maxy = bounds
    return [
        (r, c)
        for r in range(int(np.floor(miny / side)), int(np.floor(maxy / side)) + 1)
        for c in range(int(np.floor(minx / side)), int(np.floor(maxx / side)) + 1)
    ]


def download_image_tiles(
    make_image, band: str, crs: str, res: float, tiles: list[tuple[int, int]], out_dir: Path,
    project: str | None, tile_px: int = TILE_PX, workers: int = 4,
) -> list[Path]:
    """Download make_image() (an ee.Image with band `band`; masked pixels -> NaN) as GeoTIFF tiles in crs at res metres.

    make_image is called once GEE is initialised, and only when a tile is missing.

    Tile (row, col) spans x in [col, col+1) * side and y in [row, row+1) * side
    with side = res * tile_px. Existing tiles are reused. Returns the paths of
    tiles that hold any data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    side = res * tile_px
    todo = [(r, c) for r, c in tiles if not (out_dir / f"t_{r}_{c}.tif").exists() and not (out_dir / f"t_{r}_{c}.empty").exists()]
    if todo:
        ensure_gee(project)
        image = make_image()
        logger.info(f"Downloading {len(todo)}/{len(tiles)} tiles of {band} at {res} m into {out_dir}")

        def fetch(rc):
            r, c = rc
            _download_tile(image, band, crs, affine.Affine(res, 0, c * side, 0, -res, (r + 1) * side),
                           tile_px, out_dir / f"t_{r}_{c}.tif")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(fetch, todo), total=len(todo), desc=f"{band} tiles"))
    return [out_dir / f"t_{r}_{c}.tif" for r, c in tiles if (out_dir / f"t_{r}_{c}.tif").exists()]


def _download_tile(image, band: str, crs: str, transform: affine.Affine, tile_px: int, path: Path) -> None:
    """One xee download onto a pinned grid; writes path, or a .empty marker when the tile has no data."""
    import ee
    import xarray as xr

    ds = xr.open_dataset(
        ee.ImageCollection([image.select([band]).toFloat()]),
        engine="ee", crs=crs, crs_transform=transform, shape_2d=(tile_px, tile_px),
    )
    da = ds[band]
    if "time" in da.dims:
        da = da.isel(time=0, drop=True)
    # xee returns dims (X, Y); orient to (y, x) descending-y like a GeoTIFF
    da = da.rename({d: ("x" if d.lower() in ("x", "lon") else "y") for d in da.dims}).transpose("y", "x")
    da = da.sortby("y", ascending=False).sortby("x").load()
    if not np.isfinite(da.values).any():
        path.with_suffix(".empty").touch()
        return
    da = da.rio.write_crs(crs).rio.write_nodata(float("nan"))
    write_raster(da, path, compress="deflate", tiled=True)


def fetch_features(
    collections: list[tuple[str, tuple]], out_dir: Path, project: str | None,
    properties: list[str], chunk_deg: float = 0.02, workers: int = 8,
) -> gpd.GeoDataFrame:
    """Features of each (FeatureCollection id, EPSG:4326 bounds) pair, fetched per chunk_deg chunk.

    Each (collection, chunk) is cached as parquet in out_dir; a feature
    straddling chunks comes back more than once (dedupe downstream). Returns
    the `properties` + geometry in EPSG:4326.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for cid, bounds in collections:
        for ch in grid_chunks(bounds, chunk_deg):
            name = f"{cid.rstrip('/').split('/')[-1]}_{ch[0]:.4f}_{ch[1]:.4f}_{ch[2]:.4f}_{ch[3]:.4f}.parquet"
            jobs.append((cid, ch, out_dir / name))
    todo = [j for j in jobs if not j[2].exists()]
    if todo:
        ensure_gee(project)
        logger.info(f"Fetching {len(todo)}/{len(jobs)} feature chunks into {out_dir}")

        def fetch(job):
            cid, ch, path = job
            _fetch_chunk(cid, ch, properties, path)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(tqdm(ex.map(fetch, todo), total=len(todo), desc="feature chunks"))
    parts = [gpd.read_parquet(p) for _, _, p in jobs]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return gpd.GeoDataFrame({k: [] for k in properties}, geometry=[], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def _fetch_chunk(collection_id: str, bounds: tuple, properties: list[str], path: Path, depth: int = 0) -> None:
    """computeFeatures for one chunk (split into quarters when GEE refuses it as too large)."""
    import ee

    region = ee.Geometry.Rectangle(list(bounds), proj="EPSG:4326", geodesic=False)
    fc = ee.FeatureCollection(collection_id).filterBounds(region).select(properties)
    try:
        gdf = ee.data.computeFeatures({"expression": fc, "fileFormat": "GEOPANDAS_GEODATAFRAME"})
    except ee.EEException as e:
        if depth >= 4 or "too" not in str(e).lower():
            raise
        minx, miny, maxx, maxy = bounds
        mx, my = (minx + maxx) / 2, (miny + maxy) / 2
        quarters = [(minx, miny, mx, my), (mx, miny, maxx, my), (minx, my, mx, maxy), (mx, my, maxx, maxy)]
        parts = []
        for i, q in enumerate(quarters):
            qp = path.with_name(f"{path.stem}_q{depth}{i}.parquet")
            _fetch_chunk(collection_id, q, properties, qp, depth + 1)
            parts.append(gpd.read_parquet(qp))
            qp.unlink()
        gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)
    if gdf.empty or "geometry" not in gdf.columns:
        # GEE returns a frame without a geometry column for chunks with no features
        gdf = gpd.GeoDataFrame({k: [] for k in properties}, geometry=[], crs=4326)
    elif gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf[[c for c in properties if c in gdf.columns] + ["geometry"]]
    tmp = path.with_suffix(".parquet.part")
    gdf.to_parquet(tmp, index=False)
    tmp.replace(path)


def list_asset_names(root: str, cache: Path, project: str | None) -> list[str]:
    """Names of the assets directly under a GEE folder, cached as JSON at `cache`."""
    if cache.exists():
        return json.loads(cache.read_text())
    import ee

    ensure_gee(project)
    names, token = [], None
    while True:
        page = ee.data.listAssets({"parent": root, **({"pageToken": token} if token else {})})
        names += [a["name"].rstrip("/").split("/")[-1] for a in page.get("assets", [])]
        token = page.get("nextPageToken")
        if not token:
            break
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(sorted(names)))
    return sorted(names)


def bounds_4326(buildings: gpd.GeoDataFrame, pad_deg: float = 0.001) -> tuple:
    minx, miny, maxx, maxy = buildings.to_crs(4326).total_bounds
    return (minx - pad_deg, miny - pad_deg, maxx + pad_deg, maxy + pad_deg)

