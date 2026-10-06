"""
Canopy height model sources for tree segmentation (optional module).

Both sources end as GDAL VRT mosaics of north-up GeoTIFFs sharing one CRS and
resolution, so the block driver in tree_segmentation reads every source the
same way:

- local tiles (data.chm_tiles_dir): a cached extent index avoids opening every
  tile per geo_code. Where tiles overlap (e.g. survey years) either the last
  path in sorted order wins ("latest": the latest year for <dir>/<year>/
  layouts), or each overlapping set becomes its own layer on a shared grid and
  the reader takes the per-pixel maximum ("max": a tree seen in any survey
  counts — useful when a later survey was flown leaf-off).
- Meta/WRI global canopy height (1 m, EPSG:3857, whole metres) on the public
  AWS bucket, downloaded once into the cache as whole tiles:
  - v1 (Tolan et al. 2024, `forests/v1/alsgedi_global_v6_float`): zoom-9
    quadkey GeoTIFFs in strips, which make remote windowed reads slow.
  - v2 (`forests/v2/global/dinov3_global_chm_v2_ml3`): zoom-10 quadkey uint8
    COGs with no declared nodata — unimaged pixels (e.g. sea) read 0, like
    bare ground. Each tile ships a metadata GeoJSON of the imagery footprints
    (with acq_date); pixels outside them are written as nodata on download,
    and the footprints are kept beside the tile.
"""

import hashlib
import json
import math
import os
from pathlib import Path
from xml.sax.saxutils import escape

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.warp
import rasterio.windows
import requests
from loguru import logger
from pyproj import CRS
from shapely.geometry import box

from ..utils.data_processing import is_chm_tile

_META_BUCKET = "https://dataforgood-fb-data.s3.amazonaws.com/forests"
META_CHM_URL = f"{_META_BUCKET}/v1/alsgedi_global_v6_float/chm/{{quadkey}}.tif"
META_ZOOM = 9

# Meta CHM releases: tile URL, quadkey zoom, cache subdirectory, imagery-footprint URL
META_VERSIONS = {
    "v1": {"url": META_CHM_URL, "zoom": META_ZOOM, "dir": "meta_chm", "footprints": None},
    "v2": {
        "url": f"{_META_BUCKET}/v2/global/dinov3_global_chm_v2_ml3/chm/{{quadkey}}.tif",
        "zoom": 10,
        "dir": "meta_chm_v2",
        "footprints": f"{_META_BUCKET}/v2/global/dinov3_global_chm_v2_ml3/metadata/{{quadkey}}.geojson",
    },
}
# nodata written into v2 tiles outside the imagery footprints (heights are uint8 metres)
META_V2_NODATA = 255


# --------------------------------------------------------------------------- #
# Local tiles
# --------------------------------------------------------------------------- #

def raster_index(files_dir: Path, pattern: str, cache_dir: Path) -> gpd.GeoDataFrame:
    """Extents of the rasters under files_dir matching pattern (recursive), cached in cache_dir.

    The cache is keyed by directory and pattern and rebuilt when the set of
    files changes. Bounds are stored in EPSG:4326 for selection; crs keeps
    each raster's own CRS; unreadable rasters are kept without an extent.
    Known non-height rasters (Defra VOM hillshades) are skipped.
    """
    files = sorted(str(p) for p in Path(files_dir).rglob(pattern) if is_chm_tile(p))
    key = hashlib.sha1(f"{Path(files_dir).resolve()}|{pattern}".encode()).hexdigest()[:12]
    cache = Path(cache_dir) / f"raster_index_{key}.parquet"
    if cache.exists():
        idx = gpd.read_parquet(cache)
        if sorted(idx["path"]) == files:
            return idx

    logger.info(f"Indexing {len(files)} rasters in {files_dir}")
    rows = []
    for f in files:
        try:
            with rasterio.open(f) as src:
                rows.append({
                    "path": f,
                    "crs": src.crs.to_string(),
                    "geometry": box(*rasterio.warp.transform_bounds(src.crs, "EPSG:4326", *src.bounds)),
                })
        except Exception as e:
            # kept with no extent, so the cache still matches the directory listing
            logger.warning(f"Could not read extent of {f}: {e}")
            rows.append({"path": f, "crs": None, "geometry": None})
    idx = gpd.GeoDataFrame(rows, columns=["path", "crs", "geometry"], geometry="geometry", crs="EPSG:4326")
    cache.parent.mkdir(parents=True, exist_ok=True)
    idx.to_parquet(cache)
    return idx


def tiles_for_boundary(boundary_gdf: gpd.GeoDataFrame, files_dir: Path, pattern: str, cache_dir: Path) -> list[str]:
    """Sorted raster paths whose extent intersects the boundary."""
    idx = raster_index(files_dir, pattern, cache_dir)
    if idx.empty:
        return []
    area = boundary_gdf.to_crs("EPSG:4326").union_all()
    return sorted(idx.loc[idx.geometry.notna() & idx.intersects(area), "path"])


# --------------------------------------------------------------------------- #
# Meta / WRI global canopy height
# --------------------------------------------------------------------------- #

def _tile_xy(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    lat = max(min(lat, 85.0511), -85.0511)
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def quadkey(x: int, y: int, zoom: int) -> str:
    digits = []
    for i in range(zoom, 0, -1):
        mask = 1 << (i - 1)
        digits.append(str((1 if x & mask else 0) + (2 if y & mask else 0)))
    return "".join(digits)


def quadkeys_for_bounds(west: float, south: float, east: float, north: float, zoom: int = META_ZOOM) -> list[str]:
    """Quadkeys of the web-mercator tiles covering a lon/lat box."""
    x0, y0 = _tile_xy(west, north, zoom)
    x1, y1 = _tile_xy(east, south, zoom)
    return [quadkey(x, y, zoom) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def download_meta_tiles(
    boundary_gdf: gpd.GeoDataFrame, cache_dir: Path, overwrite: bool = False, version: str = "v1",
) -> list[str]:
    """Download the Meta CHM tiles covering the boundary into the cache; returns local paths.

    version selects the release (META_VERSIONS); each has its own cache
    subdirectory. Tiles missing from the bucket (open ocean) are skipped.
    Downloads go to a .part file first, so an interrupted run never leaves a
    truncated tile. v2 tiles are masked to their imagery footprints before
    they are cached.
    """
    if version not in META_VERSIONS:
        raise ValueError(f"Unknown Meta CHM version {version!r}; expected one of {sorted(META_VERSIONS)}")
    spec = META_VERSIONS[version]
    out_dir = Path(cache_dir) / spec["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = quadkeys_for_bounds(*boundary_gdf.to_crs("EPSG:4326").total_bounds, zoom=spec["zoom"])
    paths = []
    for qk in keys:
        path = out_dir / f"{qk}.tif"
        if path.exists() and not overwrite:
            ensure_tiled(path)
            paths.append(str(path))
            continue
        url = spec["url"].format(quadkey=qk)
        with requests.get(url, stream=True, timeout=60) as r:
            if r.status_code in (403, 404):
                logger.warning(f"Meta CHM tile {qk} not available ({r.status_code}), skipping")
                continue
            r.raise_for_status()
            size = int(r.headers.get("Content-Length", 0))
            logger.info(f"Downloading Meta CHM {version} tile {qk} ({size / 1e6:.0f} MB)")
            part = path.with_suffix(".tif.part")
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 << 20):
                    f.write(chunk)
            if size and part.stat().st_size != size:
                part.unlink()
                raise IOError(f"Incomplete download of Meta CHM tile {qk}")
        if spec["footprints"]:
            footprints = _download_footprints(spec["footprints"].format(quadkey=qk), path.with_suffix(".geojson"))
            mask_to_footprints(part, footprints, path)
            part.unlink()
        else:
            part.rename(path)
        ensure_tiled(path)
        paths.append(str(path))
    return paths


def _download_footprints(url: str, out_path: Path) -> dict:
    """Fetch a v2 tile's imagery-footprint GeoJSON (EPSG:4326, with acq_date) and keep it beside the tile."""
    r = requests.get(url, timeout=60)
    if r.status_code in (403, 404):
        # without footprints the tile cannot be masked; treat it as fully imaged
        logger.warning(f"No imagery footprints at {url}; the tile is used unmasked")
        return {"type": "FeatureCollection", "features": []}
    r.raise_for_status()
    out_path.write_text(r.text)
    return r.json()


def mask_to_footprints(src_path: Path, footprints: dict, dst_path: Path, block_rows: int = 4096) -> None:
    """Copy a uint8 v2 CHM tile to dst_path as a tiled GeoTIFF with nodata outside the imagery footprints.

    With no footprint features the tile is copied unmasked (nodata still
    declared, so every pixel stays valid). Written row-band by row-band to
    keep memory bounded on 32768 px tiles.
    """
    import rasterio.features
    from rasterio.warp import transform_geom

    with rasterio.open(src_path) as src:
        geoms = [transform_geom("EPSG:4326", src.crs, f["geometry"]) for f in footprints.get("features", [])]
        profile = src.profile.copy()
        profile.update(driver="GTiff", tiled=True, blockxsize=512, blockysize=512, compress="deflate",
                       nodata=META_V2_NODATA, bigtiff="IF_SAFER")
        tmp = dst_path.with_name(f"{dst_path.name}.{os.getpid()}.mask")
        n_masked = 0
        with rasterio.open(tmp, "w", **profile) as dst:
            for row in range(0, src.height, block_rows):
                win = rasterio.windows.Window(0, row, src.width, min(block_rows, src.height - row))
                z = src.read(1, window=win)
                if geoms:
                    inside = rasterio.features.geometry_mask(
                        geoms, (int(win.height), int(win.width)), rasterio.windows.transform(win, src.transform),
                        invert=True,
                    )
                    n_masked += int((~inside).sum())
                    z = np.where(inside, z, META_V2_NODATA).astype(z.dtype)
                dst.write(z, 1, window=win)
        total = src.width * src.height
    tmp.replace(dst_path)
    if n_masked:
        logger.info(f"Masked {100 * n_masked / total:.1f}% of {dst_path.name} outside the imagery footprints")


def footprint_dates(paths: list[str]) -> tuple[str, str] | None:
    """(earliest, latest) imagery acq_date across the cached footprints of v2 tiles, or None."""
    dates = []
    for p in paths:
        fp = Path(p).with_suffix(".geojson")
        if fp.exists():
            dates += [f["properties"].get("acq_date") for f in json.loads(fp.read_text()).get("features", [])]
    dates = sorted(d for d in dates if d)
    return (dates[0], dates[-1]) if dates else None


def ensure_tiled(path: Path, block: int = 512, threads: int = 8) -> None:
    """Rewrite a GeoTIFF stored in strips as an internally tiled one (same pixels), in place.

    Meta's tiles are 65,536 px wide single-row strips, so any window read
    decompresses whole rows across the tile — reading a 2 km window meant
    decompressing ~65x more pixels than needed, every time. Tiled blocks make
    windowed and warped reads touch only the blocks they need.
    """
    import rasterio.shutil

    path = Path(path)
    with rasterio.open(path) as src:
        bh, bw = src.block_shapes[0]
        if bh >= block and bw >= block or src.width <= block:
            return
        profile = {"tiled": True, "blockxsize": block, "blockysize": block,
                   "compress": (src.compression.value if src.compression else "deflate"),
                   "num_threads": str(threads), "bigtiff": "IF_SAFER"}
    logger.info(f"Re-tiling {path.name} ({bh}x{bw} px strips -> {block} px tiles), once per tile")
    tmp = path.with_name(f"{path.name}.{os.getpid()}.retile")
    rasterio.shutil.copy(path, tmp, driver="GTiff", **profile)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Mosaic
# --------------------------------------------------------------------------- #

def build_vrt(paths: list[str], vrt_path: Path, extent: tuple | None = None) -> Path:
    """Write a Float32 mosaic VRT of single-band, north-up rasters sharing CRS and pixel size.

    Later paths are drawn over earlier ones; source nodata stays transparent
    and uncovered areas read as NaN. `extent` (minx, miny, maxx, maxy) forces
    the VRT grid, so several VRTs can share one pixel grid.
    """
    if not paths:
        raise FileNotFoundError("No CHM rasters to mosaic")
    infos = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.transform.b or src.transform.d:
                raise ValueError(f"{p} is rotated; only north-up rasters are supported")
            infos.append((p, src.crs, src.transform, src.width, src.height, src.nodata))

    crs0, t0 = infos[0][1], infos[0][2]
    res_x, res_y = t0.a, -t0.e
    for p, crs, t, *_ in infos[1:]:
        if CRS.from_user_input(crs) != CRS.from_user_input(crs0):
            raise ValueError(f"{p} has CRS {crs}, expected {crs0}; CHM tiles must share one CRS")
        if not (math.isclose(t.a, res_x, rel_tol=1e-6) and math.isclose(-t.e, res_y, rel_tol=1e-6)):
            raise ValueError(f"{p} has pixel size {t.a} x {-t.e}, expected {res_x} x {res_y}")

    if extent is not None:
        minx, miny, maxx, maxy = extent
    else:
        minx = min(t.c for _, _, t, *_ in infos)
        maxy = max(t.f for _, _, t, *_ in infos)
        maxx = max(t.c + w * res_x for _, _, t, w, _, _ in infos)
        miny = min(t.f - h * res_y for _, _, t, _, h, _ in infos)
    width, height = round((maxx - minx) / res_x), round((maxy - miny) / res_y)

    sources = []
    for p, _, t, w, h, nodata in infos:
        nd = f"<NODATA>{nodata}</NODATA>" if nodata is not None else ""
        sources.append(
            f"""    <ComplexSource>
      <SourceFilename relativeToVRT="0">{escape(str(Path(p).resolve()))}</SourceFilename>
      <SourceBand>1</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{w}" ySize="{h}"/>
      <DstRect xOff="{round((t.c - minx) / res_x)}" yOff="{round((maxy - t.f) / res_y)}" xSize="{w}" ySize="{h}"/>
      {nd}
    </ComplexSource>"""
        )
    xml = f"""<VRTDataset rasterXSize="{width}" rasterYSize="{height}">
  <SRS>{escape(CRS.from_user_input(crs0).to_wkt())}</SRS>
  <GeoTransform>{minx!r}, {res_x!r}, 0, {maxy!r}, 0, {-res_y!r}</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <NoDataValue>nan</NoDataValue>
{chr(10).join(sources)}
  </VRTRasterBand>
</VRTDataset>
"""
    vrt_path = Path(vrt_path)
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    vrt_path.write_text(xml)
    return vrt_path


def chm_mosaic(
    boundary_gdf: gpd.GeoDataFrame,
    source: str,
    cache_dir: Path,
    name: str,
    chm_tiles_dir: str | None = None,
    chm_pattern: str = "*.tif",
    overwrite: bool = False,
    overlap: str = "latest",
    meta_version: str = "v1",
) -> list[Path]:
    """VRT layers over the CHM covering the boundary, from local tiles ("chm_tiles") or Meta ("meta").

    meta_version picks the Meta release (META_VERSIONS). With
    overlap="latest" a single VRT where later tiles win; with "max", one VRT
    per set of non-overlapping tiles on a shared grid, to be combined by
    per-pixel maximum.
    """
    if overlap not in ("latest", "max"):
        raise ValueError(f"overlap must be 'latest' or 'max', got {overlap!r}")
    if source == "chm_tiles":
        if not chm_tiles_dir:
            raise ValueError("tree_segmentation.source 'chm_tiles' needs data.chm_tiles_dir")
        paths = tiles_for_boundary(boundary_gdf, Path(chm_tiles_dir), chm_pattern, Path(cache_dir))
        if not paths:
            raise FileNotFoundError(f"No CHM tiles matching {chm_pattern} in {chm_tiles_dir} overlap the boundary")
    elif source == "meta":
        paths = download_meta_tiles(boundary_gdf, Path(cache_dir), overwrite=overwrite, version=meta_version)
        if not paths:
            raise FileNotFoundError("No Meta CHM tiles cover the boundary")
        source = f"meta_{meta_version}"  # versions get separate VRTs
    else:
        raise ValueError(f"Unknown CHM source '{source}', expected 'chm_tiles' or 'meta'")
    return mosaic_layers(paths, Path(cache_dir) / "vrt", f"{source}_{name}", overlap)


def mosaic_layers(paths: list[str], vrt_dir: Path, name: str, overlap: str = "latest") -> list[Path]:
    """VRT(s) over paths: one where later paths win ("latest"), or one per set of
    non-overlapping paths on a shared grid, to combine by per-pixel maximum ("max")."""
    if overlap not in ("latest", "max"):
        raise ValueError(f"overlap must be 'latest' or 'max', got {overlap!r}")
    if overlap == "latest":
        return [build_vrt(paths, Path(vrt_dir) / f"{name}.vrt")]
    layers = _non_overlapping_layers(paths)
    extent = _union_extent(paths)
    return [build_vrt(layer, Path(vrt_dir) / f"{name}_L{i}.vrt", extent=extent) for i, layer in enumerate(layers)]


def resolve_overlap(paths: list, overlap: str, out_dir: Path, bounds: tuple | None = None, chunk: int = 4096) -> list[str]:
    """Rasters covering `paths` that do not overlap each other, for consumers that sum per tile.

    Returns the paths unchanged when no two overlap. Otherwise composites them
    (overlap="latest": last path in sorted order wins; "max": per-pixel
    maximum) over their union extent, clipped to `bounds` (raster CRS), and
    writes it to out_dir as GeoTIFF chunks of `chunk` pixels; chunks with no
    data are skipped.
    """
    from .tree_segmentation import _read_layers  # lazy: tree_segmentation workers needn't import this module

    paths = sorted(str(p) for p in paths)
    if len(_non_overlapping_layers(paths)) <= 1:
        return paths
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.tif"):
        old.unlink()
    layers = [str(p) for p in mosaic_layers(paths, out_dir / "vrt", "composite", overlap)]
    with rasterio.open(layers[0]) as src:
        full = rasterio.windows.Window(0, 0, src.width, src.height)
        window = full
        if bounds is not None:
            window = rasterio.windows.from_bounds(*bounds, transform=src.transform).round_offsets().round_lengths().intersection(full)
        transform, crs = src.transform, src.crs
    out = []
    col0, row0, w, h = int(window.col_off), int(window.row_off), int(window.width), int(window.height)
    for r in range(row0, row0 + h, chunk):
        for c in range(col0, col0 + w, chunk):
            win = rasterio.windows.Window(c, r, min(chunk, col0 + w - c), min(chunk, row0 + h - r))
            z = _read_layers(layers, win)
            if not np.isfinite(z).any():
                continue
            path = out_dir / f"chunk_{r}_{c}.tif"
            with rasterio.open(
                path, "w", driver="GTiff", width=z.shape[1], height=z.shape[0], count=1, dtype="float32",
                crs=crs, transform=rasterio.windows.transform(win, transform), nodata=np.nan,
                compress="deflate", tiled=True, blockxsize=512, blockysize=512,
            ) as dst:
                dst.write(z, 1)
            out.append(str(path))
    logger.info(f"Composited {len(paths)} overlapping CHM tiles ({overlap}) into {len(out)} chunks")
    return out


def _bounds(path: str) -> tuple:
    with rasterio.open(path) as src:
        return tuple(src.bounds)


def _union_extent(paths: list[str]) -> tuple:
    b = [_bounds(p) for p in paths]
    return (min(x[0] for x in b), min(x[1] for x in b), max(x[2] for x in b), max(x[3] for x in b))


def _non_overlapping_layers(paths: list[str]) -> list[list[str]]:
    """Greedily split paths (in order) into layers whose rasters do not overlap (touching is fine)."""
    layers: list[tuple[list[str], list]] = []
    for p in paths:
        geom = box(*_bounds(p))
        for members, geoms in layers:
            if not any(geom.intersection(g).area > 0 for g in geoms):
                members.append(p)
                geoms.append(geom)
                break
        else:
            layers.append(([p], [geom]))
    return [members for members, _ in layers]


def boundary_bounds_in(raster_path: Path, boundary_gdf: gpd.GeoDataFrame, pad: float = 0.0) -> tuple:
    """Boundary bounds in the raster's CRS, padded by pad CRS units."""
    with rasterio.open(raster_path) as src:
        b = boundary_gdf.to_crs(src.crs).total_bounds
    return (b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad)

