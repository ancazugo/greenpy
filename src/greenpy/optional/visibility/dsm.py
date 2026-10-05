"""Surface model for the raster engine: ground, buildings and vegetation on one grid, in absolute elevation.

Ground is the DTM warped bilinearly onto the grid (0 everywhere without
terrain). Each building is burned flat at the ground under its
representative point plus its height (per-pixel maximum, pixel-centre
rule). Vegetation is the canopy height model warped onto the grid, or the
tree crowns burned at their heights when no CHM is configured, on top of the
ground. crown_id marks each target tree's crown so the kernel can ignore the
tree it is looking at.
"""

import math
from dataclasses import dataclass

import numpy as np
import rasterio
import rasterio.features
import rasterio.vrt
from affine import Affine
from rasterio.enums import Resampling


@dataclass
class DSM:
    dsm: np.ndarray  # float32: max(buildings, ground + vegetation), absolute
    bldg: np.ndarray  # float32: buildings, else ground (what still blocks inside a target's crown)
    crown_id: np.ndarray  # int32 (layers, rows, cols): target index + 1 inside target crowns, else 0
    x0: float
    y0: float
    res: float
    ground: np.ndarray | None = None  # float32 ground elevation; None = flat (0)

    @property
    def terrain(self) -> bool:
        return self.ground is not None

    def ground_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Ground elevation at points (nearest pixel, clamped to the grid); 0 without terrain."""
        if self.ground is None:
            return np.zeros(len(x))
        return sample_grid(self.ground, self.x0, self.y0, self.res, x, y)


def sample_grid(grid: np.ndarray, x0: float, y0: float, res: float, x, y) -> np.ndarray:
    """Values of a north-up grid at points (nearest pixel, clamped to the grid)."""
    rows, cols = grid.shape
    c = np.clip(np.floor((np.asarray(x) - x0) / res).astype(np.int64), 0, cols - 1)
    r = np.clip(np.floor((y0 - np.asarray(y)) / res).astype(np.int64), 0, rows - 1)
    return grid[r, c].astype(np.float64)

    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0, self.x0, 0, -self.res, self.y0)


def grid_for(bounds: tuple, res: float) -> tuple[float, float, int, int]:
    """(x0, y0, width, height) of a grid aligned to multiples of res covering bounds."""
    minx, miny, maxx, maxy = bounds
    x0, y0 = math.floor(minx / res) * res, math.ceil(maxy / res) * res
    width = max(1, math.ceil((maxx - x0) / res))
    height = max(1, math.ceil((y0 - miny) / res))
    return x0, y0, width, height


def _burn_max(geoms, values, shape, transform, keep=None) -> np.ndarray:
    """Rasterise geometries keeping the per-pixel maximum value (shapes drawn in ascending order).

    Only geometries where `keep` is True are drawn (default: positive values).
    """
    out = np.zeros(shape, np.float32)
    if len(geoms) == 0:
        return out
    keep = values > 0 if keep is None else keep
    order = np.argsort(values, kind="stable")
    shapes = ((geoms[i], float(values[i])) for i in order if keep[i])
    return rasterio.features.rasterize(shapes, out=out, transform=transform, dtype="float32")


def read_dem(layers: list, crs: str, transform: Affine, shape: tuple) -> np.ndarray:
    """Ground elevation from DEM rasters warped bilinearly onto the grid; gaps filled with the median."""
    ground = np.full(shape, np.nan, np.float32)
    for path in layers:
        with rasterio.open(path) as src:
            with rasterio.vrt.WarpedVRT(
                src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                resampling=Resampling.bilinear, src_nodata=src.nodata if src.nodata is not None else np.nan,
                nodata=np.nan,
            ) as vrt:
                ground = np.fmax(ground, vrt.read(1, out_dtype="float32"))
    if np.isnan(ground).all():
        raise ValueError("The DEM has no data over this area — check terrain.source")
    ground[np.isnan(ground)] = np.nanmedian(ground)
    return ground


def read_chm(layers: list, crs: str, transform: Affine, shape: tuple, res: float) -> np.ndarray:
    """Canopy heights from CHM layers (shared grid; per-pixel max across layers) warped onto the grid; no data -> 0."""
    veg = np.zeros(shape, np.float32)
    for path in layers:
        with rasterio.open(path) as src:
            src_res = abs(src.transform.a)
            if src.crs is not None and src.crs.is_projected:
                from ..tree_segmentation import ground_scale
                cx, cy = src.xy(src.height // 2, src.width // 2)
                src_res *= ground_scale(src.crs, cx, cy)
            resampling = Resampling.nearest if res <= 1.5 * src_res else Resampling.max
            with rasterio.vrt.WarpedVRT(
                src, crs=crs, transform=transform, width=shape[1], height=shape[0],
                resampling=resampling, src_nodata=src.nodata if src.nodata is not None else np.nan, nodata=np.nan,
            ) as vrt:
                layer = vrt.read(1, out_dtype="float32")
        veg = np.fmax(veg, np.nan_to_num(layer, nan=0.0))
    veg[veg < 0] = 0
    return veg


def build_dsm(
    bounds: tuple, res: float, crs: str,
    bldg_geoms: np.ndarray, bldg_h: np.ndarray,
    target_crowns: np.ndarray,
    veg_geoms: np.ndarray | None = None, veg_h: np.ndarray | None = None,
    chm_layers: list | None = None, mask_chm_buildings: bool = True, dem_layers: list | None = None,
) -> DSM:
    """Surface over bounds: ground from dem_layers (flat when None), buildings, and vegetation
    from chm_layers (if given) else from veg_geoms/veg_h crowns."""
    x0, y0, width, height = grid_for(bounds, res)
    transform = Affine(res, 0, x0, 0, -res, y0)
    shape = (height, width)
    bldg_geoms = np.asarray(bldg_geoms)
    bldg_h = np.asarray(bldg_h, dtype=float)

    ground = read_dem(dem_layers, crs, transform, shape) if dem_layers else None
    # flat roofs at the ground under each building's representative point plus its height
    base = np.zeros(len(bldg_geoms))
    if ground is not None and len(bldg_geoms):
        import shapely
        rep = shapely.point_on_surface(bldg_geoms)
        base = sample_grid(ground, x0, y0, res, shapely.get_x(rep), shapely.get_y(rep))
    bldg_rel = _burn_max(bldg_geoms, bldg_h, shape, transform)
    is_bldg = bldg_rel > 0
    tops = _burn_max(bldg_geoms, bldg_h + base, shape, transform, keep=bldg_h > 0)
    g = ground if ground is not None else np.zeros(shape, np.float32)

    if chm_layers:
        veg = read_chm(chm_layers, crs, transform, shape, res)
        if mask_chm_buildings:
            veg[is_bldg] = 0  # CHMs pick up roofs; buildings come from footprints
    else:
        veg = _burn_max(np.asarray(veg_geoms), np.asarray(veg_h, dtype=float), shape, transform) \
            if veg_geoms is not None and len(veg_geoms) else np.zeros(shape, np.float32)
    veg_abs = g + veg
    bldg = np.where(is_bldg, tops, g).astype(np.float32)
    dsm = np.where(is_bldg, np.maximum(tops, veg_abs), veg_abs).astype(np.float32)

    layers = crown_layers(np.asarray(target_crowns))
    crown_id = np.zeros((max(1, len(layers)), *shape), np.int32)
    for k, members in enumerate(layers):
        rasterio.features.rasterize(
            ((target_crowns[i], i + 1) for i in members), out=crown_id[k], transform=transform, dtype="int32",
        )
    return DSM(dsm, bldg, crown_id, x0, y0, res, ground)


def crown_layers(crowns: np.ndarray) -> list[list[int]]:
    """Split crown indices into layers of crowns that do not overlap (greedy colouring).

    Crowns that merely touch (segmented neighbours) share a layer; ones whose
    interiors overlap, or that contain one another (duplicates, nested
    crowns), do not.
    """
    if len(crowns) == 0:
        return []
    import shapely

    tree = shapely.STRtree(crowns)
    a1, b1 = tree.query(crowns, predicate="overlaps")
    a2, b2 = tree.query(crowns, predicate="contains")
    a, b = np.concatenate([a1, a2, b2]), np.concatenate([b1, b2, a2])
    keep = a < b
    neighbours: dict[int, list[int]] = {}
    for i, j in zip(b[keep], a[keep]):  # j < i: colour i after its lower-index neighbours
        neighbours.setdefault(int(i), []).append(int(j))
    colour = np.zeros(len(crowns), np.int64)
    for i in range(len(crowns)):
        used = {colour[j] for j in neighbours.get(i, ())}
        c = 0
        while c in used:
            c += 1
        colour[i] = c
    return [np.flatnonzero(colour == c).tolist() for c in range(colour.max() + 1)]
