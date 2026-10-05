"""Surface model for the raster engine: building heights plus vegetation on one grid, ground at 0.

Buildings are burned at their height (per-pixel maximum, pixel-centre rule).
Vegetation is the canopy height model warped onto the grid, or the tree
crowns burned at their heights when no CHM is configured. Ground is flat
(z = 0): terrain is not modelled. crown_id marks each target tree's crown so
the kernel can ignore the tree it is looking at.
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
    dsm: np.ndarray  # float32: max(buildings, vegetation)
    bldg: np.ndarray  # float32: buildings only (what still blocks inside a target's crown)
    crown_id: np.ndarray  # int32: target index + 1 inside target crowns, else 0
    x0: float
    y0: float
    res: float

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


def _burn_max(geoms, values, shape, transform) -> np.ndarray:
    """Rasterise geometries keeping the per-pixel maximum value (shapes drawn in ascending order)."""
    out = np.zeros(shape, np.float32)
    if len(geoms) == 0:
        return out
    order = np.argsort(values, kind="stable")
    shapes = ((geoms[i], float(values[i])) for i in order if values[i] > 0)
    return rasterio.features.rasterize(shapes, out=out, transform=transform, dtype="float32")


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
    chm_layers: list | None = None, mask_chm_buildings: bool = True,
) -> DSM:
    """Surface over bounds: buildings, plus vegetation from chm_layers (if given) else from veg_geoms/veg_h crowns."""
    x0, y0, width, height = grid_for(bounds, res)
    transform = Affine(res, 0, x0, 0, -res, y0)
    shape = (height, width)

    bldg = _burn_max(np.asarray(bldg_geoms), np.asarray(bldg_h, dtype=float), shape, transform)
    if chm_layers:
        veg = read_chm(chm_layers, crs, transform, shape, res)
        if mask_chm_buildings:
            veg[bldg > 0] = 0  # CHMs pick up roofs; buildings come from footprints
    else:
        veg = _burn_max(np.asarray(veg_geoms), np.asarray(veg_h, dtype=float), shape, transform) \
            if veg_geoms is not None and len(veg_geoms) else np.zeros(shape, np.float32)

    crown_id = np.zeros(shape, np.int32)
    if len(target_crowns):
        crown_id = rasterio.features.rasterize(
            ((g, i + 1) for i, g in enumerate(target_crowns)), out=crown_id, transform=transform, dtype="int32",
        )
    return DSM(np.maximum(bldg, veg), bldg, crown_id, x0, y0, res)
