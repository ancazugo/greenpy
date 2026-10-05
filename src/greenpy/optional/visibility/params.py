"""Visibility parameters shared by the raster and vector engines."""

from dataclasses import dataclass

from ...config.schema import GreenPyConfig


@dataclass(frozen=True)
class VisibilityParams:
    # trees count within `buffer` metres of the footprint (as T3), with area and height above these
    buffer: float
    tree_area: float
    tree_height: float
    # observers: facade points every facade_spacing m, facade_offset m outside the wall,
    # at eye_height above each floor of storey_height
    facade_spacing: float = 5.0
    facade_offset: float = 0.5
    eye_height: float = 1.5
    storey_height: float = 3.0
    # targets: treetop + crown_points crown points at crown_point_height x tree height
    crown_points: int = 4
    crown_point_height: float = 2 / 3
    # raster engine
    resolution: float = 1.0
    vegetation: str = "auto"
    mask_chm_buildings: bool = True
    tile_size: float = 2000.0
    # metres ignored at both sightline ends (None = resolution)
    end_skip: float | None = None

    @property
    def skip(self) -> float:
        return self.resolution if self.end_skip is None else self.end_skip

    @classmethod
    def from_cfg(cls, cfg: GreenPyConfig, buffer: float, tree_area: float, tree_height: float) -> "VisibilityParams":
        v = cfg.visibility
        return cls(
            buffer=float(buffer), tree_area=float(tree_area), tree_height=float(tree_height),
            facade_spacing=v.facade_spacing, facade_offset=v.facade_offset, eye_height=v.eye_height,
            storey_height=cfg.heights.storey_height, crown_points=v.crown_points,
            crown_point_height=v.crown_point_height, resolution=v.resolution, vegetation=v.vegetation,
            mask_chm_buildings=v.mask_chm_buildings, tile_size=v.tile_size, end_skip=v.end_skip,
        )
