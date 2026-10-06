"""Pure patch-grid geometry: UTM 10 m pixel grids, footprints, polygon rasterisation and areas."""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np
import shapely
from pyproj import Transformer
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


@dataclass(frozen=True)
class PatchGrid:
    """North-up square pixel grid in a UTM CRS; (x0, y0) is the upper-left corner."""

    epsg: int
    x0: float
    y0: float
    size: int
    pixel_m: float

    @property
    def crs(self) -> str:
        """CRS code string, e.g. 'EPSG:32721'."""
        return f"EPSG:{self.epsg}"

    @property
    def affine(self) -> list[float]:
        """Earth Engine crsTransform [scaleX, shearX, translateX, shearY, scaleY, translateY]."""
        return [self.pixel_m, 0.0, self.x0, 0.0, -self.pixel_m, self.y0]

    @property
    def bounds_xy(self) -> tuple[float, float, float, float]:
        """(xmin, ymin, xmax, ymax) in the grid CRS."""
        extent = self.size * self.pixel_m
        return self.x0, self.y0 - extent, self.x0 + extent, self.y0

    def compute_pixels_grid(self) -> dict:
        """Grid dict for `ee.data.computePixels`."""
        scale_x, shear_x, trans_x, shear_y, scale_y, trans_y = self.affine
        return {
            "dimensions": {"width": self.size, "height": self.size},
            "affineTransform": {"scaleX": scale_x, "shearX": shear_x, "translateX": trans_x,
                                "shearY": shear_y, "scaleY": scale_y, "translateY": trans_y},
            "crsCode": self.crs,
        }

    def pixel_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (xs, ys) arrays of shape [size, size] with pixel-centre coordinates."""
        offsets = (np.arange(self.size) + 0.5) * self.pixel_m
        xs, ys = np.meshgrid(self.x0 + offsets, self.y0 - offsets)
        return xs, ys


def utm_epsg(lon: float, lat: float) -> int:
    """Return the WGS84/UTM EPSG code of the zone containing (lon, lat)."""
    zone = min(int((lon + 180.0) // 6.0) + 1, 60)
    return (32600 if lat >= 0 else 32700) + zone


@functools.lru_cache(maxsize=64)
def _transformer(src: int, dst: int) -> Transformer:
    """Cached always-xy transformer between two EPSG codes."""
    return Transformer.from_crs(src, dst, always_xy=True)


def make_grid(lon: float, lat: float, size: int, pixel_m: float) -> PatchGrid:
    """Return the grid of `size` x `size` pixels centred on (lon, lat), snapped to whole pixels."""
    epsg = utm_epsg(lon, lat)
    x, y = _transformer(4326, epsg).transform(lon, lat)
    half = size * pixel_m / 2.0
    x0 = round((x - half) / pixel_m) * pixel_m
    y0 = round((y + half) / pixel_m) * pixel_m
    return PatchGrid(epsg=epsg, x0=float(x0), y0=float(y0), size=size, pixel_m=pixel_m)


def to_grid_crs(geom: BaseGeometry, epsg: int) -> BaseGeometry:
    """Project a lon/lat geometry to the given EPSG code."""
    return shapely_transform(_transformer(4326, epsg).transform, geom)


def lonlat_bounds(grid: PatchGrid) -> tuple[float, float, float, float]:
    """Return (lon_min, lat_min, lon_max, lat_max) enclosing the grid footprint."""
    xmin, ymin, xmax, ymax = grid.bounds_xy
    lons, lats = _transformer(grid.epsg, 4326).transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    return min(lons), min(lats), max(lons), max(lats)


def rasterize(geoms_lonlat: list[BaseGeometry], grid: PatchGrid) -> np.ndarray:
    """Return a bool [size, size] mask of pixels whose centre lies inside any of the lon/lat geometries."""
    mask = np.zeros((grid.size, grid.size), dtype=bool)
    if not geoms_lonlat:
        return mask
    projected = to_grid_crs(shapely.union_all(geoms_lonlat), grid.epsg)
    xs, ys = grid.pixel_centres()
    return np.asarray(shapely.contains_xy(projected, xs, ys), dtype=bool)


def lonlat_to_rowcol(lons: np.ndarray, lats: np.ndarray, grid: PatchGrid) -> tuple[np.ndarray, np.ndarray]:
    """Map lon/lat points to integer (row, col) pixel indices of the grid."""
    xs, ys = _transformer(4326, grid.epsg).transform(np.asarray(lons, dtype=float).tolist(),
                                                     np.asarray(lats, dtype=float).tolist())
    cols = np.floor((np.asarray(xs) - grid.x0) / grid.pixel_m).astype(int)
    rows = np.floor((grid.y0 - np.asarray(ys)) / grid.pixel_m).astype(int)
    return rows, cols


def area_ha(geom_lonlat: BaseGeometry) -> float:
    """Return the planar area in hectares of a lon/lat geometry, measured in its local UTM zone."""
    centre = geom_lonlat.centroid
    return float(to_grid_crs(geom_lonlat, utm_epsg(centre.x, centre.y)).area / 10_000.0)
