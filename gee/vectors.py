"""Local DETER / PRODES vector handling: reading, filtering and per-patch rasterisation (no Earth Engine)."""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import box

from gee.dates import NO_DATE, parse_date, to_days
from gee.grid import PatchGrid, lonlat_bounds, rasterize


def read_deter(path: str, date_attr: str, class_attr: str) -> gpd.GeoDataFrame:
    """Read a DETER file into columns event_date, event_class, geometry (EPSG:4326)."""
    raw = gpd.read_file(path)
    for attr in (date_attr, class_attr):
        if attr not in raw.columns:
            raise KeyError(f"DETER attribute '{attr}' not found in {path}; columns: {list(raw.columns)}")
    out = gpd.GeoDataFrame({
        "event_date": [parse_date(v) for v in raw[date_attr]],
        "event_class": raw[class_attr].astype(str).to_numpy(),
    }, geometry=raw.geometry.to_numpy(), crs=raw.crs)
    return out.to_crs(4326)


def filter_events(events: gpd.GeoDataFrame, classes: list[str], start: dt.date, end: dt.date,
                  bbox: list[float] | None = None) -> gpd.GeoDataFrame:
    """Keep events of the given classes dated in [start, end], optionally inside a lon/lat bbox."""
    keep = events["event_class"].isin(classes) & events["event_date"].between(start, end)
    selected = events[keep]
    if bbox is not None:
        selected = selected[selected.geometry.within(box(*bbox))]
    return selected.sort_values(["event_date", "event_class"], kind="stable")


def read_prodes(path: str, year_attr: str) -> gpd.GeoDataFrame:
    """Read a PRODES yearly-deforestation file into columns prodes_year, geometry (EPSG:4326)."""
    raw = gpd.read_file(path)
    if year_attr not in raw.columns:
        raise KeyError(f"PRODES attribute '{year_attr}' not found in {path}; columns: {list(raw.columns)}")
    years = raw[year_attr].astype(float).round().astype(int).to_numpy()
    return gpd.GeoDataFrame({"prodes_year": years}, geometry=raw.geometry.to_numpy(), crs=raw.crs).to_crs(4326)


def prodes_year_end(year: int, end_month: int, end_day: int) -> dt.date:
    """Return the last day covered by a PRODES year (PRODES year Y runs 1 Aug Y-1 to 31 Jul Y)."""
    return dt.date(year, end_month, end_day)


def intersecting(gdf: gpd.GeoDataFrame, grid: PatchGrid) -> gpd.GeoDataFrame:
    """Return rows whose geometry intersects the grid footprint."""
    footprint = box(*lonlat_bounds(grid))
    idx = gdf.sindex.query(footprint, predicate="intersects")
    return gdf.iloc[np.sort(idx)]


def reference_days(events: gpd.GeoDataFrame, grid: PatchGrid) -> np.ndarray:
    """Return int32 [H, W] earliest event day (days since epoch) per pixel over intersecting events, -1 if none."""
    out = np.full((grid.size, grid.size), NO_DATE, dtype=np.int32)
    for row in intersecting(events, grid).sort_values("event_date", kind="stable").itertuples():
        mask = rasterize([row.geometry], grid) & (out == NO_DATE)
        out[mask] = to_days(row.event_date)
    return out


def prior_mask(prodes: gpd.GeoDataFrame, grid: PatchGrid, cutoff: dt.date, end_month: int,
               end_day: int) -> np.ndarray:
    """Return bool [H, W] pixels deforested in PRODES years that ended before `cutoff`."""
    rows = intersecting(prodes, grid)
    done = [g for g, y in zip(rows.geometry, rows["prodes_year"]) if prodes_year_end(y, end_month, end_day) < cutoff]
    return rasterize(done, grid)


def prodes_year_of(prodes: gpd.GeoDataFrame, geom: object) -> int:
    """Return the PRODES year of the polygon overlapping `geom` the most (by area), -1 if none overlaps."""
    idx = prodes.sindex.query(geom, predicate="intersects")
    if len(idx) == 0:
        return -1
    rows = prodes.iloc[idx]
    overlaps = shapely.area(shapely.intersection(rows.geometry.to_numpy(), geom))
    best = int(np.argmax(overlaps))
    return int(rows["prodes_year"].to_numpy()[best]) if overlaps[best] > 0 else -1
