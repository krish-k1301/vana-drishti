"""Sentinel-1 GRD collection, single-relative-orbit selection and same-day mosaics.

GEE `COPERNICUS/S1_GRD` is already in dB after the default chain (thermal noise removal, radiometric
calibration, terrain correction), as in BraDD-S1TS. No speckle filter is applied on purpose.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

import ee

ACQ_PROPERTIES = ["system:time_start", "relativeOrbitNumber_start", "orbitProperties_pass", "platform_number"]


@dataclass(frozen=True)
class Acquisition:
    """One S1 GRD scene's timing and orbit metadata."""

    time_ms: int
    orbit: int
    orbit_pass: str
    platform: str

    @property
    def date(self) -> dt.date:
        """UTC acquisition date."""
        return dt.datetime.fromtimestamp(self.time_ms / 1000.0, tz=dt.timezone.utc).date()


def parse_acquisitions(rows: list[list]) -> list[Acquisition]:
    """Parse rows of [time_ms, relative orbit, pass, platform] into Acquisitions."""
    return [Acquisition(int(r[0]), int(r[1]), str(r[2]), str(r[3])) for r in rows]


def select_orbit(acquisitions: list[Acquisition], pass_preference: list[str]) -> tuple[int, str]:
    """Pick the (relative orbit, pass) with most distinct dates; ties by pass preference, then lowest orbit."""
    if not acquisitions:
        raise ValueError("no Sentinel-1 acquisitions over the patch in the window")
    dates: dict[tuple[int, str], set[dt.date]] = defaultdict(set)
    for acq in acquisitions:
        dates[(acq.orbit, acq.orbit_pass)].add(acq.date)

    def rank(key: tuple[int, str]) -> tuple[int, int, int]:
        """Sort key: more dates first, preferred pass first, lower orbit first."""
        orbit, orbit_pass = key
        pass_rank = pass_preference.index(orbit_pass) if orbit_pass in pass_preference else len(pass_preference)
        return -len(dates[key]), pass_rank, orbit

    return min(dates, key=rank)


def orbit_dates(acquisitions: list[Acquisition], orbit: int, orbit_pass: str) -> list[dt.date]:
    """Return sorted distinct acquisition dates of one relative orbit and pass (same-day scenes deduplicated)."""
    return sorted({a.date for a in acquisitions if a.orbit == orbit and a.orbit_pass == orbit_pass})


def orbit_platforms(acquisitions: list[Acquisition], orbit: int, orbit_pass: str) -> list[str]:
    """Return sorted distinct platforms (A/B/C) contributing to one orbit's acquisitions."""
    return sorted({a.platform for a in acquisitions if a.orbit == orbit and a.orbit_pass == orbit_pass})


def s1_collection(s1_cfg: dict, region: object, start: dt.date, end: dt.date) -> object:
    """Return the filtered S1 GRD ImageCollection (IW, 10 m, configured platforms, all polarisations present)."""
    col = (ee.ImageCollection(s1_cfg["collection"])
           .filterBounds(region)
           .filterDate(start.isoformat(), (end + dt.timedelta(days=1)).isoformat())
           .filter(ee.Filter.eq("instrumentMode", s1_cfg["instrument_mode"]))
           .filter(ee.Filter.eq("resolution_meters", s1_cfg["resolution_meters"]))
           .filter(ee.Filter.inList("platform_number", s1_cfg["platforms"])))
    for pol in s1_cfg["polarisations"]:
        col = col.filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
    return col.select(s1_cfg["polarisations"])


def fetch_acquisitions(collection: object) -> list[Acquisition]:
    """Fetch scene metadata of a collection to the client in one request."""
    rows = collection.reduceColumns(ee.Reducer.toList(len(ACQ_PROPERTIES)), ACQ_PROPERTIES).get("list").getInfo()
    return parse_acquisitions(rows or [])


def s1_band_names(n_dates: int, polarisations: list[str]) -> list[str]:
    """Return stacked band names t000_VV, t000_VH, t001_VV, ... in time-major order."""
    return [f"t{i:03d}_{pol}" for i in range(n_dates) for pol in polarisations]


def s1_stack(collection: object, orbit: int, orbit_pass: str, dates: list[dt.date], polarisations: list[str],
             nodata: float) -> object:
    """Mosaic same-date scenes of one orbit per date and stack them into one multi-band image."""
    single = (collection.filter(ee.Filter.eq("relativeOrbitNumber_start", orbit))
              .filter(ee.Filter.eq("orbitProperties_pass", orbit_pass)))
    images = []
    for i, day in enumerate(dates):
        names = [f"t{i:03d}_{pol}" for pol in polarisations]
        day_mosaic = single.filterDate(day.isoformat(), (day + dt.timedelta(days=1)).isoformat()).mosaic()
        images.append(day_mosaic.select(polarisations, names).unmask(nodata).toFloat())
    return ee.Image.cat(images)
