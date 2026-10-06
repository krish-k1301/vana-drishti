"""Tests for gee.s1: relative-orbit selection, same-day dedupe and the mocked Earth Engine collection chain."""
import datetime as dt
from unittest.mock import MagicMock

import pytest

from gee import s1
from gee.s1 import Acquisition, orbit_dates, parse_acquisitions, select_orbit


def ms(day: dt.date, hour: int = 9) -> int:
    """Epoch milliseconds of a UTC date and hour."""
    return int(dt.datetime(day.year, day.month, day.day, hour, tzinfo=dt.timezone.utc).timestamp() * 1000)


def acq(day: dt.date, orbit: int, orbit_pass: str = "DESCENDING", hour: int = 9) -> Acquisition:
    """Shorthand acquisition."""
    return Acquisition(ms(day, hour), orbit, orbit_pass, "A")


def test_orbit_with_most_distinct_dates_wins_and_same_day_scenes_count_once() -> None:
    """The orbit with most distinct dates wins; same-day slices count once."""
    d = dt.date(2021, 1, 1)
    acqs = [acq(d, 10), acq(d, 10, hour=9), acq(d, 10, hour=9),  # three slices, same day, orbit 10
            acq(d + dt.timedelta(days=12), 10),
            acq(d, 83), acq(d + dt.timedelta(days=6), 83), acq(d + dt.timedelta(days=18), 83)]
    assert select_orbit(acqs, ["DESCENDING", "ASCENDING"]) == (83, "DESCENDING")
    assert orbit_dates(acqs, 10, "DESCENDING") == [d, d + dt.timedelta(days=12)]


def test_orbit_tie_break_by_pass_then_lowest_orbit() -> None:
    """Orbit ties break by preferred pass, then by lowest orbit number, independent of order."""
    d = dt.date(2021, 1, 1)
    tied = [acq(d, 155, "ASCENDING"), acq(d, 39, "DESCENDING"), acq(d, 112, "DESCENDING")]
    assert select_orbit(tied, ["DESCENDING", "ASCENDING"]) == (39, "DESCENDING")
    assert select_orbit(tied, ["ASCENDING", "DESCENDING"]) == (155, "ASCENDING")
    assert select_orbit(list(reversed(tied)), ["DESCENDING", "ASCENDING"]) == (39, "DESCENDING")


def test_select_orbit_empty_raises() -> None:
    """select_orbit with no acquisitions raises ValueError."""
    with pytest.raises(ValueError):
        select_orbit([], ["DESCENDING"])


def test_parse_acquisitions_and_utc_date() -> None:
    """parse_acquisitions reads orbit and pass and dates scenes in UTC."""
    rows = [[ms(dt.date(2022, 5, 3), 23), 68, "ASCENDING", "A"]]
    parsed = parse_acquisitions(rows)
    assert parsed[0].orbit == 68 and parsed[0].date == dt.date(2022, 5, 3)


def test_collection_filters_are_applied(monkeypatch) -> None:
    """s1_collection applies the instrument, resolution, platform and polarisation filters."""
    fake = MagicMock()
    monkeypatch.setattr(s1, "ee", fake)
    cfg = {"collection": "COPERNICUS/S1_GRD", "instrument_mode": "IW", "resolution_meters": 10,
           "platforms": ["A", "B"], "polarisations": ["VV", "VH"]}
    s1.s1_collection(cfg, "region", dt.date(2021, 1, 1), dt.date(2021, 3, 31))
    fake.ImageCollection.assert_called_once_with("COPERNICUS/S1_GRD")
    eq_calls = [c.args for c in fake.Filter.eq.call_args_list]
    assert ("instrumentMode", "IW") in eq_calls and ("resolution_meters", 10) in eq_calls
    fake.Filter.inList.assert_called_once_with("platform_number", ["A", "B"])
    assert [c.args for c in fake.Filter.listContains.call_args_list] == [
        ("transmitterReceiverPolarisation", "VV"), ("transmitterReceiverPolarisation", "VH")]
    chain = fake.ImageCollection.return_value.filterBounds.return_value
    chain.filterDate.assert_called_once_with("2021-01-01", "2021-04-01")


def test_stack_mosaics_one_image_per_date(monkeypatch) -> None:
    """s1_stack builds one mosaic per acquisition date with a one-day filter."""
    fake = MagicMock()
    monkeypatch.setattr(s1, "ee", fake)
    collection = MagicMock()
    dates = [dt.date(2021, 1, 1), dt.date(2021, 1, 13)]
    s1.s1_stack(collection, 10, "DESCENDING", dates, ["VV", "VH"], -9999.0)
    single = collection.filter.return_value.filter.return_value
    assert single.filterDate.call_count == 2
    single.filterDate.assert_any_call("2021-01-13", "2021-01-14")
    mosaic = single.filterDate.return_value.mosaic.return_value
    mosaic.select.assert_any_call(["VV", "VH"], ["t001_VV", "t001_VH"])
    mosaic.select.return_value.unmask.assert_called_with(-9999.0)
    assert len(fake.Image.cat.call_args.args[0]) == 2
    assert s1.s1_band_names(2, ["VV", "VH"]) == ["t000_VV", "t000_VH", "t001_VV", "t001_VH"]
