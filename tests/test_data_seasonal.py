"""Seasonal acquisition window: calendar logic, label preservation, empty-window policies, order of transforms."""
from __future__ import annotations

import datetime as dt

import pytest
import torch

from src.data import BraDDDataset, SeasonalWindow
from src.data.seasonal import EmptySeasonalWindowError


def _window(start, end, on_empty="error"):
    """A complete enabled seasonal_window block."""
    return {"enabled": True, "start": start, "end": end, "on_empty": on_empty}


def test_calendar_window_with_year_wrap() -> None:
    """A window starting after it ends wraps the year end; bounds are inclusive."""
    wrap = SeasonalWindow("11-01", "03-31", "error")
    assert wrap.contains(dt.date(2020, 12, 15)) and wrap.contains(dt.date(2020, 2, 29))
    assert wrap.contains(dt.date(2021, 3, 31)) and not wrap.contains(dt.date(2021, 6, 1))
    plain = SeasonalWindow("06-01", "09-30", "skip")
    assert plain.contains(dt.date(2021, 6, 1)) and not plain.contains(dt.date(2021, 10, 1))
    with pytest.raises(ValueError):
        SeasonalWindow("02-30", "03-01", "error")
    with pytest.raises(ValueError):
        SeasonalWindow("01-01", "03-01", "drop")


def test_images_filtered_labels_intact(bradd_root) -> None:
    """Only in-window images remain; label masks and label dates are untouched."""
    ds = BraDDDataset(bradd_root, "train", normalization="none", seasonal_window=_window("06-01", "09-30"))
    plain = BraDDDataset(bradd_root, "train", normalization="none")
    for i in range(len(ds)):
        raw, full = ds.load_raw(i), plain.load_raw(i)
        assert all(6 <= d.month <= 9 for d in raw["image_dates"]) and 0 < len(raw["image_dates"]) < len(
            full["image_dates"])
        item = ds[i]
        assert torch.equal(item["Targets"], full["label"]) and raw["label_dates"] == full["label_dates"]
        origin = min(raw["image_dates"] + raw["label_dates"])
        assert item["TargetDays"].tolist() == [(d - origin).days + 1 for d in full["label_dates"]]


def test_skip_and_error_policies(bradd_root) -> None:
    """'skip' drops samples with no in-window image at build time; 'error' raises when such a sample is read."""
    plain = BraDDDataset(bradd_root, "train", normalization="none")
    day = plain.load_raw(0)["image_dates"][1]
    md = f"{day.month:02d}-{day.day:02d}"
    has_day = [any((d.month, d.day) == (day.month, day.day) for d in plain.load_raw(i)["image_dates"])
               for i in range(len(plain))]
    skip = BraDDDataset(bradd_root, "train", normalization="none", seasonal_window=_window(md, md, "skip"))
    assert len(skip) == sum(has_day) and len(skip.skipped) == len(has_day) - sum(has_day)
    assert all(len(skip[i]["ImageDays"]) >= 1 for i in range(len(skip)))
    error = BraDDDataset(bradd_root, "train", normalization="none", seasonal_window=_window(md, md))
    assert len(error) == len(plain)
    if not all(has_day):
        with pytest.raises(EmptySeasonalWindowError, match="seasonal window"):
            error[has_day.index(False)]


def test_window_applied_before_subsample(bradd_root) -> None:
    """The temporal subsample picks among the in-window dates only."""
    spec = _window("06-01", "09-30")
    ds = BraDDDataset(bradd_root, "train", normalization="none", seasonal_window=spec,
                      temporal_subsample={"mode": "uniform", "num_dates": 3})
    raw = ds.load_raw(0)
    origin = min(raw["image_dates"] + raw["label_dates"])
    in_window = [(d - origin).days + 1 for d in raw["image_dates"]]
    days = ds[0]["ImageDays"].tolist()
    assert len(days) == min(3, len(in_window)) and set(days) <= set(in_window)
    assert days[0] == in_window[0] and days[-1] == in_window[-1]


def test_missing_keys_rejected(bradd_root) -> None:
    """An enabled block must spell out every key."""
    with pytest.raises(KeyError):
        BraDDDataset(bradd_root, "train", normalization="none", seasonal_window={"enabled": True, "start": "01-01"})
