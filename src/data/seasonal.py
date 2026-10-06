"""Seasonal acquisition window (PRD 4.3 `seasonal_window`): keep only images inside a fixed calendar window."""
from __future__ import annotations

import datetime as dt
from collections.abc import Mapping, Sequence

SPEC_KEYS = ("enabled", "start", "end", "on_empty")
ON_EMPTY = ("error", "skip")
LEAP_YEAR = 2000


class EmptySeasonalWindowError(ValueError):
    """Raised when no acquisition of a sample falls inside the seasonal window."""


def parse_month_day(text: str) -> tuple[int, int]:
    """Parse 'MM-DD' into (month, day), validating it against a leap year."""
    month, day = (int(part) for part in str(text).split("-"))
    dt.date(LEAP_YEAR, month, day)
    return month, day


class SeasonalWindow:
    """Calendar window [start, end] given as 'MM-DD'; start > end wraps the year end (e.g. 11-01 .. 03-31).

    Only images are filtered; label dates and masks stay intact. `on_empty` says what happens when a
    sample has no image inside the window: `error` raises EmptySeasonalWindowError, `skip` drops the sample
    when the dataset is built.
    """

    def __init__(self, start: str, end: str, on_empty: str) -> None:
        """Parse the bounds and validate the empty-window policy."""
        if on_empty not in ON_EMPTY:
            raise ValueError(f"on_empty must be one of {ON_EMPTY}, got '{on_empty}'")
        self.start, self.end, self.on_empty = parse_month_day(start), parse_month_day(end), on_empty

    def contains(self, day: dt.date) -> bool:
        """True if the calendar day of `day` lies inside the window (bounds included)."""
        key = (day.month, day.day)
        if self.start <= self.end:
            return self.start <= key <= self.end
        return key >= self.start or key <= self.end

    def keep_indices(self, dates: Sequence[dt.date]) -> list[int]:
        """Indices of the dates inside the window."""
        return [i for i, day in enumerate(dates) if self.contains(day)]

    def apply(self, raw: dict, name: str = "sample") -> dict:
        """Raw sample dict with only in-window images; raises if none is left."""
        keep = self.keep_indices(raw["image_dates"])
        if not keep:
            raise EmptySeasonalWindowError(f"{name}: no acquisition inside the seasonal window "
                                           f"{self.start}..{self.end}; set on_empty: skip to drop such samples")
        out = dict(raw)
        out["image"] = raw["image"][keep]
        out["image_dates"] = [raw["image_dates"][i] for i in keep]
        return out


def build_seasonal_window(spec: Mapping | None) -> SeasonalWindow | None:
    """SeasonalWindow from a config block, or None if the block is absent or disabled."""
    if spec is None:
        return None
    missing = [k for k in SPEC_KEYS if k not in spec]
    if missing:
        raise KeyError(f"seasonal_window needs keys {missing}")
    if not spec["enabled"]:
        return None
    return SeasonalWindow(spec["start"], spec["end"], spec["on_empty"])
