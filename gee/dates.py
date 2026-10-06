"""Pure date helpers: day encoding, event windows, label dates, RADD and burn-month decoding, revisit stats."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

EPOCH = dt.date(1970, 1, 1)
NO_DATE = -1


def to_days(day: dt.date) -> int:
    """Return days since 1970-01-01 for a date."""
    return (day - EPOCH).days


def from_days(days: int) -> dt.date:
    """Return the date that is `days` after 1970-01-01."""
    return EPOCH + dt.timedelta(days=int(days))


def parse_date(value: object) -> dt.date:
    """Parse a date, datetime or ISO-like string ('YYYY-MM-DD...') into a date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


@dataclass(frozen=True)
class Window:
    """Closed date interval [start, end] of one patch time series."""

    start: dt.date
    end: dt.date

    def contains(self, day: dt.date) -> bool:
        """Return True if `day` lies inside the window (inclusive)."""
        return self.start <= day <= self.end

    @property
    def length_days(self) -> int:
        """Number of days covered, inclusive of both ends."""
        return (self.end - self.start).days + 1

    @property
    def years(self) -> list[int]:
        """Calendar years overlapped by the window."""
        return list(range(self.start.year, self.end.year + 1))


def compute_window(event_date: dt.date, days_before: int, days_after: int) -> Window:
    """Return the window from `days_before` days before to `days_after` days after the event date."""
    if days_before < 0 or days_after < 0:
        raise ValueError("window offsets must be non-negative")
    return Window(event_date - dt.timedelta(days=days_before), event_date + dt.timedelta(days=days_after))


def monthly_label_dates(window: Window) -> list[dt.date]:
    """Return the window start followed by the first day of every later month inside the window."""
    dates = [window.start]
    year, month = window.start.year, window.start.month
    while True:
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        first = dt.date(year, month, 1)
        if first > window.end:
            break
        dates.append(first)
    if len(dates) < 2:
        dates.append(window.end)
    return dates


def decode_radd_yyddd(values: np.ndarray, base_year: int = 2000) -> np.ndarray:
    """Decode RADD `Date` values (YYDDD: 2-digit year since base_year, day of year) to days since epoch, -1 if none."""
    vals = np.asarray(values).astype(np.int64)
    years = vals // 1000
    doy = vals % 1000
    out = np.full(vals.shape, NO_DATE, dtype=np.int32)
    valid = (vals > 0) & (doy >= 1) & (doy <= 366)
    for yy in np.unique(years[valid]):
        sel = valid & (years == yy)
        jan1 = to_days(dt.date(base_year + int(yy), 1, 1))
        out[sel] = jan1 + doy[sel] - 1
    return out


def first_burn_day(monthly: dict[int, np.ndarray], window: Window) -> np.ndarray:
    """Causal day (days since epoch) of each pixel's earliest burn month: the LAST day of that month, -1 if none.

    Resolution is one month and the burn may have happened up to ~30 days before the returned day, never after it.
    A burn month that starts before window.start is pre-existing and encoded as window.start - 1 day (always
    "already burned"); months after window.end are ignored. `monthly` maps calendar year to an array of burn
    months (1-12, 0 = unburned); only the pulled years (the window's calendar years) are seen.
    """
    shape = next(iter(monthly.values())).shape
    out = np.full(shape, NO_DATE, dtype=np.int32)
    pre_existing = to_days(window.start) - 1
    for year in sorted(monthly):
        months = np.asarray(monthly[year]).astype(np.int64)
        for month in range(1, 13):
            first = dt.date(year, month, 1)
            last = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
            if first > window.end:
                continue
            sel = (months == month) & (out == NO_DATE)
            out[sel] = pre_existing if first < window.start else to_days(last)
    return out


def year_end_days(years: np.ndarray) -> np.ndarray:
    """Days since epoch of 31 December of each calendar year (int32); -1 where year <= 0."""
    years = np.asarray(years).astype(np.int64)
    out = np.full(years.shape, NO_DATE, dtype=np.int32)
    for year in np.unique(years[years > 0]):
        out[years == year] = to_days(dt.date(int(year), 12, 31))
    return out


def revisit_stats(dates: list[dt.date]) -> dict[str, float]:
    """Return count and min/median/max gap in days between consecutive acquisition dates."""
    ordered = sorted(dates)
    gaps = np.diff([to_days(d) for d in ordered]) if len(ordered) > 1 else np.zeros(1)
    return {
        "n_dates": float(len(ordered)),
        "gap_min_days": float(np.min(gaps)),
        "gap_median_days": float(np.median(gaps)),
        "gap_max_days": float(np.max(gaps)),
    }


def encode_yyddd(day: dt.date, base_year: int = 2000) -> int:
    """Encode a date as RADD-style YYDDD (inverse of decode_radd_yyddd)."""
    return (day.year - base_year) * 1000 + day.timetuple().tm_yday
