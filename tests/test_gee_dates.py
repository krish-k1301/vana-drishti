"""Tests for gee.dates: windows, label dates, RADD YYDDD decoding, burn months and revisit stats."""
import datetime as dt
import numpy as np
import pytest

from gee.dates import (NO_DATE, Window, compute_window, decode_radd_yyddd, encode_yyddd, first_burn_day,
                       from_days, monthly_label_dates, revisit_stats, to_days, year_end_days)


def test_window_offsets_and_length() -> None:
    """compute_window spans days_before..days_after around the event; negative offsets fail."""
    window = compute_window(dt.date(2021, 6, 15), 365, 120)
    assert window.start == dt.date(2020, 6, 15)
    assert window.end == dt.date(2021, 10, 13)
    assert window.length_days == 486
    assert window.years == [2020, 2021]
    with pytest.raises(ValueError):
        compute_window(dt.date(2021, 1, 1), -1, 10)


def test_monthly_label_dates_start_then_month_firsts() -> None:
    """Label dates are the window start then every month's first day; short windows keep start and end."""
    dates = monthly_label_dates(Window(dt.date(2021, 11, 20), dt.date(2022, 2, 10)))
    assert dates == [dt.date(2021, 11, 20), dt.date(2021, 12, 1), dt.date(2022, 1, 1), dt.date(2022, 2, 1)]
    short = monthly_label_dates(Window(dt.date(2021, 3, 2), dt.date(2021, 3, 20)))
    assert short == [dt.date(2021, 3, 2), dt.date(2021, 3, 20)]


def test_day_encoding_round_trip() -> None:
    """to_days and from_days are inverse day-since-epoch encodings."""
    assert to_days(dt.date(1970, 1, 2)) == 1
    assert from_days(to_days(dt.date(2022, 7, 31))) == dt.date(2022, 7, 31)


def test_radd_decode_known_values_and_invalid() -> None:
    """RADD YYDDD values decode to known days; 0 and invalid codes become NO_DATE."""
    values = np.array([[21001, 20366], [0, 22999]])
    days = decode_radd_yyddd(values)
    assert days[0, 0] == to_days(dt.date(2021, 1, 1))
    assert days[0, 1] == to_days(dt.date(2020, 12, 31))
    assert days[1, 0] == NO_DATE and days[1, 1] == NO_DATE
    assert days.dtype == np.int32


def test_radd_encode_decode_round_trip() -> None:
    """encode_yyddd then decode_radd_yyddd returns the original day."""
    for day in [dt.date(2019, 3, 4), dt.date(2024, 12, 31), dt.date(2023, 1, 1)]:
        assert decode_radd_yyddd(np.array([encode_yyddd(day)]))[0] == to_days(day)


def test_burn_day_is_last_day_of_first_month_and_pre_window_months_are_pre_existing() -> None:
    """Burn day = last day of the first burn month; burns before the window map to window.start - 1."""
    window = Window(dt.date(2020, 6, 15), dt.date(2021, 10, 13))
    monthly = {2020: np.array([[5, 7], [0, 6]]), 2021: np.array([[8, 3], [11, 0]])}
    days = first_burn_day(monthly, window)
    pre_existing = to_days(window.start) - 1
    assert days[0, 0] == pre_existing
    assert days[0, 1] == to_days(dt.date(2020, 7, 31))
    assert days[1, 0] == NO_DATE
    assert days[1, 1] == pre_existing


def test_burn_day_never_precedes_the_burn() -> None:
    """The encoded burn day is the month's last day, never earlier than the burn itself."""
    window = Window(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
    for month in range(1, 13):
        day = first_burn_day({2021: np.array([month])}, window)[0]
        last_possible = (dt.date(2021, month % 12 + 1, 1) if month < 12 else dt.date(2022, 1, 1)) - dt.timedelta(1)
        assert from_days(int(day)) == last_possible


def test_year_end_days() -> None:
    """year_end_days gives 31 December of each year and NO_DATE for year 0."""
    days = year_end_days(np.array([2022, 0, 2020]))
    assert days.tolist() == [to_days(dt.date(2022, 12, 31)), NO_DATE, to_days(dt.date(2020, 12, 31))]


def test_revisit_stats_gaps() -> None:
    """revisit_stats reports date count and min/median/max gaps (0 for a single date)."""
    dates = [dt.date(2022, 1, 1), dt.date(2022, 1, 13), dt.date(2022, 2, 6)]
    stats = revisit_stats(dates)
    assert stats == {"n_dates": 3.0, "gap_min_days": 12.0, "gap_median_days": 18.0, "gap_max_days": 24.0}
    assert revisit_stats([dt.date(2022, 1, 1)])["gap_max_days"] == 0.0
