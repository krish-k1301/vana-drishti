"""Reviewer tests: label construction in PrefixTruncation and burn-date encoding (Phase 3/4 causality)."""
import datetime as dt

import numpy as np
import torch

from gee.dates import Window, first_burn_day, to_days
from src.data.prefix import PrefixTruncation

SIZE = 4


def _dated_item(event_day: int, event_pixels: list[tuple[int, int]], other_pixels: list[tuple[int, int]],
                burn_pixels: dict[tuple[int, int], int]) -> dict:
    """Un-padded dated item with monthly stored labels at days 1, 20, 40 (all in item-day units)."""
    targets = torch.zeros(3, SIZE, SIZE, dtype=torch.long)
    for r, c in other_pixels:  # another DETER polygon in the patch, alerted on day 15
        targets[1:, r, c] = 1
    for r, c in event_pixels:  # the patch's own DETER polygon
        if event_day <= 20:
            targets[1:, r, c] = 1
        else:
            targets[2:, r, c] = 1
    event_mask = torch.zeros(SIZE, SIZE, dtype=torch.uint8)
    for r, c in event_pixels:
        event_mask[r, c] = 1
    burn = torch.full((SIZE, SIZE), -1, dtype=torch.long)
    for (r, c), day in burn_pixels.items():
        burn[r, c] = day
    return {"Images": torch.zeros(5, 2, SIZE, SIZE), "ImageDays": torch.tensor([1, 10, 20, 30, 40]),
            "TargetDays": torch.tensor([1, 20, 40]), "Targets": targets, "Index": torch.tensor(0),
            "EventDay": torch.tensor(event_day), "EventMask": event_mask, "BurnDay": burn,
            "RaddDay": torch.full((SIZE, SIZE), -1, dtype=torch.long)}


def test_deter_anchor_keeps_other_dated_clearings_in_the_patch() -> None:
    """A pixel the stored cumulative labels mark deforested by day 20 must stay positive at t_c = 40."""
    item = _dated_item(event_day=25, event_pixels=[(2, 2)], other_pixels=[(0, 0)], burn_pixels={})
    cut = PrefixTruncation("deter", min_dates=1, label_interval_days=30).prefix_at(item, 40)
    assert int(item["Targets"][-1, 0, 0]) == 1
    assert int(cut["Targets"][-1, 0, 0]) == 1


def test_burn_anchor_does_not_label_fires_on_negative_patches() -> None:
    """A negative patch (no event, constant stored labels) must stay all-zero under the burn anchor."""
    item = _dated_item(event_day=-1, event_pixels=[], other_pixels=[], burn_pixels={(1, 1): 15})
    cut = PrefixTruncation("burn", min_dates=1, label_interval_days=30).prefix_at(item, 40)
    assert int(cut["Targets"].sum()) == 0


def test_burn_month_is_not_dated_before_it_can_be_known() -> None:
    """A burn in August 2021 must not get a reference day earlier than 31 Aug 2021 (causal-safe encoding)."""
    window = Window(dt.date(2021, 1, 1), dt.date(2021, 12, 31))
    day = first_burn_day({2021: np.array([[8]])}, window)
    assert int(day[0, 0]) >= to_days(dt.date(2021, 8, 31))
