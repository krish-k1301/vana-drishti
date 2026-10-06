"""Tests for src.metrics.false_alarms."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.false_alarms import (  # noqa: E402
    count_false_alarms,
    false_alarm_rate,
    pixels_to_km2,
    select_threshold_for_budget,
)


def _negatives():
    pred = np.zeros((2, 10, 10), dtype=np.uint8)
    pred[0, 0:2, 0:2] = 1  # one 4 px component
    pred[0, 5, 5] = 1
    pred[0, 6, 6] = 1  # diagonal neighbour: same 8-connected component
    pred[1, 9, 9] = 1  # separate map: its own component
    pred[0, 9, 9] = 1  # same position, other map: must not merge across maps
    return pred


def test_count_components_and_pixels():
    pred = _negatives()
    assert count_false_alarms(pred, "components") == 4
    assert count_false_alarms(pred, "pixels") == 8
    assert count_false_alarms(pred[0], "components") == 3
    with pytest.raises(ValueError):
        count_false_alarms(pred, "patches")


def test_rate_and_area():
    area = pixels_to_km2(200, pixel_area_ha=0.01)
    assert area == pytest.approx(0.02)
    assert false_alarm_rate(_negatives(), area, months=2.0) == pytest.approx(4 / 0.02 / 2)


def test_select_threshold_pixels():
    scores = np.array([[0.1, 0.2, 0.3, 0.4, 0.5]])
    # area 1 km^2, 1 month: budget 2 alarms allows tau=0.4 (0.4, 0.5 positive)
    assert select_threshold_for_budget(scores, 1.0, 1.0, budget=2.0, unit="pixels") == pytest.approx(0.4)
    assert select_threshold_for_budget(scores, 1.0, 1.0, budget=5.0, unit="pixels") == pytest.approx(0.1)
    tau = select_threshold_for_budget(scores, 1.0, 1.0, budget=0.0, unit="pixels")
    assert tau > 0.5 and count_false_alarms(scores >= tau, "pixels") == 0


def test_select_threshold_components_non_monotone():
    """Raising tau from 0.1 to 0.5 splits one blob into two; the chosen tau must stay within budget above it."""
    scores = np.array([[0.9, 0.1, 0.9, 0.0, 0.0]])
    tau = select_threshold_for_budget(scores, 1.0, 1.0, budget=1.0, unit="components")
    assert tau > 0.9
    assert select_threshold_for_budget(scores, 1.0, 1.0, budget=2.0, unit="components") == pytest.approx(0.0)


def test_select_threshold_with_candidates():
    scores = np.random.default_rng(0).random((3, 8, 8))
    grid = np.linspace(0, 1, 101)
    tau = select_threshold_for_budget(scores, 1.0, 1.0, budget=10.0, unit="pixels", candidates=grid)
    assert count_false_alarms(scores >= tau, "pixels") <= 10
    assert count_false_alarms(scores >= tau - 0.01, "pixels") > 10
