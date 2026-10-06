"""Tests for src.metrics.latency."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.latency import (  # noqa: E402
    detection_days,
    first_detection_day,
    latency_summary,
    recall_at_offsets,
    recall_from_detection_days,
)

CUT = np.array([0.0, 12.0, 24.0, 36.0, 48.0])


def test_first_crossing_single_curve():
    probs = np.array([0.1, 0.6, 0.2, 0.7, 0.9])
    assert first_detection_day(CUT, probs, tau=0.5) == 12.0


def test_persistence_reports_confirmation_day():
    probs = np.array([0.1, 0.6, 0.2, 0.7, 0.9])
    assert first_detection_day(CUT, probs, tau=0.5, persist_k=2) == 48.0
    assert np.isnan(first_detection_day(CUT, probs, tau=0.5, persist_k=3))
    assert np.isnan(first_detection_day(CUT, probs, tau=0.5, persist_k=10))


def test_per_pixel_curves():
    probs = np.array([[0.0, 0.9, 0.0], [0.6, 0.9, 0.0], [0.6, 0.9, 0.4]] + [[0.6, 0.9, 0.4]] * 2)
    days = first_detection_day(CUT, probs, tau=0.5)
    np.testing.assert_array_equal(days[:2], [12.0, 0.0])
    assert np.isnan(days[2])


def test_tau_is_inclusive_and_cutoffs_validated():
    assert first_detection_day(CUT[:2], np.array([0.5, 0.5]), tau=0.5) == 0.0
    with pytest.raises(ValueError):
        first_detection_day(np.array([0.0, 0.0]), np.array([0.1, 0.2]), tau=0.5)


def test_latency_summary_keeps_missed_and_excludes_no_reference():
    det = np.array([10.0, 20.0, 30.0, np.nan, 5.0])
    ref = np.array([0.0, 0.0, 0.0, 0.0, np.nan])
    out = latency_summary(det, ref)
    assert (out["n"], out["n_detected"], out["n_missed"], out["n_no_reference"]) == (4, 3, 1, 1)
    assert out["median"] == 20.0 and out["mean"] == 20.0
    assert (out["q25"], out["q75"], out["iqr"]) == (15.0, 25.0, 10.0)


def test_negative_latency_when_beating_reference():
    out = latency_summary(np.array([-6.0]), np.array([6.0]))
    assert out["median"] == -12.0


def test_latency_summary_all_missed():
    out = latency_summary(np.array([np.nan]), np.array([3.0]))
    assert out["n_missed"] == 1 and np.isnan(out["median"])


def _curves():
    return [
        (CUT, np.array([0.0, 0.0, 0.8, 0.8, 0.8])),  # detected day 24
        (CUT, np.array([0.0, 0.0, 0.0, 0.0, 0.9])),  # detected day 48
        (CUT, np.array([0.0, 0.0, 0.0, 0.0, 0.0])),  # never
        (CUT, np.array([0.9, 0.9, 0.9, 0.9, 0.9])),  # no reference
    ]


def test_recall_at_offsets():
    ref = np.array([12.0, 12.0, 12.0, np.nan])
    out = recall_at_offsets(_curves(), ref, offsets=(0, 12, 36, 90), tau=0.5)
    assert (out["n"], out["n_no_reference"]) == (3, 1)
    assert out["recall"] == pytest.approx({0: 0.0, 12: 1 / 3, 36: 2 / 3, 90: 2 / 3})


def test_recall_equals_detection_day_rule():
    """Truncating to cutoffs <= deadline agrees with comparing the (causal) detection day to the deadline."""
    rng = np.random.default_rng(0)
    curves = [(CUT, rng.random(5)) for _ in range(50)]
    ref = rng.integers(-10, 40, size=50).astype(float)
    for k in (1, 2):
        det = detection_days(curves, tau=0.6, persist_k=k)
        out = recall_at_offsets(curves, ref, offsets=(0, 12), tau=0.6, persist_k=k)
        for off in (0, 12):
            assert out["recall"][off] == pytest.approx(np.mean(det <= ref + off))


def test_recall_from_detection_days_matches_curves():
    ref = np.array([12.0, 12.0, 12.0, np.nan])
    det = detection_days(_curves(), tau=0.5)
    fast = recall_from_detection_days(det, ref, offsets=(0, 12, 36))
    slow = recall_at_offsets(_curves(), ref, offsets=(0, 12, 36), tau=0.5)
    assert fast == slow
