"""False alarms per km^2 per month on stable-forest negatives and budget-based threshold choice (PRD 7 Phase 5).

Every predicted positive on a negative patch is a false alarm. Unit `components` (default) counts 8-connected
predicted components per map, which matches how an alert system issues alerts; unit `pixels` counts pixels.
A pixel is predicted positive when score >= tau (same rule as `src.metrics.latency`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from src.metrics.regions import as_numpy

UNITS = ("components", "pixels")
PLANAR_EIGHT_CONNECTED = np.zeros((3, 3, 3), dtype=bool)
PLANAR_EIGHT_CONNECTED[1] = True


def pixels_to_km2(n_pixels: int, pixel_area_ha: float) -> float:
    """Return the area in km^2 of `n_pixels` pixels of `pixel_area_ha` hectares each (1 km^2 = 100 ha)."""
    return n_pixels * pixel_area_ha / 100.0


def count_false_alarms(pred: np.ndarray, unit: str = "components") -> int:
    """Count predicted positives on negative maps [H, W] or [N, H, W] as components (per map) or pixels."""
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    maps = as_numpy(pred) != 0
    if unit == "pixels":
        return int(maps.sum())
    if maps.ndim == 2:
        maps = maps[None]
    _, n = ndimage.label(maps, structure=PLANAR_EIGHT_CONNECTED)
    return int(n)


def false_alarm_rate(pred: np.ndarray, area_km2: float, months: float, unit: str = "components") -> float:
    """Return false alarms per km^2 per month of monitoring."""
    if area_km2 <= 0 or months <= 0:
        raise ValueError("area_km2 and months must be positive")
    return count_false_alarms(pred, unit) / area_km2 / months


def select_threshold_for_budget(
    neg_scores_val: np.ndarray,
    area_km2: float,
    months: float,
    budget: float,
    unit: str = "components",
    candidates: Sequence[float] | None = None,
) -> float:
    """Smallest tau whose false-alarm rate, and that of every larger candidate, is <= budget.

    Use VALIDATION negatives only (the caller guarantees the split). Candidates default to the unique scores
    plus one value just above the maximum (zero alarms), so a feasible tau always exists. The "every larger
    candidate" clause matters only for components, whose count is not monotone in tau (a blob can split).
    Candidates are kept in the scores' float dtype, so `scores >= tau` reproduces the counted alarms exactly.
    """
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}")
    if area_km2 <= 0 or months <= 0:
        raise ValueError("area_km2 and months must be positive")
    scores = as_numpy(neg_scores_val)
    dtype = scores.dtype.type if np.issubdtype(scores.dtype, np.floating) else np.float64
    scores = scores.astype(dtype)
    top = np.nextafter(scores.max(), dtype(np.inf))
    grid = np.unique(scores if candidates is None else np.asarray(candidates, dtype=dtype))
    grid = np.append(grid[grid < top], top)
    if unit == "pixels":
        sorted_scores = np.sort(scores.ravel())
        counts = sorted_scores.size - np.searchsorted(sorted_scores, grid, side="left")
    else:
        counts = np.array([count_false_alarms(scores >= tau, unit) for tau in grid])
    within = counts / area_km2 / months <= budget
    violating = np.flatnonzero(~within)
    first_ok = violating[-1] + 1 if violating.size else 0
    return float(grid[first_ok])
