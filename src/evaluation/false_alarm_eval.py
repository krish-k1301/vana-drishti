"""Stable-forest negatives for Phase 5: ever-alarmed score maps, tau at a false-alarm budget, alarm rates.

A negative pixel's score is its maximum change probability over all cutoffs (an alarm raised at any cutoff
stays raised), and the monitoring time of a patch is the span between its first and last cutoff. Alarms are
counted per patch (components never cross patches) and divided by total area x mean months, which equals
the sum over patches of area x months because every patch has the same size.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from src.metrics.false_alarms import count_false_alarms, pixels_to_km2, select_threshold_for_budget

DAYS_PER_MONTH = 365.25 / 12  # calendar average, converts cutoff spans in days to months


def negative_rows(meta: pd.DataFrame, sampling_types: list[str]) -> list[int]:
    """Row indices of stable-forest negatives: listed sampling type and no DETER event."""
    keep = meta["sampling_type"].isin(sampling_types) & (meta["deter_class"].astype(str) == "")
    return [int(i) for i in meta.index[keep]]


def negative_maps(rows: list[int], score_fn: Callable[[int], tuple[np.ndarray, np.ndarray]]) -> dict:
    """Ever-alarmed score maps [N, H, W] and monitoring months [N] for patches with >= 2 cutoffs."""
    maps, months, skipped = [], [], 0
    for row in rows:
        cutoffs, probs = score_fn(row)
        if len(cutoffs) < 2:
            skipped += 1
            continue
        maps.append(probs.max(axis=0))
        months.append((cutoffs[-1] - cutoffs[0]) / DAYS_PER_MONTH)
    if not maps:
        raise ValueError("no stable-forest negative with >= 2 cutoffs; cannot measure false alarms")
    return {"maps": np.stack(maps), "months": np.asarray(months), "n_skipped": skipped}


def exposure(negatives: dict, pixel_area_ha: float) -> tuple[float, float]:
    """(total area km^2, mean monitoring months) of a negative set."""
    return pixels_to_km2(negatives["maps"].size, pixel_area_ha), float(negatives["months"].mean())


def choose_tau(negatives: dict, pixel_area_ha: float, fa_cfg: dict) -> float:
    """Smallest tau within the false-alarm budget on these (VALIDATION) negatives."""
    area, months = exposure(negatives, pixel_area_ha)
    candidates = fa_cfg["threshold_candidates"]
    return select_threshold_for_budget(negatives["maps"], area, months, float(fa_cfg["budget_per_km2_month"]),
                                       unit=fa_cfg["unit"], candidates=candidates)


def false_alarm_row(split: str, negatives: dict, tau: float, pixel_area_ha: float, fa_cfg: dict) -> dict:
    """False alarms per km^2 per month at `tau` on one split's negatives."""
    area, months = exposure(negatives, pixel_area_ha)
    n_alarms = count_false_alarms(negatives["maps"] >= tau, fa_cfg["unit"])
    return {"split": split, "unit": fa_cfg["unit"], "tau": tau, "budget_per_km2_month": fa_cfg["budget_per_km2_month"],
            "n_patches": len(negatives["maps"]), "n_skipped": negatives["n_skipped"], "n_alarms": n_alarms,
            "area_km2": area, "mean_months": months, "rate_per_km2_month": n_alarms / area / months}
