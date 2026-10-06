"""Detection latency and recall-at-offset from causal probability curves (PRD 3 and 9).

Conventions (reference-agnostic, so DETER, burn and RADD dates all use the same code):
- A curve is (cutoffs [K] in days, probs [K] or [K, N]); cutoffs strictly increasing; prob at cutoff t_c was
  computed from observations up to t_c only.
- Detected at cutoff i if probs >= tau at cutoffs i-persist_k+1 .. i (all `persist_k` consecutive cutoffs).
  The detection day is the cutoff at which persistence is *confirmed* (the last of the run), so it never uses
  data after the day it reports. `persist_k=1` is plain first crossing.
- Reference days are floats; NaN means "no reference date" (e.g. no burn). Such events are excluded from that
  reference's statistics and counted in `n_no_reference`.
- Never-detected events get detection day NaN and are counted as `n_missed`, never dropped (PRD 9).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

Curve = tuple[np.ndarray, np.ndarray]


def first_detection_day(cutoffs: np.ndarray, probs: np.ndarray, tau: float, persist_k: int = 1) -> np.ndarray:
    """Return the detection day per column of probs [K] or [K, N]; NaN where never detected."""
    cutoffs = np.asarray(cutoffs, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if cutoffs.ndim != 1 or probs.shape[0] != cutoffs.shape[0]:
        raise ValueError("cutoffs must be [K] and probs [K] or [K, N]")
    if np.any(np.diff(cutoffs) <= 0):
        raise ValueError("cutoffs must be strictly increasing")
    if persist_k < 1:
        raise ValueError("persist_k must be >= 1")
    above = (probs >= tau).astype(np.int64)
    run_sum = np.cumsum(above, axis=0)
    lagged = np.concatenate([np.zeros_like(run_sum[:persist_k]), run_sum[:-persist_k]], axis=0)
    confirmed = (run_sum - lagged) >= persist_k
    hit = confirmed.any(axis=0)
    first_idx = confirmed.argmax(axis=0)
    return np.where(hit, cutoffs[first_idx], np.nan)


def detection_days(curves: Sequence[Curve], tau: float, persist_k: int = 1) -> np.ndarray:
    """Return one detection day per event curve (each probs must be [K]); NaN where missed."""
    return np.array([float(first_detection_day(c, p, tau, persist_k)) for c, p in curves], dtype=float)


def latency_summary(det_days: np.ndarray, ref_days: np.ndarray) -> dict:
    """Summarise latency = detection day - reference day over events that have a reference."""
    det_days = np.asarray(det_days, dtype=float)
    ref_days = np.asarray(ref_days, dtype=float)
    has_ref = ~np.isnan(ref_days)
    detected = has_ref & ~np.isnan(det_days)
    lat = det_days[detected] - ref_days[detected]
    stats = {"median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan, "mean": np.nan}
    if lat.size:
        q25, median, q75 = np.percentile(lat, [25, 50, 75])
        stats = {"median": median, "q25": q25, "q75": q75, "iqr": q75 - q25, "mean": float(lat.mean())}
    return {
        "n": int(has_ref.sum()),
        "n_no_reference": int((~has_ref).sum()),
        "n_detected": int(detected.sum()),
        "n_missed": int((has_ref & np.isnan(det_days)).sum()),
        "latencies": lat,
    } | {k: float(v) for k, v in stats.items()}


def recall_at_offsets(
    curves: Sequence[Curve],
    ref_days: np.ndarray,
    offsets: Sequence[float] = (0, 12, 24, 48, 90),
    *,
    tau: float,
    persist_k: int = 1,
) -> dict:
    """Fraction of referenced events detected using only cutoffs <= ref + offset (offsets from PRD 7 Phase 5)."""
    ref_days = np.asarray(ref_days, dtype=float)
    if len(curves) != ref_days.shape[0]:
        raise ValueError("one reference day per curve required")
    has_ref = ~np.isnan(ref_days)
    recall = {}
    for offset in offsets:
        hits = [
            _detected_by(cut, prob, ref + offset, tau, persist_k)
            for (cut, prob), ref, ok in zip(curves, ref_days, has_ref)
            if ok
        ]
        recall[offset] = float(np.mean(hits)) if hits else float("nan")
    return {"n": int(has_ref.sum()), "n_no_reference": int((~has_ref).sum()), "recall": recall}


def _detected_by(cutoffs: np.ndarray, probs: np.ndarray, deadline: float, tau: float, persist_k: int) -> bool:
    """Return True if the curve truncated to cutoffs <= deadline is detected."""
    keep = np.asarray(cutoffs, dtype=float) <= deadline
    if not keep.any():
        return False
    return bool(~np.isnan(first_detection_day(np.asarray(cutoffs)[keep], np.asarray(probs)[keep], tau, persist_k)))


def recall_from_detection_days(det_days: np.ndarray, ref_days: np.ndarray, offsets: Sequence[float]) -> dict:
    """Recall at ref + offset from causal detection days (equals `recall_at_offsets`; used for per-pixel curves)."""
    det_days = np.asarray(det_days, dtype=float)
    ref_days = np.asarray(ref_days, dtype=float)
    has_ref = ~np.isnan(ref_days)
    det, ref = det_days[has_ref], ref_days[has_ref]
    recall = {off: float(np.mean(det <= ref + off)) if ref.size else float("nan") for off in offsets}
    return {"n": int(has_ref.sum()), "n_no_reference": int((~has_ref).sum()), "recall": recall}
