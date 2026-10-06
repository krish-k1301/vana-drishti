"""Tidy latency and recall tables over strata (PRD 7 Phase 5): one row per (reference, level, stratum)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from src.evaluation.early_records import REFERENCES
from src.metrics.latency import latency_summary, recall_at_offsets, recall_from_detection_days

EVENT_STRATA = ("all", "stage", "size_bin")
PIXEL_STRATA = ("all", "stage", "size_bin", "edge_interior")
SUMMARY_KEYS = ("n", "n_no_reference", "n_detected", "n_missed", "median", "q25", "q75", "iqr", "mean")


def groups(frame: pd.DataFrame, strata: Sequence[str]) -> list[tuple[str, str, pd.DataFrame]]:
    """(stratum type, stratum, rows) for the whole frame and for every value of each stratum column."""
    out = [("all", "all", frame)] if "all" in strata else []
    for column in (s for s in strata if s != "all"):
        out += [(column, str(value), part) for value, part in frame.groupby(column, sort=True)]
    return out


def latency_table(events: pd.DataFrame, pixels: pd.DataFrame) -> list[dict]:
    """Latency summaries (days; missed counted, no-reference excluded) for every reference, level and stratum."""
    rows = []
    for level, frame, strata in (("event", events, EVENT_STRATA), ("pixel", pixels, PIXEL_STRATA)):
        if frame.empty:
            continue
        for reference in REFERENCES:
            for kind, stratum, part in groups(frame, strata):
                summary = latency_summary(part["detection_day"].to_numpy(), part[f"ref_{reference}"].to_numpy())
                rows.append({"reference": reference, "level": level, "stratum_type": kind, "stratum": stratum,
                             **{k: summary[k] for k in SUMMARY_KEYS}})
    return rows


def recall_table(events: pd.DataFrame, curves: list[tuple[np.ndarray, np.ndarray]], pixels: pd.DataFrame,
                 offsets: Sequence[float], tau: float, persist_k: int, reference: str) -> list[dict]:
    """Recall at reference date + offset per level and stratum; events use only cutoffs <= deadline."""
    rows = []
    if events.empty:
        return rows
    for kind, stratum, part in groups(events, EVENT_STRATA):
        result = recall_at_offsets([curves[i] for i in part.index], part[f"ref_{reference}"].to_numpy(), offsets,
                                   tau=tau, persist_k=persist_k)
        rows += _recall_rows(reference, "event", kind, stratum, result)
    for kind, stratum, part in groups(pixels, PIXEL_STRATA):
        detected = part["detection_day"].to_numpy()
        result = recall_from_detection_days(detected, part[f"ref_{reference}"].to_numpy(), offsets)
        rows += _recall_rows(reference, "pixel", kind, stratum, result)
    return rows


def _recall_rows(reference: str, level: str, kind: str, stratum: str, result: dict) -> list[dict]:
    """Flatten one recall result into rows (one per offset)."""
    return [{"reference": reference, "level": level, "stratum_type": kind, "stratum": stratum,
             "offset_days": offset, "recall": value, "n": result["n"], "n_no_reference": result["n_no_reference"]}
            for offset, value in result["recall"].items()]
