"""Per-event and per-pixel records from sliding-prefix probabilities (PRD 3, 7 Phase 5).

Reference days (all on the sample's day-offset origin, NaN = none):
- `deter`: the DETER alert day `EventDay`, for the event and for every event-mask pixel.
- `burn` / `radd`: per pixel `BurnDay` / `RaddDay`; per event the earliest such day inside the event mask.
Strata: DETER stage (class name mapped by the eval config), size bin of the event mask (event: total mask
area; pixel: its 8-connected component), and edge vs interior of the event mask (pixels only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.io import pixel_area_ha
from src.metrics.latency import first_detection_day
from src.metrics.strata import EDGE_NAMES, bin_labels, edge_interior, size_bins

REFERENCES = ("deter", "burn", "radd")


def days_or_nan(values: np.ndarray) -> np.ndarray:
    """Float day offsets with the -1 'none' sentinel replaced by NaN."""
    values = np.asarray(values, dtype=float)
    return np.where(values < 0, np.nan, values)


def earliest(values: np.ndarray) -> float:
    """Earliest valid day among `values` (NaN if none)."""
    valid = days_or_nan(values)
    return float(np.nanmin(valid)) if np.isfinite(valid).any() else float("nan")


class EarlyCollector:
    """Accumulates curves, event rows, event curves and per-pixel detections for a fixed tau."""

    def __init__(self, eval_cfg: dict, tau: float) -> None:
        """Read stratification and detection settings from the early eval config."""
        self.tau = tau
        self.persist_k = int(eval_cfg["persist_k"])
        self.event_curve = eval_cfg["event_curve"]
        if self.event_curve not in ("mean", "max"):
            raise ValueError("event_curve must be 'mean' or 'max'")
        self.stages = dict(eval_cfg["deter_classes"])
        self.pixel_area = pixel_area_ha(float(eval_cfg["pixel_size_m"]))
        self.bins = tuple(eval_cfg["size_bins_ha"])
        self.labels = bin_labels(self.bins)
        self.edge_width = int(eval_cfg["edge_width_px"])
        self.curve_rows: list[dict] = []
        self.event_rows: list[dict] = []
        self.event_curves: list[tuple[np.ndarray, np.ndarray]] = []
        self.pixel_frames: list[pd.DataFrame] = []

    def add(self, item: dict, meta_row: pd.Series, cutoffs: np.ndarray, probs: np.ndarray) -> None:
        """Add one test event: item from DatedDataset, its meta row, cutoffs [K] and probabilities [K, H, W]."""
        mask = item["EventMask"].numpy().astype(bool)
        event_day = float(item["EventDay"])
        stage = self.stages.get(meta_row["deter_class"], meta_row["deter_class"] or "none")
        inside = probs[:, mask]
        curves = {"mean": inside.mean(axis=1), "max": inside.max(axis=1)}
        base = {"index": int(item["Index"]), "file": meta_row["file"], "stage": stage}
        for k, cutoff in enumerate(cutoffs):
            self.curve_rows.append({**base, "cutoff_day": int(cutoff), "days_since_event": float(cutoff - event_day),
                                    "prob_mean": float(curves["mean"][k]), "prob_max": float(curves["max"][k])})
        curve = curves[self.event_curve]
        area = float(mask.sum()) * self.pixel_area
        refs = {"deter": event_day, "burn": earliest(item["BurnDay"].numpy()[mask]),
                "radd": earliest(item["RaddDay"].numpy()[mask])}
        detection = float(first_detection_day(cutoffs, curve, self.tau, self.persist_k))
        self.event_rows.append({**base, "size_bin": self.labels[int(np.digitize(area, self.bins))], "area_ha": area,
                                "detection_day": detection, **{f"ref_{r}": v for r, v in refs.items()},
                                **{f"latency_{r}": detection - v for r, v in refs.items()}})
        self.event_curves.append((cutoffs, curve))
        self.pixel_frames.append(self._pixels(item, mask, cutoffs, inside, stage, event_day))

    def _pixels(self, item: dict, mask: np.ndarray, cutoffs: np.ndarray, inside: np.ndarray, stage: str,
                event_day: float) -> pd.DataFrame:
        """Per event-mask pixel: detection day, references and strata."""
        bin_map, _ = size_bins(mask, pixel_area_ha=self.pixel_area, bins=self.bins)
        region = edge_interior(mask, width_px=self.edge_width)
        return pd.DataFrame({
            "stage": stage,
            "size_bin": [self.labels[b] for b in bin_map[mask]],
            "edge_interior": [EDGE_NAMES[r] for r in region[mask]],
            "detection_day": first_detection_day(cutoffs, inside, self.tau, self.persist_k),
            "ref_deter": event_day,
            "ref_burn": days_or_nan(item["BurnDay"].numpy()[mask]),
            "ref_radd": days_or_nan(item["RaddDay"].numpy()[mask]),
        })

    def events(self) -> pd.DataFrame:
        """One row per event."""
        return pd.DataFrame(self.event_rows)

    def pixels(self) -> pd.DataFrame:
        """One row per event-mask pixel of every event."""
        return pd.concat(self.pixel_frames, ignore_index=True) if self.pixel_frames else pd.DataFrame()
