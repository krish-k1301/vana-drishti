"""Pilot quality checks on a converted dated set (pure: reads meta.csv and Samples/*.pt only)."""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import torch

DATED_KEYS = {"event_date": None, "deter_class": None, "burn_month": torch.int16, "radd_date": torch.int32,
              "prodes_year": None, "polygon_area_ha": None, "event_mask": torch.uint8, "region_block": None,
              "ref_day": torch.int32, "label_hansen": torch.int64}


def _in_range_fraction(values: np.ndarray, bounds: list[float]) -> float:
    """Fraction of values inside [lo, hi]."""
    return float(np.mean((values >= bounds[0]) & (values <= bounds[1])))


def _is_single_orbit(value: object) -> bool:
    """True if the recorded relative orbit is one integer orbit number."""
    try:
        return float(value) == int(float(value)) > 0
    except (TypeError, ValueError):
        return False


def check_format(sample: dict) -> bool:
    """True if core keys, dtypes and shapes match INTERFACES.md and all dated extra keys are present."""
    image, label = sample.get("image"), sample.get("label")
    if not isinstance(image, torch.Tensor) or not isinstance(label, torch.Tensor):
        return False
    shape_ok = (image.dtype == torch.float32 and image.ndim == 4 and image.shape[1] == 2
                and label.dtype == torch.int64 and label.ndim == 3 and label.shape[1:] == image.shape[2:]
                and len(sample.get("image_dates", [])) == image.shape[0]
                and len(sample.get("label_dates", [])) == label.shape[0] >= 2)
    dates_ok = all(isinstance(d, dt.date) for d in sample.get("image_dates", []) + sample.get("label_dates", []))
    keys_ok = all(k in sample and (t is None or sample[k].dtype == t) for k, t in DATED_KEYS.items())
    keys_ok = keys_ok and sample["label_hansen"].shape == label.shape and sample["ref_day"].shape == label.shape[1:]
    return bool(shape_ok and dates_ok and keys_ok)


def check_sample(sample: dict, row: pd.Series, qa_cfg: dict) -> dict[str, bool]:
    """Run every per-sample QA check; True means the check passed."""
    checks = {"format": check_format(sample)}
    if not checks["format"]:
        return checks
    image = sample["image"].numpy()
    label = sample["label"].numpy()
    vv, vh = image[:, 0], image[:, 1]
    tolerance = 1.0 - qa_cfg["max_out_of_range_fraction"]
    finite = bool(np.isfinite(image).all())
    checks["finite"] = finite
    checks["vv_range"] = finite and _in_range_fraction(vv, qa_cfg["vv_range"]) >= tolerance
    checks["vh_range"] = finite and _in_range_fraction(vh, qa_cfg["vh_range"]) >= tolerance
    median_lo, median_hi = qa_cfg["vv_median_range"]
    checks["db_scale"] = finite and median_lo <= float(np.median(vv)) <= median_hi
    checks["single_orbit"] = _is_single_orbit(row["relative_orbit"])
    checks["t_range"] = qa_cfg["t_range"][0] <= image.shape[0] <= qa_cfg["t_range"][1]
    checks["revisit_gap"] = float(row["gap_max_days"]) <= qa_cfg["max_gap_days"]
    dates = sample["image_dates"]
    checks["dates_sorted"] = all(a < b for a, b in zip(dates, dates[1:]))
    checks["labels_binary"] = bool(np.isin(label, [0, 1]).all())
    changed = bool((label[-1] != label[0]).any())
    positive = row["sampling_type"] == "positive"
    checks["label_change"] = (changed and bool(sample["event_mask"].numpy().any())) if positive else not changed
    return checks


def run_qa(root: str, qa_cfg: dict) -> pd.DataFrame:
    """Return one row per sample in `<root>/meta.csv` with a boolean column per check and an `ok` column."""
    meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0, keep_default_na=False)
    records = []
    for _, row in meta.iterrows():
        sample = torch.load(os.path.join(root, "Samples", row["file"]), weights_only=False)
        checks = check_sample(sample, row, qa_cfg)
        records.append({"patch_id": row["patch_id"], "sampling_type": row["sampling_type"], **checks})
    table = pd.DataFrame(records)
    if table.empty:
        return table.assign(ok=pd.Series(dtype=bool))
    check_cols = [c for c in table.columns if c not in ("patch_id", "sampling_type")]
    table[check_cols] = table[check_cols].fillna(False).astype(bool)
    table["ok"] = table[check_cols].all(axis=1)
    return table


def qa_passed(table: pd.DataFrame, qa_cfg: dict) -> tuple[bool, list[str]]:
    """Overall pilot verdict and the reasons it failed (empty when passed)."""
    reasons = []
    n_pos = int((table.get("sampling_type", pd.Series(dtype=str)) == "positive").sum())
    if n_pos < qa_cfg["min_positives"]:
        reasons.append(f"only {n_pos} positive samples converted (< {qa_cfg['min_positives']})")
    if len(table) and float((~table["ok"]).mean()) > qa_cfg["max_failed_fraction"]:
        reasons.append(f"{int((~table['ok']).sum())} of {len(table)} samples failed a check")
    if not len(table):
        reasons.append("no samples converted")
    return not reasons, reasons


def random_qa_sample(root: str, n: int, seed: int) -> pd.DataFrame:
    """Random positives for the owner's NICFI spot-check: patch id, centroid lon/lat, DETER date and class."""
    meta = pd.read_csv(os.path.join(root, "meta.csv"), index_col=0, keep_default_na=False)
    positives = meta[meta["sampling_type"] == "positive"]
    chosen = positives.sample(n=min(n, len(positives)), random_state=seed)
    return chosen[["patch_id", "lon", "lat", "event_date", "deter_class", "dated_set", "file"]]
