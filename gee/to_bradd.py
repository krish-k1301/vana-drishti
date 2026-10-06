"""Pure conversion of pulled patch arrays into BraDD-format samples (`Samples/<id>_<date>.pt`) and `meta.csv`."""
from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd
import torch

from gee.dates import NO_DATE, to_days, year_end_days

META_COLUMNS = ["alert_idx", "center_idx", "date", "sampling_type", "state", "file", "close_set",
                "event_date", "deter_class", "region_block", "dated_set"]
EXTRA_COLUMNS = ["patch_id", "lon", "lat", "relative_orbit", "orbit_pass", "platforms", "n_dates",
                 "gap_median_days", "gap_max_days", "modis_burn_flag"]


def cumulative_labels(ref_day: np.ndarray, prior: np.ndarray, label_dates: list[dt.date]) -> np.ndarray:
    """Return int64 [t, H, W]: label[k] = prior OR (pixel reference day <= label_dates[k])."""
    dated = ref_day != NO_DATE
    masks = [prior.astype(bool) | (dated & (ref_day <= to_days(d))) for d in label_dates]
    return np.stack(masks).astype(np.int64)


def hansen_reference(lossyear: np.ndarray, base_year: int) -> np.ndarray:
    """int32 reference day per pixel from Hansen loss: 31 Dec of the loss year (causal for an annual product), -1 none.

    RADD is never used here: it stays a per-pixel comparison attribute (`radd_date`), never label or dating source.
    """
    return year_end_days(np.where(lossyear > 0, lossyear + base_year, 0))


def sample_file_name(patch_id: str, anchor_date: dt.date) -> str:
    """BraDD-style sample file name `<id>_<YYYY-MM-DD>.pt`."""
    return f"{patch_id}_{anchor_date.isoformat()}.pt"


def build_sample(image: np.ndarray, image_dates: list[dt.date], label_dates: list[dt.date], label: np.ndarray,
                 extras: dict) -> dict:
    """Assemble one sample dict in the INTERFACES.md format (core keys + dated extra keys).

    Dated keys beyond INTERFACES.md: `ref_day` int32 [H,W] = days since epoch of the event that labels each pixel
    (Amazon: DETER date of every label-class polygon covering it, earliest wins; Phase 6: 31 Dec of the Hansen loss
    year), -1 = none; `label_hansen` int64 [t,H,W] = cumulative Hansen labels on the same label_dates.
    `burn_month` = last day of the first burn month (1-month resolution; window.start - 1 = pre-existing).
    """
    if image.ndim != 4 or image.shape[1] != 2 or image.shape[0] != len(image_dates):
        raise ValueError(f"image must be [T,2,H,W] with T == len(image_dates); got {image.shape}")
    if label.shape[0] != len(label_dates) or label.shape[1:] != image.shape[2:]:
        raise ValueError(f"label must be [t,H,W] matching label_dates and image; got {label.shape}")
    if np.shape(extras["label_hansen"]) != label.shape:
        raise ValueError("label_hansen must have the same shape as label")
    sample = {
        "image_dates": list(image_dates),
        "label_dates": list(label_dates),
        "image": torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)),
        "label": torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64)),
        "event_date": extras["event_date"],
        "deter_class": str(extras["deter_class"]),
        "burn_month": torch.from_numpy(np.asarray(extras["burn_month"], dtype=np.int16)),
        "radd_date": torch.from_numpy(np.asarray(extras["radd_date"], dtype=np.int32)),
        "prodes_year": int(extras["prodes_year"]),
        "polygon_area_ha": float(extras["polygon_area_ha"]),
        "event_mask": torch.from_numpy(np.asarray(extras["event_mask"], dtype=np.uint8)),
        "ref_day": torch.from_numpy(np.asarray(extras["ref_day"], dtype=np.int32)),
        "label_hansen": torch.from_numpy(np.ascontiguousarray(extras["label_hansen"], dtype=np.int64)),
        "region_block": str(extras["region_block"]),
    }
    return sample


def save_sample(root: str, file_name: str, sample: dict) -> str:
    """Write a sample to `<root>/Samples/<file_name>` and return the path."""
    folder = os.path.join(root, "Samples")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, file_name)
    torch.save(sample, path)
    return path


def write_meta(root: str, rows: list[dict]) -> str:
    """Write `<root>/meta.csv` (index column first) with the BraDD + dated columns, then provenance extras."""
    frame = pd.DataFrame(rows, columns=META_COLUMNS + EXTRA_COLUMNS)
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "meta.csv")
    frame.to_csv(path)
    return path
