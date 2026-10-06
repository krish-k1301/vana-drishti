"""Synthetic dataset in the BraDD-S1TS on-disk format (SMOKE data only, never real measurements)."""
from __future__ import annotations

import datetime as dt
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SIZE = 48
EPOCH = dt.date(1970, 1, 1)
SPLITS = ("train", "validation", "test")
SAMPLING_WEIGHTS = {"positive": 8712, "boundary": 2294, "oldDeforest": 7557, "forest": 5472,
                    "herbaceous": 1780, "agriculture": 152, "shrubs": 21}
STATE_WEIGHTS = {"PA": 11604, "MT": 3264, "AM": 2901, "RO": 2875, "AC": 2329, "RR": 1457,
                 "MA": 1048, "AP": 390, "TO": 120}
DETER_CLASSES = ("DESMATAMENTO_CR", "DESMATAMENTO_VEG", "CICATRIZ_DE_QUEIMADA")
REGION_BLOCKS = {"train": ("R00", "R01", "R02", "R03"), "validation": ("R04",), "test": ("R05",)}
FOREST_DB = (-7.0, -13.0)
OPEN_DB = (-10.0, -18.0)
CLEARING_DROP_DB = (3.0, 5.0)


def _weighted(rng: np.random.Generator, weights: Mapping[str, int]) -> str:
    """Draw one key with probability proportional to its weight."""
    keys = list(weights)
    p = np.array([weights[k] for k in keys], dtype=float)
    return str(rng.choice(keys, p=p / p.sum()))


def _blob(rng: np.random.Generator) -> np.ndarray:
    """Random elliptical boolean mask [48,48]."""
    yy, xx = np.mgrid[:SIZE, :SIZE]
    cy, cx = rng.uniform(10, SIZE - 10, 2)
    ry, rx = rng.uniform(3, 10, 2)
    return ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1.0


def _acquisition_offsets(rng: np.random.Generator, span: int, n_dates: int) -> np.ndarray:
    """Sorted unique day offsets in [0, span], always including both ends."""
    inner = rng.choice(np.arange(1, span), size=n_dates - 2, replace=False)
    return np.sort(np.concatenate([[0, span], inner])).astype(np.int64)


def _images(rng: np.random.Generator, offsets: np.ndarray, base: tuple[float, float],
            cleared: np.ndarray, event_off: int, regrow: np.ndarray) -> torch.Tensor:
    """dB time series [T,2,48,48]: speckle around a base, drop after the event, recovery on regrowth."""
    t = len(offsets)
    img = rng.normal(0.0, 1.0, size=(t, 2, SIZE, SIZE)).astype(np.float32)
    frac_left = 1.0 - (offsets - offsets[0]) / max(int(offsets[-1] - offsets[0]), 1)
    after = (offsets >= event_off)[:, None, None]
    for c in range(2):
        img[:, c] += base[c]
        img[:, c] -= CLEARING_DROP_DB[c] * (after & cleared[None])
        img[:, c] -= CLEARING_DROP_DB[c] * frac_left[:, None, None] * regrow[None]
    return torch.from_numpy(img)


def _bradd_record(rng: np.random.Generator, sampling_type: str, alert: dt.date) -> dict:
    """One BraDD-like sample: two labels, ~1 year + 2 weeks before to 2 weeks after the alert."""
    start = alert - dt.timedelta(days=int(rng.integers(370, 390)))
    span = (alert - start).days + 14
    offsets = _acquisition_offsets(rng, span, int(rng.integers(19, 41)))
    label_offs = (int(rng.integers(5, 20)), (alert - start).days)
    empty = np.zeros((SIZE, SIZE), dtype=bool)
    cleared = _blob(rng) if sampling_type in ("positive", "boundary") else empty
    regrow = _blob(rng) & ~cleared if sampling_type == "boundary" else empty
    event_off = int(rng.integers(label_offs[0] + 1, label_offs[1] + 1))
    base = FOREST_DB if sampling_type in ("positive", "boundary", "forest") else OPEN_DB
    label = np.stack([regrow, cleared]).astype(np.int64)
    return {
        "image_dates": [start + dt.timedelta(days=int(o)) for o in offsets],
        "label_dates": [start + dt.timedelta(days=o) for o in label_offs],
        "image": _images(rng, offsets, base, cleared, event_off, regrow),
        "label": torch.from_numpy(label),
    }


def _prodes_year(day: dt.date) -> int:
    """PRODES year (1 Aug - 31 Jul) that contains the given day."""
    return day.year + 1 if day.month >= 8 else day.year


def _epoch_days(mask: np.ndarray, day: dt.date | None, dtype: type) -> torch.Tensor:
    """Per-pixel days since 1970-01-01 inside the mask, -1 elsewhere."""
    value = -1 if day is None else (day - EPOCH).days
    return torch.from_numpy(np.where(mask, value, -1).astype(dtype))


def _burn_layer(rng: np.random.Generator, deter_class: str | None, mask: np.ndarray,
                event: dt.date) -> torch.Tensor:
    """First-of-month burn day per pixel, depending on the DETER class."""
    if deter_class == "CICATRIZ_DE_QUEIMADA":
        return _epoch_days(mask, event.replace(day=1), np.int16)
    if deter_class == "DESMATAMENTO_CR":
        later = event + dt.timedelta(days=int(rng.integers(30, 60)))
        half = mask & (np.arange(SIZE)[None, :] < SIZE // 2)
        return _epoch_days(half, later.replace(day=1), np.int16)
    return _epoch_days(mask, None, np.int16)


def _dated_record(rng: np.random.Generator, sampling_type: str, event: dt.date,
                  deter_class: str | None, block: str) -> dict:
    """One dated sample: ~12 months before to ~4 months after the event, monthly cumulative labels."""
    start = event - dt.timedelta(days=int(rng.integers(350, 380)))
    span = (event - start).days + int(rng.integers(110, 130))
    offsets = _acquisition_offsets(rng, span, int(rng.integers(30, 46)))
    label_offs = np.arange(0, span + 1, 30)
    mask = _blob(rng) if deter_class else np.zeros((SIZE, SIZE), dtype=bool)
    pre = _blob(rng) & ~mask if sampling_type == "oldDeforest" else np.zeros_like(mask)
    event_off = (event - start).days
    label = np.stack([pre | (mask & (event_off <= d)) for d in label_offs]).astype(np.int64)
    base = OPEN_DB if sampling_type in ("oldDeforest", "herbaceous") else FOREST_DB
    radd = event + dt.timedelta(days=int(rng.integers(5, 40)))
    return {
        "image_dates": [start + dt.timedelta(days=int(o)) for o in offsets],
        "label_dates": [start + dt.timedelta(days=int(o)) for o in label_offs],
        "image": _images(rng, offsets, base, mask, event_off, np.zeros_like(mask)),
        "label": torch.from_numpy(label),
        "event_date": event if deter_class else None,
        "deter_class": deter_class or "",
        "burn_month": _burn_layer(rng, deter_class, mask, event),
        "radd_date": _epoch_days(mask, radd if deter_class else None, np.int32),
        "event_mask": torch.from_numpy(mask.astype(np.uint8)),
        "polygon_area_ha": float(mask.sum()) * 0.01,
        "prodes_year": _prodes_year(event) if deter_class else -1,
        "region_block": block,
    }


def _sampling_type(rng: np.random.Generator, position: int, dated: bool) -> str:
    """Guarantee a positive and a boundary/old-deforest sample in every split, then draw by weight."""
    if position == 0:
        return "positive"
    if position == 1:
        return "oldDeforest" if dated else "boundary"
    if dated:
        return str(rng.choice(["positive", "positive", "forest", "oldDeforest", "herbaceous"]))
    return _weighted(rng, SAMPLING_WEIGHTS)


def _one_sample(rng: np.random.Generator, idx: int, split: str, position: int,
                dated: bool) -> tuple[dict, dict]:
    """Build one (meta row, sample dict) pair."""
    sampling_type = _sampling_type(rng, position, dated)
    day = dt.date(2020, 7, 1) + dt.timedelta(days=int(rng.integers(0, 517)))
    row = {"alert_idx": idx + idx // 3, "center_idx": idx, "date": day.isoformat(),
           "sampling_type": sampling_type, "state": _weighted(rng, STATE_WEIGHTS),
           "file": f"{idx:07d}_{day.isoformat()}.pt", "close_set": split}
    if not dated:
        return row, _bradd_record(rng, sampling_type, day)
    deter_class = str(rng.choice(DETER_CLASSES)) if sampling_type == "positive" else None
    block = str(rng.choice(REGION_BLOCKS[split]))
    row.update({"event_date": day.isoformat() if deter_class else "", "deter_class": deter_class or "",
                "region_block": block, "dated_set": split})
    return row, _dated_record(rng, sampling_type, day, deter_class, block)


def make_bradd_fixture(root: str | Path, n_per_split: int | Mapping[str, int] = 4,
                       dated: bool = False, seed: int = 0) -> Path:
    """Write a tiny synthetic dataset (meta.csv + Samples/*.pt) in the BraDD format and return its root."""
    root = Path(root)
    (root / "Samples").mkdir(parents=True, exist_ok=True)
    counts = n_per_split if isinstance(n_per_split, Mapping) else {s: n_per_split for s in SPLITS}
    rng = np.random.default_rng(seed)
    order = [(split, pos) for split in SPLITS for pos in range(counts.get(split, 0))]
    rng.shuffle(order)
    rows = []
    for idx, (split, pos) in enumerate(order):
        row, sample = _one_sample(rng, idx, split, int(pos), dated)
        torch.save(sample, root / "Samples" / row["file"])
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "meta.csv")
    return root
