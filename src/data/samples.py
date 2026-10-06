"""Low-level readers for the BraDD-S1TS on-disk format (meta.csv + Samples/*.pt)."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

EPOCH = dt.date(1970, 1, 1)


def read_meta(root: str | Path) -> pd.DataFrame:
    """Read `<root>/meta.csv` with its first column as the index, as upstream does."""
    return pd.read_csv(Path(root) / "meta.csv", index_col=0, keep_default_na=False)


def phase_table(meta: pd.DataFrame, split_column: str, phase: str) -> pd.DataFrame:
    """Rows of one phase (train/validation/test) of a split column, re-indexed from 0."""
    if split_column not in meta.columns:
        raise KeyError(f"split column '{split_column}' not in meta.csv columns {list(meta.columns)}")
    return meta[meta[split_column] == phase].reset_index(drop=True)


def subset_rows(table: pd.DataFrame, max_samples: int | None, seed: int) -> pd.DataFrame:
    """Seeded random subset of at most `max_samples` rows, kept in meta order; all rows if None."""
    if max_samples is None or max_samples >= len(table):
        return table
    picked = np.sort(np.random.default_rng(seed).choice(len(table), size=max_samples, replace=False))
    return table.iloc[picked].reset_index(drop=True)


def load_sample(path: str | Path) -> dict:
    """Load one sample dict; weights_only=False because it stores datetime.date objects."""
    return torch.load(path, map_location="cpu", weights_only=False)


def sample_origin(sample: dict) -> dt.date:
    """Earliest image or label date: the day-offset origin used by upstream."""
    return min(list(sample["image_dates"]) + list(sample["label_dates"]))


def days_since(dates: list[dt.date], origin: dt.date) -> torch.Tensor:
    """Day offsets from the origin, starting at 1 (upstream convention), int64 [len(dates)]."""
    return torch.tensor([(d - origin).days + 1 for d in dates], dtype=torch.long)


def epoch_days_to_offsets(values: torch.Tensor, origin: dt.date) -> torch.Tensor:
    """Convert per-pixel days-since-1970 (-1 = none) to day offsets on the sample origin.

    Valid dates before the origin map to 0 (on or before the first date); -1 stays -1.
    """
    values = values.long()
    shifted = (values - (origin - EPOCH).days + 1).clamp(min=0)
    return torch.where(values >= 0, shifted, torch.full_like(values, -1))
