"""Sliding-prefix (causal) inference for Phase 5: one change probability map per cutoff day t_c.

For a dated sample and cutoff t_c, `PrefixTruncation.prefix_at` drops every image after t_c, so the model can
never see the future whatever its trailing margin. The model is asked for the single interval
[first label day, t_c] and the softmax probability of class 1 (change) is kept per pixel.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.collate import collate_batch
from src.data.prefix import PrefixTruncation


def cutoff_days(sample: dict, error_days_before: int, min_dates: int) -> np.ndarray:
    """Image days t_c after the first label day with >= `min_dates` images inside (start - margin, t_c]."""
    days = sample["ImageDays"].numpy()
    start = int(sample["TargetDays"][0])
    in_window = days > start - error_days_before
    count = np.cumsum(in_window)
    keep = in_window & (count >= min_dates) & (days > start)
    return days[keep].astype(np.int64)


def two_label_prefix(sample: dict, prefix: PrefixTruncation, cutoff: int) -> dict:
    """Model inputs for the interval [first label day, cutoff], built from `prefix_at` (no image after cutoff)."""
    cut = prefix.prefix_at(sample, cutoff)
    if int(cut["ImageDays"].max()) > cutoff:
        raise RuntimeError("prefix truncation kept an image after the cutoff")
    return {
        "Images": cut["Images"],
        "ImageDays": cut["ImageDays"],
        "TargetDays": torch.stack([sample["TargetDays"][0], torch.tensor(cutoff)]).long(),
        "Targets": torch.stack([sample["Targets"][0], cut["Targets"][-1]]).long(),
    }


@torch.no_grad()
def prefix_probabilities(model: nn.Module, sample: dict, cutoffs: np.ndarray, prefix: PrefixTruncation,
                         batch_size: int) -> np.ndarray:
    """Change probability [K, H, W] (float32) for every cutoff, run in chunks of `batch_size` prefixes."""
    maps = []
    for start in range(0, len(cutoffs), batch_size):
        items = [two_label_prefix(sample, prefix, int(c)) for c in cutoffs[start:start + batch_size]]
        batch = collate_batch(items)
        logits = model(batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])
        maps.append(torch.softmax(logits[:, 0], dim=1)[:, 1].float().cpu().numpy())
    return np.concatenate(maps, axis=0)


def prefix_from_config(data_cfg: dict) -> PrefixTruncation:
    """PrefixTruncation with the run's anchor/min_dates/label interval (anchor only shapes unused targets)."""
    spec = data_cfg["prefix_truncation"]
    return PrefixTruncation(spec["anchor"], int(spec["min_dates"]), int(spec["label_interval_days"]))
