"""Sliding-prefix (causal) inference for Phase 5: one cumulative "cleared by t_c" probability map per cutoff.

At cutoff t_c, `PrefixTruncation.prefix_at` drops every image after t_c and builds the training label-day grid
(backwards from t_c every `label_interval_days`). The model runs in segment mode over all those intervals, giving
one change probability p_k per interval (softmax class 1). Intervals the model marks invalid
(`SegmentNetwork.interval_mask`: no image in the window) contribute nothing. The per-pixel cumulative
probability combines the intervals:
- `noisy_or` (default, orchestrator decision): 1 - prod_k (1 - p_k), "changed in at least one interval";
- `max`: max_k p_k.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.collate import collate_batch
from src.data.prefix import PrefixTruncation

COMBINERS = ("noisy_or", "max")
MODEL_KEYS = ("Images", "ImageDays", "TargetDays")


def prefix_inputs(sample: dict, prefix: PrefixTruncation, cutoff: int) -> dict:
    """Model inputs at one cutoff: images up to t_c and the prefix label-day grid ending at t_c."""
    cut = prefix.prefix_at(sample, cutoff)
    if int(cut["ImageDays"].max()) > cutoff:
        raise RuntimeError("prefix truncation kept an image after the cutoff")
    return {key: cut[key] for key in MODEL_KEYS}


def combine_intervals(probs: torch.Tensor, valid: torch.Tensor, combiner: str) -> torch.Tensor:
    """Combine per-interval change probabilities [B, t-1, H, W] (invalid intervals ignored) to [B, H, W]."""
    probs = probs * valid[:, :, None, None]
    if combiner == "noisy_or":
        return 1.0 - torch.prod(1.0 - probs, dim=1)
    if combiner == "max":
        return probs.max(dim=1).values
    raise ValueError(f"unknown interval combiner '{combiner}', expected one of {COMBINERS}")


@torch.no_grad()
def valid_cutoffs(model: nn.Module, sample: dict, prefix: PrefixTruncation, min_dates: int) -> np.ndarray:
    """Image days with >= `min_dates` images up to them and >= 1 interval the model can score."""
    days = sample["ImageDays"]
    candidates = days[days > 0][min_dates - 1:].tolist()
    keep = []
    for cutoff in candidates:
        inputs = prefix_inputs(sample, prefix, int(cutoff))
        if bool(model.interval_mask(inputs["ImageDays"][None], inputs["TargetDays"][None]).any()):
            keep.append(int(cutoff))
    return np.asarray(keep, dtype=np.int64)


@torch.no_grad()
def cumulative_probabilities(model: nn.Module, sample: dict, cutoffs: np.ndarray, prefix: PrefixTruncation,
                             batch_size: int, combiner: str) -> np.ndarray:
    """Cumulative change probability [K, H, W] (float32) at every cutoff, `batch_size` prefixes per forward."""
    maps = []
    for start in range(0, len(cutoffs), batch_size):
        batch = collate_batch([prefix_inputs(sample, prefix, int(c)) for c in cutoffs[start:start + batch_size]])
        args = (batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])
        probs = torch.softmax(model(*args), dim=2)[:, :, 1]
        valid = model.interval_mask(args[1], args[2], args[3])
        maps.append(combine_intervals(probs, valid.to(probs.dtype), combiner).float().cpu().numpy())
    return np.concatenate(maps, axis=0)


def prefix_from_config(data_cfg: dict) -> PrefixTruncation:
    """PrefixTruncation with the run's anchor/min_dates/label interval (the training label-day grid)."""
    spec = data_cfg["prefix_truncation"]
    return PrefixTruncation(spec["anchor"], int(spec["min_dates"]), int(spec["label_interval_days"]))
