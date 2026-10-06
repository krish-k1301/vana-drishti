"""Confusion metrics split by a per-pixel region map (shared by size-bin and edge/interior strata)."""
from __future__ import annotations

import numpy as np
import torch

from src.metrics.segmentation import ConfusionMeter


def as_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a numpy view/copy of a numpy array or (CPU-moved) torch tensor."""
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def as_maps(array: np.ndarray | torch.Tensor) -> np.ndarray:
    """Return a binary uint8 stack [N, H, W] from a single map [H, W] or a stack."""
    maps = as_numpy(array)
    if maps.ndim == 2:
        maps = maps[None]
    if maps.ndim != 3:
        raise ValueError(f"expected [H, W] or [N, H, W], got shape {maps.shape}")
    return (maps != 0).astype(np.uint8)


class RegionMeter:
    """One `ConfusionMeter` per region id; pixels whose region id is not in `names` are ignored."""

    def __init__(self, names: dict[int, str]) -> None:
        """Create a meter per region; `names` maps region id -> label used in `compute()`."""
        self.names = dict(names)
        self.meters = {rid: ConfusionMeter() for rid in self.names}

    def update(self, pred: np.ndarray, target: np.ndarray, region_map: np.ndarray) -> None:
        """Add binary pred/target maps and the matching integer region map (all the same shape)."""
        pred, target, region_map = as_numpy(pred), as_numpy(target), as_numpy(region_map)
        if not pred.shape == target.shape == region_map.shape:
            raise ValueError("pred, target and region_map must have the same shape")
        for rid, meter in self.meters.items():
            sel = region_map == rid
            meter.update(torch.from_numpy((pred[sel] != 0).astype(np.int64)),
                         torch.from_numpy((target[sel] != 0).astype(np.int64)))

    def compute(self) -> dict:
        """Return {region label: scores} for every region."""
        return {self.names[rid]: meter.compute() for rid, meter in self.meters.items()}

    def reset(self) -> None:
        """Clear all region meters."""
        for meter in self.meters.values():
            meter.reset()
