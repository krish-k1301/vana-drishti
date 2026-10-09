"""Per-pixel explanation helpers for U-TAE runs: one interval's probability, temporal attention and pixel picks."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.data.collate import collate_batch


def busiest_interval(model: nn.Module, item: dict) -> int:
    """Index of the valid label interval with the most true change pixels (first valid one if none change)."""
    batch = collate_batch([item])
    valid = model.interval_mask(batch["ImageDays"], batch["TargetDays"], batch["PadMask"])[0]
    targets = item["Targets"]
    amount = (targets[:-1] != targets[1:]).flatten(1).sum(1).masked_fill(~valid, -1)
    return int(amount.argmax())


@torch.no_grad()
def explain_interval(model: nn.Module, item: dict, k: int) -> dict:
    """Probability, decision, truth and head-mean temporal attention of label interval `k` for one item.

    The U-TAE backbone runs on exactly the dates the segment network selects for interval k; its attention
    is [dates, h, w] at the encoder's coarsest resolution (6x6 for 48x48 inputs), averaged over heads.
    """
    days, target_days = item["ImageDays"][None], item["TargetDays"][None]
    keep = model.window(days, target_days, torch.ones_like(days, dtype=torch.bool), k)[0]
    logits, attention = model.backbone(item["Images"][keep][None], days[0, keep][None], return_att=True)
    targets = item["Targets"]
    prob = logits[0].softmax(0)[1].numpy()
    return {
        "interval": k,
        "kept": torch.nonzero(keep).flatten().numpy(),
        "prob": prob,
        "pred": logits[0].argmax(0).numpy(),
        "true": (targets[k] != targets[k + 1]).long().numpy(),
        "prior": targets[k].numpy(),
        "attention": attention[:, 0].mean(0).numpy(),
    }


def pick_pixels(result: dict, margin: int = 3) -> dict[str, tuple[int, int]]:
    """Example pixels (row, col) at least `margin` px from the border: surest hit, surest stable, worst error."""
    prob, true, pred, prior = result["prob"], result["true"], result["pred"], result["prior"]
    inner = np.zeros_like(true, dtype=bool)
    inner[margin:-margin, margin:-margin] = True
    true, pred = np.where(inner, true, -1), np.where(inner, pred, -1)
    stable = (true == 0) & (prior == 0)
    candidates = {
        "detected clearing": (np.where((true == 1) & (pred == 1), prob, -np.inf), "max"),
        "stable, no change": (np.where(stable & (pred == 0), prob, np.inf), "min"),
        "missed clearing": (np.where((true == 1) & (pred == 0), prob, np.inf), "min"),
        "false alarm": (np.where(stable & (pred == 1), prob, -np.inf), "max"),
    }
    picks = {}
    for name, (score, mode) in candidates.items():
        if np.isfinite(score).any():
            flat = int(score.argmax() if mode == "max" else score.argmin())
            picks[name] = divmod(flat, score.shape[1])
    errors = [name for name in ("missed clearing", "false alarm") if name in picks]
    return {name: rc for name, rc in picks.items() if name not in errors[1:]}


def attention_cell(attention: np.ndarray, pixel: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    """Attention over dates of the coarse cell that contains `pixel` in an image of `size` (rows, cols)."""
    rows, cols = attention.shape[-2:]
    return attention[:, pixel[0] * rows // size[0], pixel[1] * cols // size[1]]


def neighbourhood_series(image: np.ndarray, pixel: tuple[int, int], radius: int) -> np.ndarray:
    """[T, C] mean over the (2*radius+1)^2 window around `pixel` (clipped at the border) of an image [T, C, H, W]."""
    row, col = pixel
    window = image[:, :, max(0, row - radius):row + radius + 1, max(0, col - radius):col + radius + 1]
    return window.mean(axis=(2, 3))


def change_features(raw: dict, result: dict, label_dates: list) -> dict[str, np.ndarray]:
    """Per-pixel dB change (mean after - mean before the interval midpoint) of VV and VH over the kept dates."""
    dates = [raw["image_dates"][i] for i in result["kept"]]
    images = raw["image"][result["kept"]].numpy()
    start, end = label_dates[result["interval"]], label_dates[result["interval"] + 1]
    middle = start + (end - start) / 2
    before = np.array([d <= middle for d in dates])
    if before.all() or not before.any():
        before = np.arange(len(dates)) < max(1, len(dates) // 2)
    delta = images[~before].mean(0) - images[before].mean(0)
    return {"vv": delta[0], "vh": delta[1]}
