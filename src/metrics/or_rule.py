"""Measure the effect of upstream's `label[0] OR prediction` scoring rule (PRD 4.2)."""
from __future__ import annotations

import torch

from src.metrics.segmentation import ConfusionMeter, change_detection_outputs


class OrRuleMeter:
    """Scores the same predictions with and without the OR rule and counts the pixels the OR forces positive."""

    def __init__(self) -> None:
        """Create empty pixel meters for both scoring variants."""
        self.reset()

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch: logits [B, t-1, 2, H, W], targets [B, t, H, W]."""
        out_or = change_detection_outputs(logits, targets, use_or=True)
        out_plain = change_detection_outputs(logits, targets, use_or=False)
        self.with_or.update(out_or["score_pred"], out_or["score_target"])
        self.without_or.update(out_plain["score_pred"], out_plain["score_target"])
        prior = targets[:, :-1].flatten(end_dim=1).eq(1)
        flipped = prior & out_plain["score_pred"].eq(0)
        self.n_pixels += prior.numel()
        self.n_prior_positive += int(prior.sum())
        self.n_forced_fp += int((flipped & out_or["score_target"].eq(0)).sum())
        self.n_forced_tp += int((flipped & out_or["score_target"].eq(1)).sum())

    def compute(self) -> dict:
        """Return both score sets plus the OR's IoU delta and forced-pixel counts."""
        return or_rule_effect(self.with_or, self.without_or) | {
            "n_pixels": self.n_pixels,
            "n_prior_positive": self.n_prior_positive,
            "n_forced_fp": self.n_forced_fp,
            "n_forced_tp": self.n_forced_tp,
        }

    def reset(self) -> None:
        """Clear both meters and all counts."""
        self.with_or = ConfusionMeter()
        self.without_or = ConfusionMeter()
        self.n_pixels = 0
        self.n_prior_positive = 0
        self.n_forced_fp = 0
        self.n_forced_tp = 0


def or_rule_effect(meter_with_or: ConfusionMeter, meter_without_or: ConfusionMeter) -> dict:
    """Return both score dicts and the IoU/F1 change caused by the OR rule (with minus without)."""
    with_or = meter_with_or.compute()
    without_or = meter_without_or.compute()
    return {
        "with_or": with_or,
        "without_or": without_or,
        "iou_delta": with_or["iou"] - without_or["iou"],
        "f1_delta": with_or["f1"] - without_or["f1"],
    }
