"""Pixel and patch scoring that matches upstream BraDD-S1TS `ForwardFunction` and `SegmentationScores`."""
from __future__ import annotations

import torch

EPSILON = 1e-6


def change_detection_outputs(logits: torch.Tensor, targets: torch.Tensor, use_or: bool = True) -> dict:
    """Build loss/score tensors exactly like upstream `ForwardFunction._change_detection` (optionally without the OR)."""
    prior = targets[:, :-1]
    change_targets = torch.where(prior.eq(targets[:, 1:]), 0, 1).long()
    change_pred = logits.detach().argmax(2).long()
    score_pred = prior | change_pred if use_or else change_pred
    return {
        "loss_pred": logits.flatten(end_dim=1),
        "loss_target": change_targets.flatten(end_dim=1),
        "score_pred": score_pred.flatten(end_dim=1),
        "score_target": targets[:, 1:].flatten(end_dim=1),
    }


def confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_class: int) -> torch.Tensor:
    """Return the [num_class, num_class] confusion matrix (rows = target, cols = prediction), as upstream."""
    bins = (pred.long() + target.long() * num_class).reshape(-1)
    return torch.bincount(bins, minlength=num_class**2).view(num_class, num_class).cpu()


def scores_from_matrix(cm: torch.Tensor, positive_class: int) -> dict:
    """Return integer tp/fp/fn/tn and float IoU/precision/recall/F1 of one class (upstream formulas, EPSILON 1e-6)."""
    tp = int(cm[positive_class, positive_class])
    fp = int(cm[:, positive_class].sum()) - tp
    fn = int(cm[positive_class, :].sum()) - tp
    tn = int(cm.sum()) - tp - fp - fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "iou": tp / (tp + fp + fn + EPSILON),
        "precision": tp / (tp + fp + EPSILON),
        "recall": tp / (tp + fn + EPSILON),
        "f1": tp / (tp + 0.5 * (fp + fn) + EPSILON),
    }


class ConfusionMeter:
    """Confusion matrix accumulated over the whole evaluation set (PRD 9); scores the positive class."""

    def __init__(self, num_class: int = 2, positive_class: int = 1) -> None:
        """Create an empty meter; `positive_class` is the class whose IoU/F1 is reported (1 = deforestation)."""
        self.num_class = num_class
        self.positive_class = positive_class
        self.matrix = torch.zeros((num_class, num_class), dtype=torch.long)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add one batch of integer predictions and targets of identical shape."""
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        self.matrix += confusion_matrix(pred, target, self.num_class)

    def compute(self) -> dict:
        """Return counts and scores from everything accumulated since the last reset."""
        return scores_from_matrix(self.matrix, self.positive_class)

    def reset(self) -> None:
        """Clear the accumulated matrix."""
        self.matrix.zero_()


class PatchMeter:
    """Patch-level meter: a map is positive if it has >= 1 positive pixel, for prediction and target (PRD 9)."""

    def __init__(self) -> None:
        """Create an empty binary patch meter."""
        self.meter = ConfusionMeter(num_class=2, positive_class=1)

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        """Add a batch of [N, H, W] binary maps (one per sample and label interval)."""
        if pred.shape != target.shape:
            raise ValueError(f"pred shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        patch_pred = pred.flatten(start_dim=1).ne(0).any(dim=1).long()
        patch_target = target.flatten(start_dim=1).ne(0).any(dim=1).long()
        self.meter.update(patch_pred, patch_target)

    def compute(self) -> dict:
        """Return patch-level counts and scores."""
        return self.meter.compute()

    def reset(self) -> None:
        """Clear the accumulated matrix."""
        self.meter.reset()
