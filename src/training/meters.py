"""Per-phase evaluation meters: pixel and patch scores, with and without the OR rule (PRD 4.2, 9)."""
import torch

from src.metrics.or_rule import OrRuleMeter
from src.metrics.segmentation import PatchMeter, change_detection_outputs

SCORE_KEYS = ("iou", "precision", "recall", "f1")
COUNT_KEYS = ("tp", "fp", "fn", "tn")


class PhaseMeters:
    """Accumulates loss, pixel scores (via OrRuleMeter) and patch scores for one phase over a whole epoch."""

    def __init__(self) -> None:
        """Create empty meters."""
        self.reset()

    def reset(self) -> None:
        """Clear every accumulator."""
        self.pixel = OrRuleMeter()
        self.patch_or = PatchMeter()
        self.patch_plain = PatchMeter()
        self.loss_sum = 0.0
        self.loss_batches = 0

    def update(self, logits: torch.Tensor, targets: torch.Tensor, loss: torch.Tensor | None) -> None:
        """Add one batch: logits [B,t-1,2,H,W], targets [B,t,H,W], optional scalar loss."""
        logits = logits.detach()
        self.pixel.update(logits, targets)
        with_or = change_detection_outputs(logits, targets, use_or=True)
        plain = change_detection_outputs(logits, targets, use_or=False)
        self.patch_or.update(with_or["score_pred"], with_or["score_target"])
        self.patch_plain.update(plain["score_pred"], plain["score_target"])
        if loss is not None:
            self.loss_sum += float(loss.detach())
            self.loss_batches += 1

    def compute(self) -> dict:
        """Nested results: pixel/patch x with_or/without_or score dicts, OR-rule counts, mean loss."""
        pixel = self.pixel.compute()
        return {
            "pixel": {"with_or": pixel["with_or"], "without_or": pixel["without_or"]},
            "patch": {"with_or": self.patch_or.compute(), "without_or": self.patch_plain.compute()},
            "or_rule": {k: pixel[k] for k in ("iou_delta", "f1_delta", "n_pixels", "n_prior_positive",
                                              "n_forced_fp", "n_forced_tp")},
            "loss": self.loss_sum / self.loss_batches if self.loss_batches else None,
        }


def flat_scores(results: dict, phase: str) -> dict[str, float]:
    """Flatten `PhaseMeters.compute()` to loggable scalars, e.g. `validation/pixel_iou` (with OR)."""
    flat: dict[str, float] = {}
    for level in ("pixel", "patch"):
        for variant, suffix in (("with_or", ""), ("without_or", "_without_or")):
            for key in SCORE_KEYS:
                flat[f"{phase}/{level}_{key}{suffix}"] = float(results[level][variant][key])
    flat[f"{phase}/or_iou_delta"] = float(results["or_rule"]["iou_delta"])
    if results["loss"] is not None:
        flat[f"{phase}/loss"] = float(results["loss"])
    return flat


def score_rows(results: dict) -> list[dict]:
    """One CSV row per (level, variant) with counts and scores."""
    rows = []
    for level in ("pixel", "patch"):
        for variant in ("with_or", "without_or"):
            scores = results[level][variant]
            rows.append({"level": level, "variant": variant,
                         **{k: scores[k] for k in COUNT_KEYS + SCORE_KEYS}})
    return rows
