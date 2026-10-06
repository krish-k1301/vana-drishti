"""Tests for src.metrics.segmentation and src.metrics.or_rule, including equivalence with upstream."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.metrics.or_rule import OrRuleMeter, or_rule_effect  # noqa: E402
from src.metrics.segmentation import ConfusionMeter, PatchMeter, change_detection_outputs  # noqa: E402

UPSTREAM = REPO / "third_party" / "bradd_s1ts" / "source"


def _load_upstream(name: str):
    """Load one upstream module file directly (both only import torch), bypassing `source/__init__.py`."""
    spec = importlib.util.spec_from_file_location(f"upstream_{name}", UPSTREAM / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_batch(seed: int, b: int = 3, t: int = 4, hw: int = 8):
    gen = torch.Generator().manual_seed(seed)
    logits = torch.randn((b, t - 1, 2, hw, hw), generator=gen)
    targets = torch.randint(0, 2, (b, t, hw, hw), generator=gen).long()
    return logits, targets


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_change_detection_matches_upstream(seed) -> None:
    """change_detection_outputs equals the upstream ChangeDetection forward function."""
    forward = _load_upstream("forward_function").ForwardFunction("ChangeDetection")
    logits, targets = _random_batch(seed)
    batch = {"Images": None, "ImageDays": None, "TargetDays": None, "Targets": targets}
    ref = forward(batch, lambda *_: logits)
    ours = change_detection_outputs(logits, targets)
    for up_key, our_key in [("LossPred", "loss_pred"), ("LossTarget", "loss_target"),
                            ("ScorePred", "score_pred"), ("ScoreTarget", "score_target")]:
        assert torch.equal(ref[up_key], ours[our_key]), up_key


def test_confusion_meter_matches_upstream_scores_over_batches() -> None:
    """ConfusionMeter accumulates the same matrix and scores as upstream score.py."""
    scorer = _load_upstream("score").SegmentationScores(numClass=2, evalScores=False)
    meter = ConfusionMeter()
    for seed in range(4):
        out = change_detection_outputs(*_random_batch(seed))
        scorer(out["score_pred"], out["score_target"])
        meter.update(out["score_pred"], out["score_target"])
    cm = scorer._confusion_matrix
    ours = meter.compute()
    assert torch.equal(meter.matrix, cm)
    assert (ours["tp"], ours["fp"], ours["fn"], ours["tn"]) == (
        int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0]), int(cm[0, 0]))
    ref = scorer.reset()
    for up_key, our_key in [("IoU", "iou"), ("F1", "f1"), ("Precision", "precision"), ("Recall", "recall")]:
        assert ours[our_key] == pytest.approx(float(ref[up_key]), rel=1e-6)


def test_confusion_meter_hand_case_and_reset() -> None:
    """ConfusionMeter counts and scores a hand-checked case and resets to zero."""
    meter = ConfusionMeter()
    meter.update(torch.tensor([1, 1, 0, 0, 1]), torch.tensor([1, 0, 1, 0, 1]))
    out = meter.compute()
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (2, 1, 1, 1)
    assert out["iou"] == pytest.approx(0.5)
    assert out["f1"] == pytest.approx(2 / 3)
    assert out["precision"] == pytest.approx(2 / 3)
    meter.reset()
    assert meter.compute()["tp"] == 0


def test_confusion_meter_shape_mismatch() -> None:
    """ConfusionMeter rejects predictions and targets of different shapes."""
    with pytest.raises(ValueError):
        ConfusionMeter().update(torch.zeros(3, dtype=torch.long), torch.zeros(4, dtype=torch.long))


def test_patch_meter_hand_case() -> None:
    """PatchMeter calls a patch positive if any pixel is positive (hand-checked TP/FP/FN/TN)."""
    pred = torch.zeros((4, 3, 3), dtype=torch.long)
    target = torch.zeros((4, 3, 3), dtype=torch.long)
    pred[0, 0, 0] = 1
    target[0, 2, 2] = 1  # patch 0: TP (positive anywhere counts)
    pred[1, 1, 1] = 1  # patch 1: FP
    target[2, 0, 1] = 1  # patch 2: FN; patch 3: TN
    meter = PatchMeter()
    meter.update(pred, target)
    out = meter.compute()
    assert (out["tp"], out["fp"], out["fn"], out["tn"]) == (1, 1, 1, 1)
    assert out["iou"] == pytest.approx(1 / 3)


def test_use_or_false_is_plain_argmax() -> None:
    """Without the OR rule the scored prediction is the plain argmax."""
    logits, targets = _random_batch(5)
    out = change_detection_outputs(logits, targets, use_or=False)
    assert torch.equal(out["score_pred"], logits.argmax(2).flatten(end_dim=1))


def test_or_rule_forces_false_positives_on_one_to_zero_pixels() -> None:
    """Real data: 0.0% 1->1 and 0.87% 1->0 pixels, so the OR can only add FPs (DECISIONS.md)."""
    targets = torch.zeros((1, 2, 2, 4), dtype=torch.long)
    targets[0, 1, 0, :2] = 1  # 0 -> 1 change: true deforestation, 2 px
    targets[0, 0, 1, :2] = 1  # 1 -> 0: label[0] = 1, label[1] = 0, 2 px
    logits = torch.zeros((1, 1, 2, 2, 4))
    logits[0, 0, 1, 0, :2] = 1.0  # model predicts change exactly on the 0 -> 1 pixels
    meter = OrRuleMeter()
    meter.update(logits, targets)
    out = meter.compute()
    assert out["without_or"]["iou"] == pytest.approx(1.0)
    assert out["with_or"]["fp"] == 2 and out["with_or"]["iou"] == pytest.approx(0.5)
    assert out["iou_delta"] == pytest.approx(-0.5)
    assert (out["n_prior_positive"], out["n_forced_fp"], out["n_forced_tp"]) == (2, 2, 0)
    assert out["n_pixels"] == 8


def test_or_rule_effect_from_plain_meters() -> None:
    """or_rule_effect reports the IoU difference between with-OR and without-OR meters."""
    with_or, without_or = ConfusionMeter(), ConfusionMeter()
    with_or.update(torch.tensor([1, 1]), torch.tensor([1, 0]))
    without_or.update(torch.tensor([1, 0]), torch.tensor([1, 0]))
    out = or_rule_effect(with_or, without_or)
    assert out["iou_delta"] == pytest.approx(0.5 - 1.0, abs=1e-5)
