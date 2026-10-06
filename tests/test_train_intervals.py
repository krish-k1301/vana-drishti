"""Padded / invalid label intervals are excluded from the loss, every meter and the logged counts."""
import pytest
import torch
from torch import nn

from src.data.collate import collate_batch
from src.losses.build_loss import build_loss
from src.models.segment_network import SegmentNetwork
from src.training.meters import select_intervals
from src.training.module import ChangeDetectionModule

OPTIM = {"name": "adamw"}  # not used: no optimiser is built in these tests


class SumBackbone(nn.Module):
    """1x1 conv over the sum of the window's frames: zero padding frames cannot change it."""

    def __init__(self) -> None:
        """Seeded 2->2 1x1 conv."""
        super().__init__()
        torch.manual_seed(0)
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Logits [B,2,H,W]."""
        return self.conv(images.sum(dim=1))


def sample(n_labels: int, seed: int) -> dict:
    """Unpadded item with images every 20 days and labels at days 1, 100, 200 (first n_labels)."""
    gen = torch.Generator().manual_seed(seed)
    days = torch.arange(1, 220, 20)
    return {
        "Images": torch.randn((len(days), 2, 8, 8), generator=gen),
        "ImageDays": days,
        "TargetDays": torch.tensor([1, 100, 200])[:n_labels],
        "Targets": torch.randint(0, 2, (n_labels, 8, 8), generator=gen),
        "Index": torch.tensor(seed),
    }


def module(loss_cfg: dict) -> ChangeDetectionModule:
    """Module with the deterministic backbone, trailing margin 0 and inclusive end (Phase 4 window)."""
    net = SegmentNetwork(SumBackbone(), error_days_before=30, error_days_after=0, inclusive_end=True)
    return ChangeDetectionModule(net, build_loss(loss_cfg), OPTIM)


@pytest.mark.parametrize("loss_cfg", [{"name": "cross_entropy"}, {"name": "focal", "alpha": 0.75, "gamma": 1.0}])
def test_mixed_label_counts_equal_separate_scoring(loss_cfg: dict) -> None:
    """A batch mixing t=3 and t=2 samples gives the loss and meters of scoring the valid intervals alone."""
    long, short = sample(3, 1), sample(2, 2)
    batch = collate_batch([long, short])
    assert batch["TargetDays"].tolist() == [[1, 100, 200], [1, 100, 0]]
    mixed = module(loss_cfg)
    with torch.no_grad():
        loss_mixed = mixed._step(batch, "validation")
    separate = module(loss_cfg)
    with torch.no_grad():
        loss_long = separate._step(collate_batch([long]), "validation")
        loss_short = separate._step(collate_batch([short]), "validation")
    assert torch.allclose(loss_mixed, (2 * loss_long + loss_short) / 3)  # equal pixels per interval
    got, expected = mixed.meters["validation"].compute(), separate.meters["validation"].compute()
    for level in ("pixel", "patch"):
        assert got[level] == expected[level]
    assert got["or_rule"] == expected["or_rule"]
    assert got["or_rule"]["n_pixels"] == 3 * 8 * 8
    assert got["skipped_intervals"] == 1 and expected["skipped_intervals"] == 0


def test_padded_interval_never_reaches_loss_or_meters() -> None:
    """Changing the padded interval's logits or targets changes nothing that is scored."""
    batch = collate_batch([sample(3, 1), sample(2, 2)])
    net = SegmentNetwork(SumBackbone(), 30, 0, inclusive_end=True)
    mask = net.interval_mask(batch["ImageDays"], batch["TargetDays"], batch["PadMask"])
    assert mask.tolist() == [[True, True], [True, False]]
    with torch.no_grad():
        logits = net(batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])
    assert torch.equal(logits[1, 1], torch.zeros_like(logits[1, 1]))
    kept_logits, kept_targets = select_intervals(logits, batch["Targets"], mask)
    poisoned_logits, poisoned_targets = logits.clone(), batch["Targets"].clone()
    poisoned_logits[1, 1] = 50.0
    poisoned_targets[1, 2] = 1
    again_logits, again_targets = select_intervals(poisoned_logits, poisoned_targets, mask)
    assert torch.equal(kept_logits, again_logits) and torch.equal(kept_targets, again_targets)
    assert kept_logits.shape == (3, 1, 2, 8, 8) and kept_targets.shape == (3, 2, 8, 8)
