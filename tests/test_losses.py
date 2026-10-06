"""Losses: FocalLoss equals upstream FocalLoss for the same alpha/gamma; build_loss validation."""

import pytest
import torch

from src.losses.build_loss import build_loss
from src.losses.focal import FocalLoss
from src.models.vendor import import_upstream

UpstreamFocal = import_upstream().loss.focal_loss.FocalLoss


@pytest.mark.parametrize(("alpha", "gamma"), [(0.75, 1.0), (0.75, 2.0), (0.25, 0.0), (0.5, 3.5)])
@pytest.mark.parametrize("target_kind", ["random", "all_positive", "all_negative"])
def test_focal_matches_upstream(alpha: float, gamma: float, target_kind: str) -> None:
    """Value and gradient agree with upstream on random logits, including single-class targets."""
    gen = torch.Generator().manual_seed(3)
    logits = (torch.randn((4, 2, 12, 12), generator=gen) * 4).requires_grad_(True)
    target = {
        "random": torch.randint(0, 2, (4, 12, 12), generator=gen),
        "all_positive": torch.ones((4, 12, 12), dtype=torch.long),
        "all_negative": torch.zeros((4, 12, 12), dtype=torch.long),
    }[target_kind]
    ours = FocalLoss(alpha=alpha, gamma=gamma)(logits, target)
    (grad_ours,) = torch.autograd.grad(ours, logits)
    upstream = UpstreamFocal(alpha=alpha, gamma=gamma)(logits, target)
    (grad_upstream,) = torch.autograd.grad(upstream, logits)
    assert torch.allclose(ours, upstream, rtol=1e-6, atol=1e-7)
    assert torch.allclose(grad_ours, grad_upstream, rtol=1e-5, atol=1e-8)


def test_focal_has_no_defaults_and_needs_two_classes() -> None:
    """alpha and gamma must be given (PRD 4.4 issue 4); only binary logits are accepted."""
    with pytest.raises(TypeError):
        FocalLoss()  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="2 classes"):
        FocalLoss(alpha=0.75, gamma=1.0)(torch.randn(1, 3, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long))


def test_build_loss() -> None:
    """build_loss returns CE or focal with the configured values and rejects bad sections."""
    assert isinstance(build_loss({"name": "cross_entropy"}), torch.nn.CrossEntropyLoss)
    focal = build_loss({"name": "focal", "alpha": 0.75, "gamma": 1.0})
    assert isinstance(focal, FocalLoss) and (focal.alpha, focal.gamma) == (0.75, 1.0)
    with pytest.raises(ValueError, match=r"missing keys \['gamma'\]"):
        build_loss({"name": "focal", "alpha": 0.75})
    with pytest.raises(ValueError, match="unknown keys"):
        build_loss({"name": "cross_entropy", "gamma": 1.0})
    with pytest.raises(ValueError, match="loss.name"):
        build_loss({"name": "dice"})
