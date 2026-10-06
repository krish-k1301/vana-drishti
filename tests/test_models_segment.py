"""SegmentNetwork: bit-identical to upstream Network('UTAE', 'segment'), windowing, padding, causality."""
import copy

import pytest
import torch
from torch import nn

from src.models.build_model import build_model
from src.models.segment_network import SegmentNetwork
from src.models.vendor import import_upstream

UTAE_PARAMS = {
    "input_dim": 2, "encoder_widths": [64, 64, 64, 128], "decoder_widths": [32, 32, 64, 128],
    "out_conv": [32, 2], "str_conv_k": 4, "str_conv_s": 2, "str_conv_p": 1, "agg_mode": "att_group",
    "encoder_norm": "group", "n_head": 8, "d_model": 256, "d_k": 4, "pad_value": 0,
    "padding_mode": "reflect",
}


def utae_cfg(before: int = 30, after: int = 30, inclusive_end: bool = False) -> dict:
    """Model section equal to configs/baseline_utae.yaml with the given margins and end bound."""
    return {
        "name": "utae", "forward_type": "segment", "error_days_before": before, "inclusive_end": inclusive_end,
        "error_days_after": after, "params": {**UTAE_PARAMS, "positional_encoding_period": 1000},
    }


def batch(hw: int = 48) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two samples, T=6, second one padded with two zero frames at the end."""
    gen = torch.Generator().manual_seed(0)
    images = torch.randn((2, 6, 2, hw, hw), generator=gen)
    days = torch.tensor([[5, 20, 60, 100, 150, 200], [40, 70, 90, 160, 0, 0]])
    pad_mask = days == 0
    images[pad_mask] = 0
    target_days = torch.tensor([[60, 120], [50, 100]])
    return images, days, target_days, pad_mask


def test_bit_identical_to_upstream_network() -> None:
    """Same weights and inputs give exactly the same logits as upstream Network (error days 30/30)."""
    torch.manual_seed(42)
    upstream_kwargs = {f"Model_{k}": copy.deepcopy(v) for k, v in UTAE_PARAMS.items()}
    upstream = import_upstream().Network("UTAE", "segment", **upstream_kwargs).eval()
    ours = build_model(utae_cfg()).eval()
    ours.backbone.load_state_dict(upstream.net.state_dict(), strict=True)
    images, days, target_days, pad_mask = batch()
    with torch.no_grad():
        expected = upstream(images.clone(), days.clone(), target_days.clone())
        got_with_mask = ours(images, days, target_days, pad_mask)
        got_without_mask = ours(images, days, target_days, None)
    assert expected.shape == (2, 1, 2, 48, 48)
    assert torch.equal(got_with_mask, expected)
    assert torch.equal(got_without_mask, expected)


def test_target_days_not_mutated_while_upstream_mutates() -> None:
    """Upstream shifts target_days in place (DECISIONS.md); our wrapper must not."""
    torch.manual_seed(0)
    images, days, target_days, pad_mask = batch(hw=32)
    upstream_kwargs = {f"Model_{k}": copy.deepcopy(v) for k, v in UTAE_PARAMS.items()}
    upstream = import_upstream().Network("UTAE", "segment", **upstream_kwargs).eval()
    upstream_targets = target_days.clone()
    ours_targets = target_days.clone()
    with torch.no_grad():
        upstream(images, days, upstream_targets)
        build_model(utae_cfg()).eval()(images, days, ours_targets, pad_mask)
    assert not torch.equal(upstream_targets, target_days)
    assert torch.equal(ours_targets, target_days)


class DayRecorder(nn.Module):
    """Fake backbone that records the days it receives and returns zeros [B,2,H,W]."""

    def __init__(self) -> None:
        """Start with no recorded calls."""
        super().__init__()
        self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def forward(self, images: torch.Tensor, days: torch.Tensor) -> torch.Tensor:
        """Record (images, days) and return zero logits."""
        self.calls.append((images.clone(), days.clone()))
        return torch.zeros(images.shape[0], 2, *images.shape[-2:])


def test_windows_use_separate_margins_and_strict_bounds() -> None:
    """Each interval keeps start - before < day < end + after, re-collated with end padding."""
    recorder = DayRecorder()
    net = SegmentNetwork(recorder, error_days_before=10, error_days_after=0)
    days = torch.tensor([[10, 20, 30, 40, 50, 60], [15, 25, 35, 45, 0, 0]])
    images = torch.randn(2, 6, 2, 4, 4)
    target_days = torch.tensor([[20, 40, 60], [25, 45, 60]])
    out = net(images, days, target_days, days == 0)
    assert out.shape == (2, 2, 2, 4, 4)
    (first_images, first_days), (second_images, second_days) = recorder.calls
    assert first_days.tolist() == [[20, 30], [25, 35]]
    assert second_days.tolist() == [[40, 50], [45, 0]]
    assert torch.equal(first_images[0], images[0, 1:3])
    assert torch.equal(second_images[1, 0], images[1, 3])
    assert torch.equal(second_images[1, 1], torch.zeros_like(images[1, 3]))


def test_image_after_window_cannot_affect_output() -> None:
    """With error_days_after=0, perturbing an image dated after the end label date changes nothing."""
    torch.manual_seed(0)
    net = build_model(utae_cfg(before=30, after=0)).eval()
    images, days, target_days, pad_mask = batch(hw=32)
    perturbed = images.clone()
    perturbed[0, 4] = torch.randn_like(perturbed[0, 4]) * 50  # day 150 > end 120
    perturbed[1, 2] = torch.randn_like(perturbed[1, 2]) * 50  # day 90 < end 100: must matter
    reference = images.clone()
    reference[1, 2] = perturbed[1, 2]
    with torch.no_grad():
        out_perturbed = net(perturbed, days, target_days, pad_mask)
        out_reference = net(reference, days, target_days, pad_mask)
        out_original = net(images, days, target_days, pad_mask)
    assert torch.equal(out_perturbed, out_reference)
    assert not torch.equal(out_perturbed, out_original)


def test_pad_mask_zeroes_padded_frames() -> None:
    """Garbage in padded frames is ignored when the explicit pad_mask is given."""
    torch.manual_seed(0)
    net = build_model(utae_cfg()).eval()
    images, days, target_days, pad_mask = batch(hw=32)
    noisy = images.clone()
    noisy[pad_mask] = torch.randn_like(noisy[pad_mask])
    days_noisy = days.clone()
    days_noisy[pad_mask] = 80  # inside the window: only the mask can exclude these dates
    with torch.no_grad():
        got = net(noisy, days_noisy, target_days, pad_mask)
        expected = net(images, days, target_days, pad_mask)
    assert torch.equal(got, expected)


def test_real_all_zero_frame_raises() -> None:
    """A real frame that looks like padding is an error, not silent garbage."""
    net = SegmentNetwork(DayRecorder(), error_days_before=30, error_days_after=30)
    images, days, target_days, pad_mask = batch(hw=8)
    zeroed = images.clone()
    zeroed[0, 2] = 0
    with pytest.raises(ValueError, match="non-padded frame"):
        net(zeroed, days, target_days, pad_mask)


def test_padded_labels_and_empty_windows_are_masked() -> None:
    """Padded label days (0) and windows without images are skipped, zero-filled and flagged invalid."""
    recorder = DayRecorder()
    net = SegmentNetwork(recorder, error_days_before=10, error_days_after=0, inclusive_end=True)
    days = torch.tensor([[10, 20, 30, 40, 50, 60], [15, 25, 35, 45, 0, 0]])
    images = torch.randn(2, 6, 2, 4, 4)
    target_days = torch.tensor([[20, 40, 60], [25, 45, 0]])
    mask = net.interval_mask(days, target_days, days == 0)
    assert mask.tolist() == [[True, True], [True, False]]
    out = net(images, days, target_days, days == 0)
    assert out.shape == (2, 2, 2, 4, 4)
    assert recorder.calls[0][1].shape[0] == 2 and recorder.calls[1][1].tolist() == [[40, 50, 60]]
    empty = torch.tensor([[200, 300, 400], [25, 45, 0]])
    assert net.interval_mask(days, empty, days == 0).tolist() == [[False, False], [True, False]]
    with pytest.raises(ValueError, match="no label interval"):
        net(images, days, torch.tensor([[200, 300, 400], [300, 400, 0]]), days == 0)


@pytest.mark.parametrize("inclusive_end", [True, False])
def test_inclusive_end_uses_cutoff_image_only_when_set(inclusive_end: bool) -> None:
    """error_days_after=0: the image at t_c matters iff inclusive_end; the image at t_c + 1 never does."""
    torch.manual_seed(0)
    net = build_model(utae_cfg(before=30, after=0, inclusive_end=inclusive_end)).eval()
    images, days, _, pad_mask = batch(hw=32)
    days = torch.tensor([[5, 20, 60, 100, 101, 200], [40, 70, 90, 91, 0, 0]])
    target_days = torch.tensor([[60, 100], [50, 90]])  # cutoffs t_c = 100 and 90
    at_cutoff, after_cutoff = images.clone(), images.clone()
    at_cutoff[0, 3] *= 5
    at_cutoff[1, 2] *= 5
    after_cutoff[0, 4] *= 5
    after_cutoff[1, 3] *= 5
    with torch.no_grad():
        reference = net(images, days, target_days, pad_mask)
        changed = net(at_cutoff, days, target_days, pad_mask)
        unchanged = net(after_cutoff, days, target_days, pad_mask)
    assert torch.equal(unchanged, reference)
    assert torch.equal(changed, reference) is not inclusive_end


def test_upstream_quirk_later_intervals_lose_lower_margin() -> None:
    """Upstream's in-place shift: for t > 2, interval i > 0 uses `start`, not start - 30, as lower bound."""
    upstream_kwargs = {f"Model_{k}": copy.deepcopy(v) for k, v in UTAE_PARAMS.items()}
    upstream = import_upstream().Network("UTAE", "segment", **upstream_kwargs)
    upstream.net = DayRecorder()
    ours = SegmentNetwork(DayRecorder(), error_days_before=30, error_days_after=30)
    days = torch.tensor([[50, 90, 110, 140]])
    images = torch.randn(1, 4, 2, 4, 4)
    target_days = torch.tensor([[60, 100, 130]])
    upstream(images, days, target_days.clone())
    ours(images, days, target_days)
    assert upstream.net.calls[1][1].tolist() == [[110, 140]]
    assert ours.backbone.calls[1][1].tolist() == [[90, 110, 140]]
