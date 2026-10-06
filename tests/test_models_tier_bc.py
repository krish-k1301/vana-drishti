"""Tier B/C backbones (PRD 6): shipped YAML builds, shapes, padding ignored, backward, upstream parity."""
from pathlib import Path

import pytest
import torch

from src.config import load_config
from src.models.segment_network import SegmentNetwork
from src.models.tier_bc import TIER_BC_BACKBONES, wrapper_class
from src.models.tier_bc.galileo import evenly_spaced_dates

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "models"
NAMES = ("tsvit", "exchanger_unet", "anysat", "galileo")
P = "model.params."
TINY = {
    "tsvit": [f"{P}dim=16", f"{P}temporal_depth=1", f"{P}spatial_depth=1", f"{P}heads=2", f"{P}dim_head=8",
              f"{P}patch_size=4"],
    "exchanger_unet": [f"{P}temporal_encoding.pe_dim=16", f"{P}exchanger.embed_dims=[16, 16]",
                       f"{P}exchanger.num_heads_list=[2, 2]", f"{P}exchanger.num_token_list=[4, 4]",
                       f"{P}unet.base_channels=8"],
    "anysat": [f"{P}model_size=tiny", f"{P}patch_size_m=80"],
    "galileo": [f"{P}patch_size=8"],
}
HW = 48


def config(name: str, tiny: bool) -> dict:
    """Full experiment config for `name`, optionally shrunk for fast tests (shipped file otherwise)."""
    return load_config(CONFIG_DIR / f"{name}.yaml", TINY[name] if tiny else None)


def build(name: str, tiny: bool = True) -> torch.nn.Module:
    """Backbone for `name` from its config's `model.params`, seeded."""
    torch.manual_seed(0)
    return TIER_BC_BACKBONES[name](config(name, tiny)["model"]["params"])


def batch() -> tuple[torch.Tensor, torch.Tensor]:
    """B=2, T=7 images [2,7,2,48,48] and days; sample 1 has its last 3 dates padded (zero frame, day 0)."""
    gen = torch.Generator().manual_seed(1)
    images = torch.randn((2, 7, 2, HW, HW), generator=gen)
    days = torch.tensor([[1, 8, 20, 33, 40, 50, 61], [3, 9, 15, 27, 0, 0, 0]])
    images[1, 4:] = 0
    return images, days


def test_registry_names() -> None:
    """Every Tier B/C wrapper is registered under its config name."""
    assert set(TIER_BC_BACKBONES) == set(NAMES)


@pytest.mark.parametrize("name", NAMES)
def test_shipped_config_builds(name: str) -> None:
    """The shipped YAML builds with no override and keeps the PRD 6 loss (focal alpha 0.75, gamma 1.0)."""
    cfg = config(name, tiny=False)
    assert cfg["model"]["name"] == name
    assert cfg["loss"] == {"name": "focal", "alpha": 0.75, "gamma": 1.0}
    model = build(name, tiny=False)
    print(f"{name}: {sum(p.numel() for p in model.parameters()):,} parameters (shipped config)")


@pytest.mark.parametrize("name", NAMES)
def test_forward_shape_padding_and_backward(name: str) -> None:
    """[B,2,48,48] finite logits; padded frames' content is ignored; gradients flow in training mode."""
    model = build(name).eval()
    images, days = batch()
    with torch.no_grad():
        out = model(images, days)
        changed = images.clone()
        changed[1, 4:] = 5.0
        out_changed = model(changed, days)
    assert out.shape == (2, 2, HW, HW)
    assert torch.isfinite(out).all()
    torch.testing.assert_close(out_changed, out, rtol=0.0, atol=1e-5)
    model.train()
    model(images, days).square().mean().backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


@pytest.mark.parametrize("name", NAMES)
def test_segment_network_wraps_backbone(name: str) -> None:
    """SegmentNetwork (the build_model contract) turns the backbone into [B, t-1, 2, H, W] logits."""
    cfg = config(name, tiny=True)["model"]
    net = SegmentNetwork(build(name), cfg["error_days_before"], cfg["error_days_after"]).eval()
    images, days = batch()
    with torch.no_grad():
        out = net(images, days, torch.tensor([[1, 61], [3, 27]]), days == 0)
    assert out.shape == (2, 1, 2, HW, HW)


def test_tsvit_matches_upstream_without_padding() -> None:
    """With no padded dates the wrapper equals upstream TSViT.forward fed day/365 as its time channel."""
    model = build("tsvit").eval()
    images, days = batch()
    images, days = images[:1], days[:1]
    time_channel = (days.float() / 365.0)[:, :, None, None, None].expand(1, 7, HW, HW, 1)
    upstream_input = torch.cat([images.permute(0, 1, 3, 4, 2), time_channel], dim=-1)
    with torch.no_grad():
        torch.testing.assert_close(model(images, days), model.tsvit(upstream_input))


def test_galileo_loads_nano_weights_strictly() -> None:
    """The vendored nano encoder weights load (strict) and the linear probe freezes the encoder."""
    params = config("galileo", tiny=True)["model"]["params"]
    model = TIER_BC_BACKBONES["galileo"]({**params, "freeze_encoder": True})
    saved = torch.load(REPO_ROOT / params["checkpoint_path"] / "encoder.pt", map_location="cpu", weights_only=True)
    state = model.encoder.state_dict()
    assert set(saved) == set(state)
    assert all(torch.equal(saved[k], state[k]) for k in saved)
    assert not any(p.requires_grad for p in model.encoder.parameters())
    assert all(p.requires_grad for p in model.head.parameters())
    model.train()
    assert not model.encoder.training and model.head.training


def test_galileo_keeps_at_most_max_timesteps() -> None:
    """Windows longer than the encoder's time table are evenly subsampled, keeping first and last real date."""
    images = torch.randn(2, 30, 2, 4, 4)
    days = torch.arange(1, 31).repeat(2, 1)
    days[1, 20:] = 0
    kept_images, kept_days = evenly_spaced_dates(images, days, 24)
    assert kept_days.shape == (2, 24)
    assert kept_days[0, 0] == 1 and kept_days[0, -1] == 30
    assert (kept_days[1, :20] == torch.arange(1, 21)).all() and (kept_days[1, 20:] == 0).all()
    torch.testing.assert_close(kept_images[1, :20], images[1, :20])


def test_anysat_checkpoint_loads_strictly(tmp_path: Path) -> None:
    """A checkpoint in the hub format ({'state_dict': ...}) round-trips through `checkpoint_path`."""
    params = config("anysat", tiny=True)["model"]["params"]
    source = TIER_BC_BACKBONES["anysat"](params)
    path = tmp_path / "AnySat.pth"
    torch.save({"state_dict": source.anysat.model.state_dict()}, path)
    loaded = TIER_BC_BACKBONES["anysat"]({**params, "checkpoint_path": str(path), "freeze_encoder": True})
    first = next(iter(source.anysat.model.state_dict()))
    assert torch.equal(loaded.anysat.model.state_dict()[first], source.anysat.model.state_dict()[first])
    assert not any(p.requires_grad for p in loaded.anysat.parameters())


@pytest.mark.parametrize(
    ("name", "change", "message"),
    [
        ("tsvit", {"depth": 4}, "no constructor argument"),
        ("galileo", {"input_dim": 3}, "VV, VH"),
        ("anysat", {"modality": "s2"}, "bands"),
    ],
)
def test_bad_params_rejected(name: str, change: dict, message: str) -> None:
    """Unknown keys and inputs the upstream model cannot take are errors."""
    params = {**config(name, tiny=True)["model"]["params"], **change}
    with pytest.raises(ValueError, match=message):
        TIER_BC_BACKBONES[name](params)


def test_exchanger_sections_are_checked() -> None:
    """Exchanger's nested sections must hold exactly the upstream keys."""
    params = config("exchanger_unet", tiny=True)["model"]["params"]
    del params["gp_block"]["mixer_depth"]
    with pytest.raises(ValueError, match="mixer_depth"):
        TIER_BC_BACKBONES["exchanger_unet"](params)


def test_missing_constructor_argument_rejected() -> None:
    """Every constructor argument must come from config (no defaults)."""
    params = config("tsvit", tiny=True)["model"]["params"]
    del params["scale_dim"]
    with pytest.raises(ValueError, match="missing required"):
        TIER_BC_BACKBONES["tsvit"](params)
    assert wrapper_class("tsvit").__name__ == "TSViTSeg"
