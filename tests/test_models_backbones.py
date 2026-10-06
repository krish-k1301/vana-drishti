"""build_model factory: every registered model, config validation, wrappers vs upstream, U-TAE shapes."""
import copy

import pytest
import torch

from src.models.build_model import BACKBONES, build_backbone, build_model
from src.models.inspection import count_parameters, describe_utae_shapes
from src.models.unet3d import TimePaddedUNet3D
from src.models.vendor import import_upstream
from tests.test_models_segment import UTAE_PARAMS, utae_cfg

UPSTREAM = import_upstream()
RECURRENT_PARAMS = {"num_classes": 2, "input_size": [16, 16], "input_dim": 2, "kernel_size": [3, 3], "pad_value": 0}
PARAMS = {
    "utae": {**UTAE_PARAMS, "positional_encoding_period": 1000},
    "utae_seq2seq": {**UTAE_PARAMS, "positional_encoding_period": 1000},
    "convlstm": {**RECURRENT_PARAMS, "hidden_dim": 8},
    "convgru": {**RECURRENT_PARAMS, "hidden_dim": 8},
    "unet3d": {"in_channel": 2, "n_classes": 2, "feats": 2, "pad_value": 0, "zero_pad": True},
}


def model_cfg(name: str) -> dict:
    """Model section for `name` with small test-sized parameters."""
    return {**utae_cfg(), "name": name, "params": copy.deepcopy(PARAMS[name])}


def inputs(t: int = 5, hw: int = 16, pad_last: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random images [2,t,2,hw,hw] and days 1..t, the second sample with `pad_last` padded dates."""
    gen = torch.Generator().manual_seed(1)
    images = torch.randn((2, t, 2, hw, hw), generator=gen)
    days = torch.arange(1, t + 1).repeat(2, 1)
    if pad_last:
        images[1, -pad_last:] = 0
        days[1, -pad_last:] = 0
    return images, days


def test_registry_names() -> None:
    """The Tier A names required by INTERFACES.md are registered."""
    assert set(BACKBONES) == {"utae", "utae_seq2seq", "convlstm", "convgru", "unet3d"}


@pytest.mark.parametrize("name", sorted(PARAMS))
def test_every_model_gives_segment_logits(name: str) -> None:
    """Each model maps a padded batch to [B, t-1, 2, H, W] finite logits."""
    torch.manual_seed(0)
    net = build_model(model_cfg(name)).eval()
    images, days = inputs(t=6, pad_last=2)
    target_days = torch.tensor([[2, 4, 6], [1, 2, 4]])
    with torch.no_grad():
        out = net(images, days, target_days, days == 0)
    assert out.shape == (2, 2, 2, 16, 16)
    assert torch.isfinite(out).all()
    assert count_parameters(net) == sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_heads": 8}, "no constructor argument"),
        ({"encoder": True}, "not configurable"),
        ({"pad_value": 1}, "pad_value must be 0"),
    ],
)
def test_bad_utae_params_rejected(change: dict, message: str) -> None:
    """Typos, output-changing flags and non-zero padding are rejected instead of swallowed by **kwargs."""
    cfg = model_cfg("utae")
    cfg["params"].update(change)
    with pytest.raises(ValueError, match=message):
        build_model(cfg)


def test_missing_keys_rejected() -> None:
    """Missing backbone arguments or model-section keys are errors (no silent defaults)."""
    cfg = model_cfg("utae")
    del cfg["params"]["n_head"]
    with pytest.raises(ValueError, match="missing required"):
        build_model(cfg)
    cfg = model_cfg("convgru")
    del cfg["params"]["hidden_dim"]
    with pytest.raises(ValueError, match="missing required"):
        build_model(cfg)
    cfg = model_cfg("utae")
    del cfg["error_days_after"]
    with pytest.raises(ValueError, match="missing"):
        build_model(cfg)
    with pytest.raises(ValueError, match="only 'segment'"):
        build_model({**model_cfg("utae"), "forward_type": "use_all_info"})


def test_positional_period_applied_to_every_encoder() -> None:
    """positional_encoding_period reaches the L-TAE (and the sLTAE for seq2seq)."""
    for name, expected_count in (("utae", 1), ("utae_seq2seq", 2)):
        params = {**PARAMS[name], "positional_encoding_period": 500}
        encoders = [m for m in build_backbone(name, params).modules()
                    if isinstance(m, UPSTREAM.utae.positional_encoder.PositionalEncoder)]
        assert len(encoders) == expected_count
        assert all(e.T == 500 for e in encoders)


@pytest.mark.parametrize(
    ("name", "upstream_cls"),
    [("convlstm", UPSTREAM.utae.convlstm.ConvLSTM_Seg), ("convgru", UPSTREAM.utae.convgru.ConvGRU_Seg)],
)
def test_recurrent_matches_upstream_and_ignores_padding(name: str, upstream_cls: type) -> None:
    """Equal to upstream without padding; with padding, equal to running the sample alone unpadded."""
    torch.manual_seed(0)
    ours = build_backbone(name, PARAMS[name]).eval()
    upstream_params = {k: tuple(v) if isinstance(v, list) else v for k, v in PARAMS[name].items()}
    upstream = upstream_cls(**upstream_params).eval()
    upstream.load_state_dict(ours.state_dict(), strict=True)
    images, days = inputs()
    with torch.no_grad():
        assert torch.equal(ours(images, days), upstream(images, days))
        padded, padded_days = inputs(pad_last=2)
        alone = ours(padded[1:, :3], padded_days[1:, :3])
        assert torch.allclose(ours(padded, padded_days)[1:], alone, atol=1e-6)
        assert not torch.allclose(upstream(padded, padded_days)[1:], alone, atol=1e-6)


def test_unet3d_pads_time_and_matches_upstream() -> None:
    """Any T >= 1 works; for T a multiple of 4 without padding the output equals upstream UNet3D."""
    torch.manual_seed(0)
    ours = build_backbone("unet3d", PARAMS["unet3d"]).eval()
    assert isinstance(ours, TimePaddedUNet3D)
    upstream = UPSTREAM.utae.unet3d.UNet3D(**PARAMS["unet3d"]).eval()
    upstream.load_state_dict(ours.state_dict(), strict=True)
    with torch.no_grad():
        images, days = inputs(t=8)
        assert torch.equal(ours(images, days), upstream(images.clone(), days))
        for t in (1, 2, 3, 5):
            images, days = inputs(t=t)
            assert ours(images, days).shape == (2, 2, 16, 16)
        with pytest.raises(RuntimeError):
            upstream(*inputs(t=3))


def test_utae_bottleneck_is_6x6_for_48x48() -> None:
    """PRD 4.2: 1 input conv + 3 strided downsamples, so 48 -> 24 -> 12 -> 6."""
    shapes = describe_utae_shapes(build_backbone("utae", PARAMS["utae"]), 48, 48)
    assert shapes["in_conv"] == (64, 48, 48)
    assert shapes["down_1"] == (64, 24, 24)
    assert shapes["down_2"] == (64, 12, 12)
    assert shapes["down_3"] == (128, 6, 6)
    assert shapes["temporal_encoder"] == (128, 6, 6)
    assert shapes["out_conv"] == (2, 48, 48)
