"""Model factory: `build_model(cfg)` turns the `model` config section into a segment-wise network."""
import inspect
from collections.abc import Callable

from torch import nn

from src.models.recurrent import MaskedConvGRUSeg, MaskedConvLSTMSeg
from src.models.segment_network import PAD_VALUE, SegmentNetwork
from src.models.tier_bc import TIER_BC_BACKBONES
from src.models.unet3d import TimePaddedUNet3D
from src.models.vendor import import_upstream

_SOURCE = import_upstream()
_UTAE = _SOURCE.utae.utae.UTAE
_PositionalEncoder = _SOURCE.utae.positional_encoder.PositionalEncoder

UTAE_KEYS = (
    "input_dim", "encoder_widths", "decoder_widths", "out_conv", "str_conv_k", "str_conv_s", "str_conv_p",
    "agg_mode", "encoder_norm", "n_head", "d_model", "d_k", "pad_value", "padding_mode",
)
POSITIONAL_PERIOD_KEY = "positional_encoding_period"
_REQUIRED_MODEL_KEYS = ("name", "forward_type", "error_days_before", "error_days_after", "params")


def _check_keys(cls: type, params: dict, required: tuple[str, ...]) -> None:
    """Reject keys missing from `params`, unknown to `cls.__init__`, or not configurable for this model."""
    signature = set(inspect.signature(cls.__init__).parameters) - {"self", "kwargs", "args"}
    unknown = sorted(set(params) - signature)
    if unknown:
        raise ValueError(f"{cls.__name__} has no constructor argument(s) {unknown}")
    not_allowed = sorted(set(params) - set(required))
    if not_allowed:
        raise ValueError(f"{cls.__name__} argument(s) {not_allowed} are not configurable here")
    missing = sorted(set(required) - set(params))
    if missing:
        raise ValueError(f"{cls.__name__} config is missing required argument(s) {missing}")


def set_positional_period(model: nn.Module, period: float) -> None:
    """Rebuild every upstream PositionalEncoder in `model` with period `period` (UTAE cannot pass it)."""
    targets = [name for name, m in model.named_modules() if isinstance(m, _PositionalEncoder)]
    if not targets:
        raise ValueError("model has no PositionalEncoder to configure")
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        old = getattr(parent, attr)
        setattr(parent, attr, _PositionalEncoder(old.d, T=period, repeat=old.repeat))


def _build_utae(params: dict, seq2seq: bool) -> nn.Module:
    """Upstream UTAE with every PRD 4.2 argument taken from config."""
    utae_params = {k: v for k, v in params.items() if k != POSITIONAL_PERIOD_KEY}
    if POSITIONAL_PERIOD_KEY not in params:
        raise ValueError(f"UTAE config is missing '{POSITIONAL_PERIOD_KEY}'")
    _check_keys(_UTAE, utae_params, UTAE_KEYS)
    model = _UTAE(**utae_params, encoder=False, return_maps=False, aux_conv=None, seq2seq=seq2seq)
    set_positional_period(model, params[POSITIONAL_PERIOD_KEY])
    return model


def _build_plain(cls: type) -> Callable[[dict], nn.Module]:
    """Factory for wrappers whose constructor arguments all come from config."""
    required = tuple(p for p in inspect.signature(cls.__init__).parameters if p != "self")

    def build(params: dict) -> nn.Module:
        """Validate `params` against the constructor, then build."""
        _check_keys(cls, params, required)
        return cls(**params)

    return build


BACKBONES: dict[str, Callable[[dict], nn.Module]] = {
    "utae": lambda params: _build_utae(params, seq2seq=False),
    "utae_seq2seq": lambda params: _build_utae(params, seq2seq=True),
    "convlstm": _build_plain(MaskedConvLSTMSeg),
    "convgru": _build_plain(MaskedConvGRUSeg),
    "unet3d": _build_plain(TimePaddedUNet3D),
}
BACKBONES.update(TIER_BC_BACKBONES)


def build_backbone(name: str, params: dict) -> nn.Module:
    """Return the bare backbone mapping (images [B,T,C,H,W], days [B,T]) to logits [B,L,H,W]."""
    if name not in BACKBONES:
        raise ValueError(f"unknown model '{name}', expected one of {sorted(BACKBONES)}")
    if "pad_value" in params and params["pad_value"] != PAD_VALUE:
        raise ValueError(f"params.pad_value must be {PAD_VALUE}: SegmentNetwork pads windows with it")
    return BACKBONES[name](params)


def build_model(cfg: dict) -> nn.Module:
    """Build the network described by the `model` config section (see .orchestrator/INTERFACES.md)."""
    missing = [k for k in _REQUIRED_MODEL_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"model config is missing {missing}")
    unknown = sorted(set(cfg) - set(_REQUIRED_MODEL_KEYS))
    if unknown:
        raise ValueError(f"model config has unknown key(s) {unknown}")
    if cfg["forward_type"] != "segment":
        raise ValueError(f"forward_type '{cfg['forward_type']}' not supported; only 'segment'")
    backbone = build_backbone(cfg["name"], cfg["params"])
    return SegmentNetwork(backbone, cfg["error_days_before"], cfg["error_days_after"])
