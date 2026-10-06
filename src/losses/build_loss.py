"""Loss factory: `build_loss(loss_cfg)` from `{name: cross_entropy | focal, alpha, gamma}`."""
from torch import nn

from src.losses.focal import FocalLoss

_KEYS = {"cross_entropy": {"name"}, "focal": {"name", "alpha", "gamma"}}


def build_loss(loss_cfg: dict) -> nn.Module:
    """Return CrossEntropyLoss() or FocalLoss(alpha, gamma); every focal hyperparameter must be given."""
    name = loss_cfg.get("name")
    if name not in _KEYS:
        raise ValueError(f"loss.name must be one of {sorted(_KEYS)}, got {name!r}")
    missing = sorted(_KEYS[name] - set(loss_cfg))
    unknown = sorted(set(loss_cfg) - _KEYS[name])
    if missing or unknown:
        raise ValueError(f"loss '{name}': missing keys {missing}, unknown keys {unknown}")
    if name == "focal":
        return FocalLoss(alpha=loss_cfg["alpha"], gamma=loss_cfg["gamma"])
    return nn.CrossEntropyLoss()
