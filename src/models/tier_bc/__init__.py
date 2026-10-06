"""Tier B/C benchmark backbones (PRD 6): TSViT, Exchanger + U-Net, AnySat (weights BLOCKED), Galileo.

`TIER_BC_BACKBONES` maps a `model.name` to a builder taking `model.params`, validated exactly like
`src.models.build_model._build_plain` (every constructor argument must be in config, nothing else).
Wrapper modules are imported only when their model is built: each pulls in vendored upstream code with its
own dependencies (Exchanger needs detectron2 and torch-scatter), so importing this package stays cheap and a
missing optional dependency only affects the model that needs it.
"""
import importlib
from collections.abc import Callable

from torch import nn

_WRAPPERS: dict[str, tuple[str, str]] = {
    "tsvit": ("src.models.tier_bc.tsvit", "TSViTSeg"),
    "exchanger_unet": ("src.models.tier_bc.exchanger_unet", "ExchangerUNet"),
    "anysat": ("src.models.tier_bc.anysat", "AnySatSeg"),
    "galileo": ("src.models.tier_bc.galileo", "GalileoSeg"),
}


def wrapper_class(name: str) -> type:
    """Import and return the wrapper class registered under `name`."""
    module_name, class_name = _WRAPPERS[name]
    return getattr(importlib.import_module(module_name), class_name)


def _lazy_plain(name: str) -> Callable[[dict], nn.Module]:
    """Builder that imports the wrapper on first use, then validates and builds like `_build_plain`."""

    def build(params: dict) -> nn.Module:
        """Validate `params` against the wrapper constructor, then build."""
        # Deferred import: build_model imports this package to register the Tier B/C names.
        from src.models.build_model import _build_plain

        return _build_plain(wrapper_class(name))(params)

    return build


TIER_BC_BACKBONES: dict[str, Callable[[dict], nn.Module]] = {name: _lazy_plain(name) for name in _WRAPPERS}
