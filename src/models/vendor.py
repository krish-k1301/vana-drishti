"""Single entry point for importing the vendored BraDD-S1TS code in `third_party/bradd_s1ts`."""
import importlib
import sys
import types
from pathlib import Path

UPSTREAM_DIR = Path(__file__).resolve().parents[2] / "third_party" / "bradd_s1ts"

_SUBMODULES = (
    "source.network",
    "source.data_module",
    "source.utae.utae",
    "source.utae.ltae",
    "source.utae.sltae",
    "source.utae.positional_encoder",
    "source.utae.convlstm",
    "source.utae.convgru",
    "source.utae.unet3d",
    "source.loss.focal_loss",
)


def _install_turtle_stub() -> None:
    """Install a stub `turtle` module when the real one cannot be imported (no tkinter).

    Upstream `source/utae/sltae.py` does `from turtle import forward` and never uses it.
    """
    try:
        importlib.import_module("turtle")
    except ImportError:
        stub = types.ModuleType("turtle")
        stub.forward = None
        sys.modules["turtle"] = stub


def import_upstream() -> types.ModuleType:
    """Put the vendored repo first on sys.path, stub `turtle` if needed, and return the `source` package.

    `source/__init__.py` imports the whole upstream package (lightning, pandas, torchvision), so those
    must be installed. The submodules listed in `_SUBMODULES` are imported eagerly so callers can use
    attribute access such as `source.utae.convgru.ConvGRU_Seg`.
    """
    upstream = str(UPSTREAM_DIR)
    if upstream in sys.path:
        sys.path.remove(upstream)
    sys.path.insert(0, upstream)
    _install_turtle_stub()
    source = importlib.import_module("source")
    for name in _SUBMODULES:
        importlib.import_module(name)
    return source
