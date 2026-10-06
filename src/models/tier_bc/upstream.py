"""Import vendored Tier B/C upstream code whose top-level package names would clash with each other.

DeepSatModels (`models`, `utils`), Exchanger4SITS (`layers`, `modules`, `ops`, `utils`, `losses`) and AnySat
(`models`) all expect their own source folder at the front of sys.path and use generic top-level names.
`import_isolated` imports them with that folder first on sys.path and with any already-imported module of
the same top-level names hidden, then restores sys.path and sys.modules. The returned module objects keep
working because every import they need is resolved at import time.
"""
import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType

THIRD_PARTY_DIR = Path(__file__).resolve().parents[3] / "third_party"


def _owned(name: str, top_level: tuple[str, ...]) -> bool:
    """True when module `name` belongs to one of the `top_level` packages."""
    return name.split(".")[0] in top_level


def _shadows(entry: str, top_level: tuple[str, ...]) -> bool:
    """True when sys.path `entry` holds a package or module named like one of `top_level`."""
    folder = Path(entry or ".")
    return any((folder / name).exists() or (folder / f"{name}.py").exists() for name in top_level)


@contextmanager
def isolated_sys_path(root: Path, top_level: tuple[str, ...]) -> Iterator[None]:
    """Make `root` the only sys.path entry providing `top_level` names and hide such modules meanwhile.

    Other entries that contain a `top_level` name (e.g. this repo's root, which holds our own `src`) are
    dropped for the duration, since a regular package there would win over a namespace package in `root`.
    """
    saved_path = list(sys.path)
    saved_modules = {name: mod for name, mod in sys.modules.items() if _owned(name, top_level)}
    for name in saved_modules:
        del sys.modules[name]
    sys.path[:] = [str(root)] + [entry for entry in saved_path if not _shadows(entry, top_level)]
    try:
        yield
    finally:
        for name in [name for name in sys.modules if _owned(name, top_level)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path


def import_isolated(root: Path, top_level: tuple[str, ...], names: tuple[str, ...]) -> dict[str, ModuleType]:
    """Import every module in `names` from `root` inside `isolated_sys_path` and return them by name."""
    if not root.is_dir():
        raise FileNotFoundError(f"vendored upstream folder not found: {root}")
    with isolated_sys_path(root, top_level):
        return {name: importlib.import_module(name) for name in names}


def import_file(path: Path, module_name: str) -> ModuleType:
    """Import a single self-contained upstream file under a private module name (cached in sys.modules)."""
    if module_name in sys.modules:
        return sys.modules[module_name]
    if not path.is_file():
        raise FileNotFoundError(f"vendored upstream file not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[module_name]
        raise
    return module
