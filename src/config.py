"""YAML config loading with `base:` inheritance, dotted CLI overrides and strict key lookup.

Rules:
  * A mapping (top level or nested) may contain `base: <path>`. The referenced YAML file, resolved
    relative to the directory of the file that names it, is loaded first and the mapping's other keys
    are deep-merged on top. Base files may themselves use `base:`.
  * Deep merge: mappings merge key by key, everything else (scalars, lists) is replaced.
  * A mapping containing `_replace: true` replaces the inherited mapping instead of merging into it
    (used when a model config swaps the whole `model.params` block).
  * Overrides `a.b.c=value` parse `value` with yaml.safe_load and may only change keys that exist.
    YAML 1.1 reads `1e-3` as a string; write floats with a dot (`1.0e-3`), in files and overrides.
No defaults are filled in anywhere: a missing key is an error (see `require`).
"""
import copy
from pathlib import Path
from typing import Any

import yaml

BASE_KEY = "base"
REPLACE_KEY = "_replace"


def deep_merge(base: Any, update: Any) -> Any:
    """Return `base` deep-merged with `update` (update wins; `_replace: true` blocks merging)."""
    if not isinstance(base, dict) or not isinstance(update, dict):
        return copy.deepcopy(update)
    if update.get(REPLACE_KEY) is True:
        return {k: copy.deepcopy(v) for k, v in update.items() if k != REPLACE_KEY}
    merged = copy.deepcopy(base)
    for key, value in update.items():
        merged[key] = deep_merge(merged[key], value) if key in merged else _strip(value)
    return merged


def _strip(value: Any) -> Any:
    """Remove `_replace` markers from a subtree that has nothing to replace."""
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {k: _strip(v) for k, v in value.items() if k != REPLACE_KEY}


def _resolve(node: Any, directory: Path, chain: tuple[Path, ...]) -> Any:
    """Expand every `base:` key inside `node`, recursing into nested mappings."""
    if not isinstance(node, dict):
        return node
    resolved = {k: _resolve(v, directory, chain) for k, v in node.items() if k != BASE_KEY}
    if BASE_KEY not in node:
        return resolved
    base_cfg = _load_file((directory / node[BASE_KEY]).resolve(), chain)
    return deep_merge(base_cfg, resolved)


def _load_file(path: Path, chain: tuple[Path, ...]) -> Any:
    """Load one YAML file and resolve its bases, refusing inheritance cycles."""
    if path in chain:
        cycle = " -> ".join(str(p) for p in (*chain, path))
        raise ValueError(f"config inheritance cycle: {cycle}")
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open() as handle:
        raw = yaml.safe_load(handle)
    return _resolve(raw if raw is not None else {}, path.parent, (*chain, path))


def apply_override(cfg: dict, override: str) -> None:
    """Apply one `dotted.key=value` override in place; the key must already exist."""
    if "=" not in override:
        raise ValueError(f"override '{override}' is not of the form key=value")
    dotted, raw_value = override.split("=", 1)
    *parents, leaf = dotted.split(".")
    node = require(cfg, ".".join(parents)) if parents else cfg
    if not isinstance(node, dict) or leaf not in node:
        raise KeyError(f"override '{override}': config has no key '{dotted}'")
    node[leaf] = yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Load `path` with its bases resolved, then apply `overrides` in order."""
    cfg = _strip(_load_file(Path(path).resolve(), ()))
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a mapping at the top level")
    for override in overrides or []:
        apply_override(cfg, override)
    return cfg


def require(cfg: dict, dotted: str) -> Any:
    """Return `cfg[a][b][c]` for `dotted='a.b.c'`, raising KeyError naming the first missing part."""
    node: Any = cfg
    walked: list[str] = []
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            where = ".".join(walked) or "<root>"
            raise KeyError(f"missing config key '{dotted}' (no '{part}' under '{where}')")
        node = node[part]
        walked.append(part)
    return node
