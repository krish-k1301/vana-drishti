"""YAML config loading, validation, per-mode output paths and the hash that ties a pilot QA to its settings."""
from __future__ import annotations

import hashlib
import json
import math
import os

import yaml

REQUIRED_SECTIONS = ("name", "region_label", "placeholder_prefix", "paths", "s1", "window", "grid", "events",
                     "labels", "negatives", "split", "modes", "export", "qa", "states")
HASHED_SECTIONS = ("s1", "window", "grid", "events", "labels")
MODES = ("pilot", "full")


def load_config(path: str) -> dict:
    """Load a GEE export YAML config and check that all required sections exist."""
    with open(path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    missing = [k for k in REQUIRED_SECTIONS if k not in cfg]
    if missing:
        raise KeyError(f"config {path} is missing sections: {missing}")
    for mode in MODES:
        if mode not in cfg["modes"]:
            raise KeyError(f"config {path} has no modes.{mode} section")
    check_spacing(cfg)
    return cfg


def check_spacing(cfg: dict) -> None:
    """Require min_center_distance_m >= patch diagonal, so two kept square patches can never overlap."""
    diagonal = cfg["grid"]["patch_size"] * cfg["grid"]["pixel_size_m"] * math.sqrt(2.0)
    if cfg["split"]["min_center_distance_m"] < diagonal:
        raise ValueError(f"split.min_center_distance_m must be >= the patch diagonal ({diagonal:.0f} m)")


def config_hash(cfg: dict) -> str:
    """Short SHA-256 of the sections that define preprocessing and labels (bbox/size of a run excluded)."""
    payload = json.dumps({k: cfg[k] for k in HASHED_SECTIONS}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def mode_paths(cfg: dict, mode: str) -> dict[str, str]:
    """Output locations of one run mode: out_root (BraDD root), raw_dir, qa_dir, task_log, plan."""
    paths = cfg["paths"]
    out_root = cfg["modes"][mode]["out_root"]
    raw_dir = os.path.join(out_root, paths["raw_subdir"])
    return {"out_root": out_root, "raw_dir": raw_dir, "qa_dir": os.path.join(out_root, paths["qa_subdir"]),
            "task_log": os.path.join(raw_dir, paths["task_log"]), "plan": os.path.join(raw_dir, paths["plan"])}


def _placeholders(node: object, prefix: str, path: str) -> list[str]:
    """Recursively list dotted paths whose string value starts with the placeholder prefix."""
    if isinstance(node, dict):
        return [p for k, v in node.items() for p in _placeholders(v, prefix, f"{path}.{k}" if path else str(k))]
    if isinstance(node, list):
        return [p for i, v in enumerate(node) for p in _placeholders(v, prefix, f"{path}[{i}]")]
    return [path] if isinstance(node, str) and node.startswith(prefix) else []


def unresolved_placeholders(cfg: dict, mode: str) -> list[str]:
    """Config keys still holding a placeholder that this mode would use (pulled labels, events, batch target)."""
    prefix = cfg["placeholder_prefix"]
    used = {k: cfg[k] for k in ("s1", "events", "negatives")}
    used["labels"] = {k: cfg["labels"][k] for k in set(cfg["labels"]["pull"]) | {"radd", "hansen"}}
    if cfg["export"]["backend"][mode] == "batch":
        used["export.batch"] = cfg["export"]["batch"]
    return _placeholders(used, prefix, "")
