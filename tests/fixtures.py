"""Test-side access to the synthetic BraDD-format generator and the repo's data configs."""
from __future__ import annotations

from pathlib import Path

import yaml

from src.data.synthetic import make_bradd_fixture

REPO_ROOT = Path(__file__).resolve().parents[1]

__all__ = ["data_config", "make_bradd_fixture"]


def data_config(name: str, **overrides: object) -> dict:
    """The repo's `configs/data/<name>.yaml` data section (without `inspect`) with top-level overrides."""
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "data" / f"{name}.yaml").read_text())
    cfg.pop("inspect", None)
    cfg.setdefault("seed", 42)
    cfg.update(overrides)
    return cfg
