"""Shared pytest setup: repo root on sys.path and synthetic BraDD-format fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.fixtures import make_bradd_fixture  # noqa: E402


@pytest.fixture(scope="session")
def bradd_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic BraDD dataset (SMOKE) shared by the whole test session; treat as read-only."""
    return make_bradd_fixture(tmp_path_factory.mktemp("bradd"), n_per_split={"train": 8, "validation": 4, "test": 4})


@pytest.fixture(scope="session")
def dated_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Synthetic dated early-detection dataset (SMOKE); treat as read-only."""
    return make_bradd_fixture(tmp_path_factory.mktemp("dated"), n_per_split=4, dated=True, seed=1)
