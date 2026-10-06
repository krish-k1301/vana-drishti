"""TemporalSubsample modes, including equivalence with upstream TemporalDropout."""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch

from src.data import BraDDDataset, TemporalSubsample, build_dataloaders
from tests.fixtures import data_config

UPSTREAM = Path(__file__).resolve().parents[1] / "third_party" / "bradd_s1ts"


def _upstream_dropout():
    """Import upstream TemporalDropout (stub `turtle`, which sltae.py imports and may lack tkinter)."""
    sys.modules.setdefault("turtle", types.SimpleNamespace(forward=None))
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    from source.dataset import TemporalDropout
    return TemporalDropout


def _toy(total: int) -> dict:
    return {"Images": torch.arange(total, dtype=torch.float32).view(total, 1, 1, 1).expand(total, 2, 3, 3),
            "ImageDays": torch.arange(1, total + 1), "Targets": torch.zeros(2, 3, 3, dtype=torch.long)}


@pytest.mark.parametrize("total,num", [(19, 5), (31, 10), (40, 2), (23, 22), (50, 50), (63, 7)])
def test_uniform_equals_upstream(total, num):
    """Mode 'uniform' selects exactly the dates upstream TemporalDropout(is_random=True) keeps."""
    upstream = _upstream_dropout()(is_random=True, num_temporal=num)(_toy(total))
    ours = TemporalSubsample("uniform", num)(_toy(total))
    assert torch.equal(ours["ImageDays"], upstream["ImageDays"])
    assert torch.equal(ours["Images"], upstream["Images"])


def test_uniform_on_fixture_sample_matches_upstream(bradd_root):
    """Same equivalence on a real dataset item from the fixture."""
    ds = BraDDDataset(bradd_root, "train", normalization="none")
    item = ds[0]
    upstream = _upstream_dropout()(is_random=True, num_temporal=6)(dict(item))
    assert torch.equal(TemporalSubsample("uniform", 6)(item)["ImageDays"], upstream["ImageDays"])


def test_random_keeps_ends_and_is_seeded():
    """Mode 'random' keeps first/last, varies between calls and is reproducible."""
    a, b = TemporalSubsample("random", 6, seed=3), TemporalSubsample("random", 6, seed=3)
    draws = [a.indices(30) for _ in range(5)]
    assert draws == [b.indices(30) for _ in range(5)]
    assert all(d[0] == 0 and d[-1] == 29 and len(set(d)) == 6 and d == sorted(d) for d in draws)
    assert len({tuple(d) for d in draws}) > 1
    a.reseed(3)
    assert a.indices(30) == draws[0]


def test_last_only_and_none():
    """Mode 'last_only' keeps only the last date; 'none' keeps all; labels untouched."""
    item = _toy(12)
    last = TemporalSubsample("last_only", 1)(item)
    assert last["ImageDays"].tolist() == [12] and last["Images"].shape[0] == 1
    assert TemporalSubsample("none")(item)["ImageDays"].shape[0] == 12
    assert last["Targets"] is item["Targets"]


def test_num_dates_at_least_length_keeps_all():
    """Asking for more dates than exist keeps every date."""
    for mode in ("uniform", "random"):
        assert TemporalSubsample(mode, 40).indices(25) == list(range(25))


def test_invalid_arguments():
    """Unknown modes and num_dates < 2 for uniform are rejected."""
    with pytest.raises(ValueError):
        TemporalSubsample("uniform", 1)
    with pytest.raises(ValueError):
        TemporalSubsample("bogus", 5)


def test_workers_reseed_random_mode(bradd_root):
    """Random subsampling with worker processes is reproducible across runs."""
    cfg = data_config("bradd", root=str(bradd_root), normalization="none", batch_size=2, num_workers=2, seed=7,
                      temporal_subsample={"mode": "random", "num_dates": 4, "at_test": False})
    runs = [[b["ImageDays"].tolist() for b in build_dataloaders(cfg)["train"]] for _ in range(2)]
    assert runs[0] == runs[1]
