"""Per-region normalisation: train-split stats per region, unseen-region policies, cache, cross-root use."""
from __future__ import annotations

import shutil

import pandas as pd
import pytest
import torch

from src.data import BraDDDataset, build_region_normalizer
from src.data.samples import read_meta
from src.data.stats import stats_over_files


def _spec(stats_path, unseen="global", forced=None, column="state"):
    """A complete per_region_norm block."""
    return {"enabled": True, "region_column": column, "stats_path": str(stats_path), "unseen_region": unseen,
            "self_normalise_regions": forced or []}


def _with_target_region(src, dst, phases=("test",), split=None):
    """Copy the fixture and relabel rows of the given phases as region 'congo' (optionally re-splitting)."""
    shutil.copytree(src, dst)
    meta = pd.read_csv(dst / "meta.csv", index_col=0)
    rows = meta["close_set"].isin(phases)
    meta.loc[rows, "state"] = "congo"
    if split is not None:
        meta["close_set"] = split
    meta.to_csv(dst / "meta.csv")
    return dst


def test_region_stats_come_from_train_rows(bradd_root, tmp_path):
    """Each seen region is normalised with the stats of its own train rows."""
    meta = read_meta(bradd_root)
    train = meta[meta["close_set"] == "train"]
    norm = build_region_normalizer(bradd_root, "close_set", train["state"], _spec(tmp_path / "r.pt"))
    for region, group in train.groupby("state"):
        expected = stats_over_files(bradd_root, group["file"])
        assert torch.allclose(norm.table[region]["mean"], expected["mean"]) and norm.sources[region] == "train"
    ds = BraDDDataset(bradd_root, "train", per_region_norm=_spec(tmp_path / "r.pt"))
    raw, region = ds.load_raw(0)["image"], ds.meta.loc[0, "state"]
    stats = norm.table[region]
    assert torch.allclose(ds[0]["Images"], (raw - stats["mean"].view(1, 2, 1, 1)) / stats["std"].view(1, 2, 1, 1))


@pytest.mark.parametrize("policy", ["global", "self", "error"])
def test_unseen_region_policies(bradd_root, tmp_path, policy):
    """A test-only region gets global train stats, its own images' stats, or an error."""
    root = _with_target_region(bradd_root, tmp_path / "ds")
    spec = _spec(tmp_path / "r.pt", unseen=policy)
    if policy == "error":
        with pytest.raises(ValueError, match="congo"):
            BraDDDataset(root, "test", per_region_norm=spec)
        return
    ds = BraDDDataset(root, "test", per_region_norm=spec)
    meta = read_meta(root)
    files = meta.loc[meta["state"] == "congo", "file"] if policy == "self" else meta.loc[
        meta["close_set"] == "train", "file"]
    assert torch.allclose(ds.region_norm.table["congo"]["mean"], stats_over_files(root, files)["mean"])
    assert ds.region_norm.sources["congo"] == policy


def test_forced_self_normalisation_and_cache(bradd_root, tmp_path):
    """Listed regions always self-normalise; the cache is reused and checked for its region column."""
    region = read_meta(bradd_root).query("close_set == 'train'")["state"].iloc[0]
    ds = BraDDDataset(bradd_root, "train", per_region_norm=_spec(tmp_path / "r.pt", forced=[region]))
    assert ds.region_norm.sources[region] == "self" and (tmp_path / "r.pt").exists()
    with pytest.raises(ValueError, match="region column"):
        BraDDDataset(bradd_root, "train", per_region_norm=_spec(tmp_path / "r.pt", column="sampling_type"))


def test_target_root_reuses_training_cache(bradd_root, tmp_path):
    """A cross-biome set with no train rows normalises with the Amazon cache plus its own images."""
    cache = tmp_path / "amazon.pt"
    BraDDDataset(bradd_root, "train", per_region_norm=_spec(cache))
    target = _with_target_region(bradd_root, tmp_path / "congo", ("train", "validation", "test"), split="test")
    ds = BraDDDataset(target, "test", per_region_norm=_spec(cache, unseen="self"))
    assert ds.region_norm.sources == {"congo": "self"} and ds[0]["Images"].shape[1] == 2
    with pytest.raises(ValueError, match="no train rows"):
        BraDDDataset(target, "test", per_region_norm=_spec(tmp_path / "missing.pt"))


def test_region_norm_config_errors(bradd_root, tmp_path):
    """Incomplete blocks, bad policies and normalization 'none' are rejected."""
    with pytest.raises(KeyError):
        BraDDDataset(bradd_root, "train", per_region_norm={"enabled": True, "region_column": "state"})
    with pytest.raises(ValueError):
        BraDDDataset(bradd_root, "train", per_region_norm=_spec(tmp_path / "r.pt", unseen="nearest"))
    with pytest.raises(ValueError):
        BraDDDataset(bradd_root, "train", normalization="none", per_region_norm=_spec(tmp_path / "r.pt"))
    off = dict(_spec(tmp_path / "r.pt"), enabled=False)
    assert BraDDDataset(bradd_root, "train", per_region_norm=off, stats_path=tmp_path / "g.pt").region_norm is None
