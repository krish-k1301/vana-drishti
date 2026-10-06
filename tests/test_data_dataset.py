"""BraDDDataset, train-only stats, collation and build_dataloaders on the synthetic fixture."""
from __future__ import annotations

import pytest
import torch

from src.data import BraDDDataset, build_dataloaders, collate_batch, compute_train_stats
from src.data.samples import load_sample, read_meta


def _raw_images(root, phase):
    meta = read_meta(root)
    files = meta.loc[meta["close_set"] == phase, "file"]
    return torch.cat([load_sample(root / "Samples" / f)["image"] for f in files]).double()


def test_stats_use_train_split_only(bradd_root):
    """Stats equal the train-split moments and differ from all-split moments."""
    stats = compute_train_stats(bradd_root, "close_set")
    train = _raw_images(bradd_root, "train").transpose(0, 1).reshape(2, -1)
    assert torch.allclose(stats["mean"].double(), train.mean(1), atol=1e-4)
    assert torch.allclose(stats["std"].double(), train.std(1, unbiased=False), atol=1e-4)
    assert torch.allclose(stats["min"].double(), train.min(1).values)
    everything = torch.cat([_raw_images(bradd_root, p) for p in ("train", "validation", "test")])
    assert not torch.allclose(stats["mean"].double(), everything.transpose(0, 1).reshape(2, -1).mean(1))


def test_stats_cached_and_reused(bradd_root, tmp_path):
    """Stats are written to stats_path and an existing file is reused as is."""
    path = tmp_path / "stats.pt"
    BraDDDataset(bradd_root, "validation", stats_path=path)
    assert path.exists()
    torch.save({"mean": torch.tensor([1.0, 2.0]), "std": torch.tensor([1.0, 1.0]),
                "min": torch.zeros(2), "max": torch.zeros(2)}, path)
    ds = BraDDDataset(bradd_root, "validation", stats_path=path)
    raw = ds.load_raw(0)["image"]
    assert torch.allclose(ds[0]["Images"][:, 1], raw[:, 1] - 2.0)


def test_item_matches_upstream_day_convention(bradd_root):
    """Day offsets start at 1 from the earliest image/label date, as upstream."""
    ds = BraDDDataset(bradd_root, "train", normalization="none")
    raw = ds.load_raw(3)
    item = ds[3]
    origin = min(raw["image_dates"] + raw["label_dates"])
    assert item["ImageDays"].tolist() == [(d - origin).days + 1 for d in raw["image_dates"]]
    assert item["TargetDays"].tolist() == [(d - origin).days + 1 for d in raw["label_dates"]]
    assert min(item["ImageDays"].min(), item["TargetDays"].min()) == 1
    assert torch.equal(item["Images"], raw["image"]) and item["Index"].item() == 3
    assert item["Targets"].dtype == torch.long


def test_zscore_output_and_bad_normalization(bradd_root):
    """Train items are ~zero-mean unit-std; unknown normalisation fails."""
    ds = BraDDDataset(bradd_root, "train")
    images = torch.cat([ds[i]["Images"] for i in range(len(ds))]).transpose(0, 1).reshape(2, -1)
    assert images.mean(1).abs().max() < 1e-3 and (images.std(1) - 1).abs().max() < 1e-3
    with pytest.raises(ValueError):
        BraDDDataset(bradd_root, "train", normalization="minmax")


def test_max_samples_is_seeded_subset(bradd_root):
    """max_samples gives a reproducible subset kept in meta order."""
    a = BraDDDataset(bradd_root, "train", normalization="none", max_samples=3, seed=5)
    b = BraDDDataset(bradd_root, "train", normalization="none", max_samples=3, seed=5)
    assert len(a) == 3 and a.meta["file"].tolist() == b.meta["file"].tolist()
    assert a.meta["file"].tolist() == sorted(a.meta["file"].tolist())


def test_subsample_skipped_at_test_unless_flagged(bradd_root):
    """Subsampling applies to validation, not test, unless subsample_at_test."""
    spec = {"mode": "uniform", "num_dates": 5}
    val = BraDDDataset(bradd_root, "validation", normalization="none", temporal_subsample=spec)
    test = BraDDDataset(bradd_root, "test", normalization="none", temporal_subsample=spec)
    flagged = BraDDDataset(bradd_root, "test", normalization="none", temporal_subsample=spec,
                           subsample_at_test=True)
    assert val[0]["Images"].shape[0] == 5 and flagged[0]["Images"].shape[0] == 5
    assert test[0]["Images"].shape[0] == len(test.load_raw(0)["image_dates"])


def test_collate_pads_end_and_masks(bradd_root):
    """Collation pads at the end with 0 and PadMask marks exactly the padding."""
    ds = BraDDDataset(bradd_root, "train", normalization="none")
    items = [ds[i] for i in range(4)]
    batch = collate_batch(items)
    lengths = [it["ImageDays"].shape[0] for it in items]
    assert batch["Images"].shape == (4, max(lengths), 2, 48, 48)
    for b, n in enumerate(lengths):
        assert not batch["PadMask"][b, :n].any() and batch["PadMask"][b, n:].all()
        assert (batch["ImageDays"][b, n:] == 0).all() and (batch["Images"][b, n:] == 0).all()
        assert torch.equal(batch["Images"][b, :n], items[b]["Images"])
    assert batch["Index"].tolist() == [0, 1, 2, 3] and batch["PadMask"].dtype == torch.bool


def test_build_dataloaders(bradd_root, tmp_path):
    """build_dataloaders returns three loaders with the configured sizes and seeded order."""
    cfg = {"root": str(bradd_root), "split_column": "close_set", "stats_path": str(tmp_path / "s.pt"),
           "normalization": "zscore", "batch_size": 3, "num_workers": 0, "seed": 42,
           "temporal_subsample": {"mode": "random", "num_dates": 6}, "max_samples": {"train": 6}}
    loaders = build_dataloaders(cfg)
    assert set(loaders) == {"train", "validation", "test"}
    assert len(loaders["train"].dataset) == 6 and len(loaders["test"].dataset) == 4
    batch = next(iter(loaders["train"]))
    assert batch["Images"].shape[:2] == (3, 6) and batch["Targets"].shape == (3, 2, 48, 48)
    first = [next(iter(build_dataloaders(cfg)["train"]))["Index"].tolist() for _ in range(2)]
    assert first[0] == first[1]
    test_batch = next(iter(loaders["test"]))
    assert {"Images", "ImageDays", "TargetDays", "Targets", "PadMask", "Index"} == set(test_batch)
