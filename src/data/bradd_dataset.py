"""BraDD-S1TS dataset with train-only Z-score stats, explicit subsampling and row indices."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.data.samples import days_since, load_sample, phase_table, read_meta, sample_origin, subset_rows
from src.data.stats import load_or_compute_stats
from src.data.transforms import TemporalSubsample, build_subsample, zscore

NORMALIZATIONS = ("zscore", "none")


class BraDDDataset(Dataset):
    """One phase (train/validation/test) of a BraDD-format dataset.

    Mirrors upstream `BraDDS1TSDataset`: day offsets are days since the earliest image/label date
    starting at 1, images are Z-scored per channel, and the temporal subsample is applied to the
    train and validation phases only (upstream skips `TemporalDropout` at test). Set
    `subsample_at_test=True` to also subsample the test phase (Phase 2 temporal-depth ablation).
    Differences from upstream: the stats come from the train split of `split_column` (upstream's
    bootstrap passes the wrong arguments, PRD 4.4 issue 1), and each item carries `Index`, its row
    in `self.meta`. `max_samples` keeps a seeded random subset of the phase, in meta order.
    """

    def __init__(self, root: str | Path, phase: str, split_column: str = "close_set",
                 stats_path: str | Path | None = None, normalization: str = "zscore",
                 temporal_subsample: TemporalSubsample | Mapping | None = None,
                 max_samples: int | None = None, seed: int = 42, subsample_at_test: bool = False) -> None:
        """Read the meta table, select the phase and prepare stats and transforms."""
        super().__init__()
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization '{normalization}', expected one of {NORMALIZATIONS}")
        self.root = Path(root)
        self.phase = phase
        self.meta = subset_rows(phase_table(read_meta(root), split_column, phase), max_samples, seed)
        self.stats = load_or_compute_stats(root, split_column, stats_path) if normalization == "zscore" else None
        use_subsample = phase != "test" or subsample_at_test
        self.subsample = build_subsample(temporal_subsample, seed) if use_subsample else None

    def __len__(self) -> int:
        """Number of samples in this phase."""
        return len(self.meta)

    def reseed(self, seed: int) -> None:
        """Reseed every random transform (used by the DataLoader worker init)."""
        if self.subsample is not None:
            self.subsample.reseed(seed)

    def load_raw(self, index: int) -> dict:
        """The raw sample dict stored on disk for row `index`."""
        return load_sample(self.root / "Samples" / self.meta.loc[index, "file"])

    def build_item(self, raw: dict, index: int) -> dict:
        """Convert a raw sample into tensors with day offsets on the upstream origin."""
        origin = sample_origin(raw)
        return {
            "Images": raw["image"].float(),
            "ImageDays": days_since(raw["image_dates"], origin),
            "TargetDays": days_since(raw["label_dates"], origin),
            "Targets": raw["label"].long(),
            "Index": torch.tensor(index, dtype=torch.long),
        }

    def apply_transforms(self, item: dict) -> dict:
        """Normalise, then subsample dates (when enabled for this phase)."""
        if self.stats is not None:
            item["Images"] = zscore(item["Images"], self.stats)
        if self.subsample is not None:
            item = self.subsample(item)
        return item

    def __getitem__(self, index: int) -> dict:
        """Item dict: Images [T,2,H,W], ImageDays [T], TargetDays [t], Targets [t,H,W], Index []."""
        return self.apply_transforms(self.build_item(self.load_raw(index), index))
