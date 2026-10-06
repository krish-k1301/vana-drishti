"""BraDD-S1TS dataset with train-only Z-score stats, explicit subsampling and row indices."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.crop import build_crop
from src.data.region_norm import RegionNormalizer, build_region_normalizer
from src.data.samples import days_since, load_sample, phase_table, read_meta, sample_origin, subset_rows
from src.data.seasonal import build_seasonal_window
from src.data.stats import load_or_compute_stats
from src.data.transforms import TemporalSubsample, build_subsample, zscore

NORMALIZATIONS = ("zscore", "none")
LABEL_KEYS = {"prodes": "label", "hansen": "label_hansen"}


class BraDDDataset(Dataset):
    """One phase (train/validation/test) of a BraDD-format dataset.

    Mirrors upstream `BraDDS1TSDataset`: day offsets are days since the earliest image/label date
    starting at 1, images are Z-scored per channel, and the temporal subsample is applied to the
    train and validation phases only (upstream skips `TemporalDropout` at test). Set
    `subsample_at_test=True` to also subsample the test phase (Phase 2 temporal-depth ablation).
    Differences from upstream: the stats come from the train split of `split_column` (upstream's
    bootstrap passes the wrong arguments, PRD 4.4 issue 1), and each item carries `Index`, its row
    in `self.meta`. `max_samples` keeps a seeded random subset of the phase, in meta order.
    Optional switches, all off when their config block is None or `enabled: false`:
    `per_region_norm` (src.data.region_norm, replaces the global Z-score), `seasonal_window`
    (src.data.seasonal, applied to the raw images first, so the day origin is that of the kept images and
    the labels), `crop` (src.data.crop, after the temporal subsample; `tile` multiplies the length).
    `label_source` picks the label stack: 'prodes' -> sample key `label`, 'hansen' -> `label_hansen`.
    Order: seasonal window -> normalisation -> temporal subsample -> crop.
    """

    def __init__(self, root: str | Path, phase: str, split_column: str = "close_set",
                 stats_path: str | Path | None = None, normalization: str = "zscore",
                 temporal_subsample: TemporalSubsample | Mapping | None = None,
                 max_samples: int | None = None, seed: int = 42, subsample_at_test: bool = False,
                 per_region_norm: Mapping | None = None, seasonal_window: Mapping | None = None,
                 crop: Mapping | None = None, label_source: str = "prodes") -> None:
        """Read the meta table, select the phase and prepare stats and transforms."""
        super().__init__()
        if normalization not in NORMALIZATIONS:
            raise ValueError(f"unknown normalization '{normalization}', expected one of {NORMALIZATIONS}")
        if label_source not in LABEL_KEYS:
            raise ValueError(f"unknown label_source '{label_source}', expected one of {tuple(LABEL_KEYS)}")
        self.root, self.phase, self.label_key = Path(root), phase, LABEL_KEYS[label_source]
        self.seasonal = build_seasonal_window(seasonal_window)
        table = subset_rows(phase_table(read_meta(root), split_column, phase), max_samples, seed)
        self.meta, self.skipped = self._drop_out_of_season(table)
        self.stats, self.region_norm = self._normalisers(split_column, stats_path, normalization, per_region_norm)
        use_subsample = phase != "test" or subsample_at_test
        self.subsample = build_subsample(temporal_subsample, seed) if use_subsample else None
        self.crop = build_crop(crop, phase, seed)
        self.tiles_per_sample = self.crop.num_tiles if self.crop is not None else 1

    def _drop_out_of_season(self, table: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """With seasonal on_empty 'skip', drop rows without in-window images (reads each sample once)."""
        if self.seasonal is None or self.seasonal.on_empty != "skip":
            return table, []
        keep = [bool(self.seasonal.keep_indices(load_sample(self.root / "Samples" / f)["image_dates"]))
                for f in table["file"]]
        skipped = table.loc[[not k for k in keep], "file"].tolist()
        return table[keep].reset_index(drop=True), skipped

    def _normalisers(self, split_column: str, stats_path: str | Path | None, normalization: str,
                     region_spec: Mapping | None) -> tuple[dict | None, RegionNormalizer | None]:
        """Global train stats, or a per-region normaliser when `per_region_norm.enabled` is true."""
        regional = bool(region_spec and region_spec.get("enabled", False))
        if normalization == "none":
            if regional:
                raise ValueError("per_region_norm needs normalization 'zscore'")
            return None, None
        if regional:
            regions = self.meta[region_spec["region_column"]] if len(self.meta) else []
            return None, build_region_normalizer(self.root, split_column, regions, region_spec)
        return load_or_compute_stats(self.root, split_column, stats_path), None

    def __len__(self) -> int:
        """Number of items in this phase (samples x tiles per sample)."""
        return len(self.meta) * self.tiles_per_sample

    def reseed(self, seed: int) -> None:
        """Reseed every random transform (used by the DataLoader worker init)."""
        if self.subsample is not None:
            self.subsample.reseed(seed)
        if self.crop is not None:
            self.crop.reseed(seed + 2)

    def load_raw(self, index: int) -> dict:
        """The raw sample dict stored on disk for meta row `index` (seasonal window applied if enabled)."""
        file = self.meta.loc[index, "file"]
        raw = load_sample(self.root / "Samples" / file)
        return self.seasonal.apply(raw, file) if self.seasonal is not None else raw

    def build_item(self, raw: dict, index: int) -> dict:
        """Convert a raw sample into tensors with day offsets on the upstream origin."""
        if self.label_key not in raw:
            raise KeyError(f"sample {self.meta.loc[index, 'file']} has no '{self.label_key}' label stack")
        origin = sample_origin(raw)
        return {
            "Images": raw["image"].float(),
            "ImageDays": days_since(raw["image_dates"], origin),
            "TargetDays": days_since(raw["label_dates"], origin),
            "Targets": raw[self.label_key].long(),
            "Index": torch.tensor(index, dtype=torch.long),
        }

    def apply_transforms(self, item: dict, tile: int = 0) -> dict:
        """Normalise, subsample dates (when enabled for this phase), then crop."""
        if self.region_norm is not None:
            region = self.meta.loc[int(item["Index"]), self.region_norm.region_column]
            item["Images"] = self.region_norm(item["Images"], region)
        elif self.stats is not None:
            item["Images"] = zscore(item["Images"], self.stats)
        if self.subsample is not None:
            item = self.subsample(item)
        return self.crop(item, tile) if self.crop is not None else item

    def __getitem__(self, index: int) -> dict:
        """Item dict: Images [T,2,H,W], ImageDays [T], TargetDays [t], Targets [t,H,W], Index [] (+CropOffset)."""
        row, tile = divmod(index, self.tiles_per_sample)
        return self.apply_transforms(self.build_item(self.load_raw(row), row), tile)
