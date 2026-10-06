"""Build train/validation/test DataLoaders from a data config dict (every key explicit, see configs/data/*.yaml)."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.config import require
from src.data.bradd_dataset import BraDDDataset
from src.data.collate import collate_batch
from src.data.dated_dataset import DatedDataset
from src.data.prefix import PrefixTruncation

PHASES = ("train", "validation", "test")
DATASETS = ("bradd", "dated")


def _reseed_worker(worker_id: int) -> None:
    """Give each worker's random transforms a distinct seed derived from the loader seed."""
    info = get_worker_info()
    if info is not None and hasattr(info.dataset, "reseed"):
        info.dataset.reseed(int(info.seed % (2 ** 31)))


def _prefix_for(cfg: dict, phase: str, seed: int) -> PrefixTruncation | None:
    """PrefixTruncation for this phase if `prefix_truncation` (null = off) lists the phase."""
    spec = require(cfg, "prefix_truncation")
    if not spec or phase not in require(cfg, "prefix_truncation.phases"):
        return None
    return PrefixTruncation(spec["anchor"], int(require(cfg, "prefix_truncation.min_dates")),
                            int(require(cfg, "prefix_truncation.label_interval_days")), seed)


def _per_phase(spec: object, phase: str) -> object:
    """Value for one phase from either a {phase: value} mapping or a single value for every phase."""
    return spec[phase] if isinstance(spec, Mapping) else spec


def build_dataset(data_cfg: Mapping, phase: str) -> Dataset:
    """One phase's dataset as described by the config; a missing key raises KeyError.

    Feature blocks per_region_norm, seasonal_window and crop must be present (`enabled: false` turns them
    off); label_source and max_samples may be one value or a {phase: value} mapping (Phase 6: train on
    hansen, test on prodes). `prefix_truncation` is required for the dated dataset (null = off).
    """
    cfg = dict(data_cfg)
    kind = require(cfg, "dataset")
    if kind not in DATASETS:
        raise ValueError(f"unknown dataset '{kind}', expected one of {DATASETS}")
    seed = int(require(cfg, "seed"))
    subsample = dict(require(cfg, "temporal_subsample"))
    kwargs = {
        "split_column": require(cfg, "split_column"),
        "stats_path": require(cfg, "stats_path"),
        "normalization": require(cfg, "normalization"),
        "temporal_subsample": subsample,
        "max_samples": _per_phase(require(cfg, "max_samples"), phase),
        "seed": seed,
        "subsample_at_test": bool(require(cfg, "temporal_subsample.at_test")),
        "per_region_norm": require(cfg, "per_region_norm"),
        "seasonal_window": require(cfg, "seasonal_window"),
        "crop": require(cfg, "crop"),
        "label_source": _per_phase(require(cfg, "label_source"), phase),
    }
    if kind == "dated":
        return DatedDataset(require(cfg, "root"), phase, prefix=_prefix_for(cfg, phase, seed), **kwargs)
    return BraDDDataset(require(cfg, "root"), phase, **kwargs)


def build_dataloaders(data_cfg: Mapping) -> dict[str, DataLoader]:
    """DataLoaders keyed train/validation/test; only train is shuffled, with a seeded generator."""
    cfg = dict(data_cfg)
    seed = int(require(cfg, "seed"))
    loaders = {}
    for phase in PHASES:
        loaders[phase] = DataLoader(
            build_dataset(cfg, phase),
            batch_size=int(require(cfg, "batch_size")),
            shuffle=phase == "train",
            num_workers=int(require(cfg, "num_workers")),
            collate_fn=collate_batch,
            worker_init_fn=_reseed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders
