"""Build train/validation/test DataLoaders from a data config dict."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

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


def _prefix_for(cfg: Mapping, phase: str, seed: int) -> PrefixTruncation | None:
    """PrefixTruncation for this phase if the config enables it there."""
    spec = cfg.get("prefix_truncation")
    if not spec or phase not in spec.get("phases", ["train"]):
        return None
    return PrefixTruncation(spec["anchor"], int(spec["min_dates"]), int(spec["label_interval_days"]),
                            int(spec.get("seed", seed)))


def _max_samples(spec: Mapping | int | None, phase: str) -> int | None:
    """Per-phase sample cap from either a {phase: n} mapping or a single int."""
    return spec.get(phase) if isinstance(spec, Mapping) else spec


def build_dataset(data_cfg: Mapping, phase: str) -> Dataset:
    """One phase's dataset as described by the config (keys documented in configs/data/bradd.yaml)."""
    kind = data_cfg.get("dataset", "bradd")
    if kind not in DATASETS:
        raise ValueError(f"unknown dataset '{kind}', expected one of {DATASETS}")
    seed = int(data_cfg.get("seed", 42))
    subsample = dict(data_cfg.get("temporal_subsample") or {})
    kwargs = {
        "split_column": data_cfg["split_column"],
        "stats_path": data_cfg.get("stats_path"),
        "normalization": data_cfg.get("normalization", "zscore"),
        "temporal_subsample": subsample or None,
        "max_samples": _max_samples(data_cfg.get("max_samples"), phase),
        "seed": seed,
        "subsample_at_test": bool(subsample.get("at_test", False)),
    }
    if kind == "dated":
        return DatedDataset(data_cfg["root"], phase, prefix=_prefix_for(data_cfg, phase, seed), **kwargs)
    return BraDDDataset(data_cfg["root"], phase, **kwargs)


def build_dataloaders(data_cfg: Mapping) -> dict[str, DataLoader]:
    """DataLoaders keyed train/validation/test; only train is shuffled, with a seeded generator."""
    seed = int(data_cfg.get("seed", 42))
    loaders = {}
    for phase in PHASES:
        loaders[phase] = DataLoader(
            build_dataset(data_cfg, phase),
            batch_size=int(data_cfg["batch_size"]),
            shuffle=phase == "train",
            num_workers=int(data_cfg.get("num_workers", 0)),
            collate_fn=collate_batch,
            worker_init_fn=_reseed_worker,
            generator=torch.Generator().manual_seed(seed),
        )
    return loaders
