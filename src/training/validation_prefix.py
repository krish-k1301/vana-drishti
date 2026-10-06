"""Deterministic prefix truncation for the validation phase of early-detection training (Phase 4).

The data loaders apply `PrefixTruncation` with a random cutoff per call, so a validation sample would get a
different cutoff every epoch and the monitored IoU would be noisy. Here each validation sample gets one fixed
cutoff, drawn with the same rule (`PrefixTruncation.draw_cutoff`, >= min_dates images) from a generator seeded
by `seed + row index`, so the monitored `validation/pixel_iou` measures the prefix task with the training
anchor, identically every epoch and for any number of workers.
"""
from torch.utils.data import DataLoader, Dataset

from src.data.collate import collate_batch
from src.data.loaders import build_dataset
from src.data.prefix import PrefixTruncation

VALIDATION_PREFIX_MODES = ("none", "fixed_per_sample")


class FixedCutoffPrefix(Dataset):
    """Wrap an un-truncated dataset and truncate sample i at a cutoff fixed by `seed + i`."""

    def __init__(self, dataset: Dataset, spec: dict, seed: int) -> None:
        """`spec` is the data config's `prefix_truncation` block (anchor, min_dates, label_interval_days)."""
        self.dataset = dataset
        self.spec = spec
        self.seed = seed

    def __len__(self) -> int:
        """Number of wrapped samples."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        """Sample `index` truncated at its fixed cutoff."""
        prefix = PrefixTruncation(self.spec["anchor"], int(self.spec["min_dates"]),
                                  int(self.spec["label_interval_days"]), seed=self.seed + index)
        return prefix(self.dataset[index])


def validation_loader(data_cfg: dict, mode: str) -> DataLoader | None:
    """Validation DataLoader for `mode` ('fixed_per_sample'), or None for 'none' (keep the standard loader)."""
    if mode not in VALIDATION_PREFIX_MODES:
        raise ValueError(f"train.validation_prefix must be one of {VALIDATION_PREFIX_MODES}, got {mode!r}")
    if mode == "none":
        return None
    spec = data_cfg.get("prefix_truncation")
    if not spec:
        raise ValueError("train.validation_prefix=fixed_per_sample needs data.prefix_truncation")
    if "validation" in spec["phases"]:
        raise ValueError("remove 'validation' from data.prefix_truncation.phases (random cutoffs) to use fixed ones")
    dataset = FixedCutoffPrefix(build_dataset(data_cfg, "validation"), spec, int(data_cfg["seed"]))
    return DataLoader(dataset, batch_size=int(data_cfg["batch_size"]), shuffle=False,
                      num_workers=int(data_cfg["num_workers"]), collate_fn=collate_batch)
