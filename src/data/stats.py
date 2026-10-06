"""Per-channel normalisation statistics computed on the train split only (fixes PRD 4.4 issue 1)."""
from __future__ import annotations

from pathlib import Path

import torch

from src.data.samples import load_sample, phase_table, read_meta

STAT_KEYS = ("mean", "std", "min", "max")


def compute_train_stats(root: str | Path, split_column: str = "close_set", phase: str = "train") -> dict:
    """Per-channel mean, population std, min and max of the raw dB images of one phase (train)."""
    table = phase_table(read_meta(root), split_column, phase)
    if table.empty:
        raise ValueError(f"no '{phase}' rows in column '{split_column}' under {root}")
    count, mean, m2 = 0, None, None
    lo, hi = None, None
    for file in table["file"]:
        image = load_sample(Path(root) / "Samples" / file)["image"].double()
        values = image.transpose(0, 1).reshape(image.shape[1], -1)
        n_new, mean_new = values.shape[1], values.mean(dim=1)
        m2_new = ((values - mean_new[:, None]) ** 2).sum(dim=1)
        if mean is None:
            count, mean, m2 = n_new, mean_new, m2_new
            lo, hi = values.min(dim=1).values, values.max(dim=1).values
            continue
        delta, total = mean_new - mean, count + n_new
        mean = mean + delta * n_new / total
        m2 = m2 + m2_new + delta ** 2 * count * n_new / total
        count = total
        lo = torch.minimum(lo, values.min(dim=1).values)
        hi = torch.maximum(hi, values.max(dim=1).values)
    return {"mean": mean.float(), "std": torch.sqrt(m2 / count).float(), "min": lo.float(), "max": hi.float()}


def load_or_compute_stats(root: str | Path, split_column: str, stats_path: str | Path | None) -> dict:
    """Reuse cached stats at `stats_path` if present, otherwise compute them (and cache if a path is given)."""
    if stats_path is not None and Path(stats_path).exists():
        stats = torch.load(stats_path, map_location="cpu", weights_only=False)
        missing = [k for k in STAT_KEYS if k not in stats]
        if missing:
            raise KeyError(f"stats file {stats_path} lacks keys {missing}")
        return {k: torch.as_tensor(stats[k], dtype=torch.float32) for k in STAT_KEYS}
    stats = compute_train_stats(root, split_column)
    if stats_path is not None:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, stats_path)
    return stats
