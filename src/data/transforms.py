"""Per-sample transforms: Z-score normalisation and temporal subsampling (fixes PRD 4.4 issue 2)."""
from __future__ import annotations

from collections.abc import Mapping

import torch

PER_DATE_KEYS = ("Images", "ImageDays")
SUBSAMPLE_MODES = ("none", "uniform", "random", "last_only")


def zscore(images: torch.Tensor, stats: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Normalise [T,C,H,W] images per channel with the given mean/std."""
    mean = torch.as_tensor(stats["mean"], dtype=images.dtype).view(1, -1, 1, 1)
    std = torch.as_tensor(stats["std"], dtype=images.dtype).view(1, -1, 1, 1)
    return (images - mean) / std


class TemporalSubsample:
    """Keep a subset of the acquisition dates of one sample.

    Modes (explicit names replace upstream's inverted `is_random` flag):
    - `none`: keep every date.
    - `uniform`: first + last + evenly spaced interior dates, using upstream's exact formula
      `torch.linspace(0, T, n).int()[1:-1]`; identical to upstream `TemporalDropout(is_random=True)`.
    - `random`: first + last + `n-2` random interior dates (upstream `is_random=False`), seeded.
    - `last_only`: only the last date (the paper's single-date case; `num_dates` is ignored).
    If `num_dates >= T` every date is kept, which is also what the upstream formula yields.
    """

    def __init__(self, mode: str = "none", num_dates: int = -1, seed: int = 42) -> None:
        """Validate the mode and set up a seeded generator for `random`."""
        if mode not in SUBSAMPLE_MODES:
            raise ValueError(f"unknown temporal subsample mode '{mode}', expected one of {SUBSAMPLE_MODES}")
        if mode in ("uniform", "random") and num_dates < 2:
            raise ValueError(f"mode '{mode}' keeps first and last date, so num_dates must be >= 2 "
                             f"(got {num_dates}); use mode 'last_only' for a single date")
        self.mode = mode
        self.num_dates = num_dates
        self._generator = torch.Generator().manual_seed(seed)

    def reseed(self, seed: int) -> None:
        """Reset the random generator (called once per DataLoader worker)."""
        self._generator.manual_seed(seed)

    def indices(self, total: int) -> list[int]:
        """Indices of the dates to keep out of `total` dates, in increasing order."""
        if self.mode == "none" or (self.mode != "last_only" and self.num_dates >= total):
            return list(range(total))
        if self.mode == "last_only":
            return [total - 1]
        if self.mode == "uniform":
            interior = torch.linspace(0, total, self.num_dates).int()[1:-1].tolist()
        else:
            interior = (torch.randperm(total - 2, generator=self._generator)[: self.num_dates - 2] + 1).tolist()
            interior.sort()
        return [0, *interior, total - 1]

    def __call__(self, sample: dict) -> dict:
        """Return the sample with only the kept dates in its per-date tensors."""
        keep = self.indices(int(sample["ImageDays"].shape[0]))
        out = dict(sample)
        for key in PER_DATE_KEYS:
            out[key] = sample[key][keep]
        return out


def build_subsample(spec: TemporalSubsample | Mapping | None, seed: int) -> TemporalSubsample | None:
    """Turn a config mapping {mode, num_dates} (or an instance, or None) into a transform."""
    if spec is None or isinstance(spec, TemporalSubsample):
        return spec
    mode = spec.get("mode", "none")
    if mode == "none":
        return None
    return TemporalSubsample(mode, int(spec.get("num_dates", -1)), int(spec.get("seed", seed)))
