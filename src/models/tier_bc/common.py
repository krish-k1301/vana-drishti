"""Helpers shared by the Tier B/C wrappers: padding masks from day offsets, masked time means, arg checks."""
from collections.abc import Iterable

import torch

from src.models.segment_network import PAD_VALUE


def padded_dates(days: torch.Tensor) -> torch.Tensor:
    """Return the [B,T] bool mask of padded dates (day == PAD_VALUE; real days start at 1).

    Padding is read from the days, never from the frame content, so a padded frame's values cannot leak in.
    """
    return days == PAD_VALUE


def masked_time_mean(x: torch.Tensor, padded: torch.Tensor, time_dim: int) -> torch.Tensor:
    """Mean of `x` over `time_dim`, ignoring entries where `padded` [B,T] is True (B is dim 0 of `x`)."""
    shape = [1] * x.dim()
    shape[0], shape[time_dim] = padded.shape
    keep = (~padded).to(x.dtype).reshape(shape)
    count = keep.sum(dim=time_dim).clamp(min=1.0)
    return (x * keep).sum(dim=time_dim) / count


def require_keys(section: str, given: dict, expected: Iterable[str]) -> None:
    """Raise ValueError unless `given` has exactly the keys in `expected`."""
    expected = set(expected)
    missing, unknown = sorted(expected - set(given)), sorted(set(given) - expected)
    if missing or unknown:
        raise ValueError(f"{section}: missing keys {missing}, unknown keys {unknown}")


def require_day_range(days: torch.Tensor, max_day: int, model: str) -> None:
    """Raise ValueError if a real day offset exceeds the model's positional table size `max_day`."""
    largest = int(days.max()) if days.numel() else 0
    if largest > max_day:
        raise ValueError(f"{model}: day offset {largest} exceeds the positional table size {max_day}")
