"""Prefix truncation for early-detection training and sliding-prefix evaluation (PRD Phase 4, 4.4 issue 5)."""
from __future__ import annotations

import torch

ANCHORS = ("deter", "burn")


class PrefixTruncation:
    """Cut a dated sample at a cutoff day t_c and rebuild cumulative labels up to t_c.

    All images after t_c are dropped. Labels are rebuilt at days t_c, t_c - L, t_c - 2L, ... (>= 1),
    with L = `label_interval_days`, so the last label is exactly at t_c; day 1 is prepended when only
    one such day exists, so there are always >= 2 label dates. The mask at day d is
    `Targets[0] (pre-existing) OR (0 <= pixel event day <= d)`. Pixel event day is `EventDay` inside
    `EventMask` for anchor 'deter', and `BurnDay` for anchor 'burn' (pixels without a burn are never
    positive under the burn anchor).
    """

    def __init__(self, anchor: str, min_dates: int, label_interval_days: int, seed: int = 42) -> None:
        """Validate arguments and seed the cutoff generator."""
        if anchor not in ANCHORS:
            raise ValueError(f"unknown anchor '{anchor}', expected one of {ANCHORS}")
        if min_dates < 1 or label_interval_days < 1:
            raise ValueError("min_dates and label_interval_days must be >= 1")
        self.anchor = anchor
        self.min_dates = min_dates
        self.label_interval_days = label_interval_days
        self._generator = torch.Generator().manual_seed(seed)

    def reseed(self, seed: int) -> None:
        """Reset the cutoff generator (called once per DataLoader worker)."""
        self._generator.manual_seed(seed)

    def pixel_event_days(self, sample: dict) -> torch.Tensor:
        """Per-pixel event day [H,W] (int64, -1 = never) for the configured anchor."""
        if self.anchor == "burn":
            return sample["BurnDay"].long()
        event = torch.as_tensor(sample["EventDay"]).long()
        inside = sample["EventMask"].bool() & (event >= 0)
        return torch.where(inside, event.expand_as(inside), torch.full(inside.shape, -1, dtype=torch.long))

    def label_days(self, cutoff: int) -> torch.Tensor:
        """Label days ending exactly at the cutoff, spaced by `label_interval_days`, all >= 1."""
        days = torch.arange(cutoff, 0, -self.label_interval_days).flip(0)
        if days.numel() < 2:
            days = torch.tensor([1, cutoff])
        return days.long()

    def draw_cutoff(self, image_days: torch.Tensor) -> int:
        """Random image day with at least `min_dates` images at or before it (last day if T is too short)."""
        candidates = image_days[image_days > 0][self.min_dates - 1:]
        if candidates.numel() == 0:
            return int(image_days.max())
        pick = torch.randint(candidates.numel(), (1,), generator=self._generator)
        return int(candidates[pick])

    def prefix_at(self, sample: dict, cutoff: int) -> dict:
        """Deterministic truncation of an (un-padded) sample at day `cutoff`; the input is not modified."""
        keep = sample["ImageDays"] <= cutoff
        if not bool(keep.any()):
            raise ValueError(f"cutoff day {cutoff} precedes the first image day")
        target_days = self.label_days(cutoff)
        event_days = self.pixel_event_days(sample)
        happened = (event_days[None] >= 0) & (event_days[None] <= target_days[:, None, None])
        out = dict(sample)
        out["Images"] = sample["Images"][keep]
        out["ImageDays"] = sample["ImageDays"][keep]
        out["TargetDays"] = target_days
        out["Targets"] = (sample["Targets"][0].bool()[None] | happened).long()
        return out

    def __call__(self, sample: dict) -> dict:
        """Truncate the sample at a randomly drawn cutoff."""
        return self.prefix_at(sample, self.draw_cutoff(sample["ImageDays"]))
