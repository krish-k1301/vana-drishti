"""Prefix truncation for early-detection training and sliding-prefix evaluation (PRD Phase 4, 4.4 issue 5)."""
from __future__ import annotations

import torch

ANCHORS = ("deter", "burn")


def reference_days(sample: dict) -> torch.Tensor:
    """Per-pixel DETER reference day [H,W] (int64, -1 none): `RefDay` when the sample carries it.

    Fallback for samples without `RefDay`: `EventDay` inside `EventMask`; elsewhere the first stored label
    day at which the pixel turns positive (a later-or-equal bound at label resolution, so never anti-causal).
    """
    if "RefDay" in sample:
        return sample["RefDay"].long()
    targets, days = sample["Targets"].bool(), sample["TargetDays"].long()
    new = targets & ~targets[:1]
    onset = torch.where(new.any(0), days[new.long().argmax(0)], torch.full(new.shape[1:], -1, dtype=torch.long))
    event = torch.as_tensor(sample["EventDay"]).long()
    inside = sample["EventMask"].bool() & (event >= 0)
    return torch.where(inside, event.expand_as(onset), onset)


class PrefixTruncation:
    """Cut a dated sample at a cutoff day t_c and rebuild cumulative labels up to t_c.

    All images after t_c are dropped. Labels are rebuilt at days t_c, t_c - L, t_c - 2L, ... (>= 1),
    with L = `label_interval_days`, so the last label is exactly at t_c; day 1 is prepended when only
    one such day exists, so there are always >= 2 label dates. The mask at day d is
    `Targets[0] (pre-existing) OR (0 <= pixel event day <= d)`, with `Targets` the stored (full-window)
    labels. Pixel event day, anchor 'deter': the DETER reference day of every polygon in the patch
    (`reference_days`). Anchor 'burn': `BurnDay`, but only on pixels the stored labels or the reference
    days mark as cleared; burns elsewhere (pasture fires, negatives) and cleared pixels without a burn are
    never positive.
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
        reference = reference_days(sample)
        if self.anchor == "deter":
            return reference
        targets = sample["Targets"].bool()
        cleared = (targets[-1] & ~targets[0]) | (reference >= 0)
        return torch.where(cleared, sample["BurnDay"].long(), torch.full_like(reference, -1))

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
