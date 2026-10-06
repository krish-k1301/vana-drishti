"""Dated early-detection dataset (Phase 3 format) with event/burn/RADD days (Phase 4)."""
from __future__ import annotations

from pathlib import Path

import torch

from src.data.bradd_dataset import BraDDDataset
from src.data.prefix import PrefixTruncation, reference_days
from src.data.samples import epoch_days_to_offsets, sample_origin


class DatedDataset(BraDDDataset):
    """BraDDDataset over the dated meta (split column `dated_set`) with extra event keys.

    Extra item keys, all in days on the same origin as `ImageDays` (-1 = none):
    `EventDay` [] (DETER date of the central polygon), `RefDay` [H,W] (DETER date of every label-class polygon,
    from the sample key `ref_day`; derived from the stored labels when that key is absent, see
    src.data.prefix.reference_days), `BurnDay` [H,W] (first burn month, encoded as its last day),
    `RaddDay` [H,W], plus `EventMask` uint8 [H,W].
    If `prefix` is given, it is applied after normalisation (and after any temporal subsample);
    without it the stored labels are returned unchanged. The prefix rebuilds labels from DETER/burn
    event days, so it cannot be combined with `label_source='hansen'` (that would mix label sources).
    """

    def __init__(self, root: str | Path, phase: str, split_column: str = "dated_set",
                 prefix: PrefixTruncation | None = None, **kwargs) -> None:
        """Same arguments as BraDDDataset plus an optional PrefixTruncation."""
        super().__init__(root, phase, split_column=split_column, **kwargs)
        if prefix is not None and self.label_key != "label":
            raise ValueError("prefix truncation rebuilds DETER/burn labels; use label_source 'prodes' with it")
        self.prefix = prefix

    def reseed(self, seed: int) -> None:
        """Reseed the base random transforms and the prefix cutoff generator."""
        super().reseed(seed)
        if self.prefix is not None:
            self.prefix.reseed(seed + 1)

    def build_item(self, raw: dict, index: int) -> dict:
        """Base item plus EventDay, RefDay, BurnDay, RaddDay and EventMask."""
        item = super().build_item(raw, index)
        origin = sample_origin(raw)
        shape = raw["label"].shape[-2:]
        none = torch.full(shape, -1, dtype=torch.long)
        event = raw.get("event_date")
        event_day = (event - origin).days + 1 if event else -1
        item["EventDay"] = torch.tensor(event_day, dtype=torch.long)
        for key, raw_key in (("BurnDay", "burn_month"), ("RaddDay", "radd_date")):
            value = raw.get(raw_key)
            item[key] = none.clone() if value is None else epoch_days_to_offsets(torch.as_tensor(value), origin)
        mask = raw.get("event_mask")
        mask = torch.zeros(shape) if mask is None else torch.as_tensor(mask)
        item["EventMask"] = mask.to(torch.uint8)
        ref = raw.get("ref_day")
        item["RefDay"] = reference_days(item) if ref is None else epoch_days_to_offsets(torch.as_tensor(ref), origin)
        return item

    def apply_transforms(self, item: dict, tile: int = 0) -> dict:
        """Base transforms (including any crop), then the prefix truncation if configured."""
        item = super().apply_transforms(item, tile)
        return self.prefix(item) if self.prefix is not None else item
