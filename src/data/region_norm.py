"""Per-region Z-score normalisation of dB values (PRD 4.3 `per_region_norm`, SN-TUNet; PRD Phase 6 mitigation).

A region is the value of a meta.csv column named in the config: `state` for BraDD (Brazilian state) and for
the GEE sets (DETER `uf` for Amazon, `congo`/`borneo` for the pilots), or `region_block` (0.5 deg blocks).
Stats per region come from the train split (cached). A region missing from the train cache gets, per
`unseen_region`: `global` (all-train stats), `self` (stats of that region's own images in the dataset being
loaded, every split, labels unused) or `error`. Regions listed in `self_normalise_regions` always use their
own images, e.g. a cross-biome target region evaluated with Amazon weights unchanged.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import pandas as pd
import torch

from src.data.samples import phase_table, read_meta
from src.data.stats import as_stats, stats_over_files
from src.data.transforms import zscore

SPEC_KEYS = ("enabled", "region_column", "stats_path", "unseen_region", "self_normalise_regions")
UNSEEN_POLICIES = ("global", "self", "error")


def compute_region_stats(root: str | Path, split_column: str, region_column: str) -> dict:
    """Global and per-region train-split stats: {'global': stats, 'regions': {name: stats}, 'region_column'}."""
    table = phase_table(read_meta(root), split_column, "train")
    if table.empty:
        raise ValueError(f"no train rows in '{split_column}' under {root}; point stats_path at the "
                         "training set's region stats cache")
    regions = {str(name): stats_over_files(root, group["file"]) for name, group in table.groupby(region_column)}
    return {"global": stats_over_files(root, table["file"]), "regions": regions, "region_column": region_column}


def load_or_compute_region_stats(root: str | Path, split_column: str, region_column: str,
                                 stats_path: str | Path | None) -> dict:
    """Reuse the cache at `stats_path` (checking its region column) or compute and cache it."""
    if stats_path is not None and Path(stats_path).exists():
        cached = torch.load(stats_path, map_location="cpu", weights_only=False)
        if cached.get("region_column") != region_column:
            raise ValueError(f"{stats_path} holds stats for region column '{cached.get('region_column')}', "
                             f"not '{region_column}'")
        return {"global": as_stats(cached["global"], stats_path), "region_column": region_column,
                "regions": {k: as_stats(v, stats_path) for k, v in cached["regions"].items()}}
    stats = compute_region_stats(root, split_column, region_column)
    if stats_path is not None:
        Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(stats, stats_path)
    return stats


class RegionNormalizer:
    """Z-score images with the stats of their region."""

    def __init__(self, region_column: str, table: Mapping[str, dict], sources: Mapping[str, str]) -> None:
        """Store region -> stats and region -> stats source ('train', 'global' or 'self')."""
        self.region_column = region_column
        self.table = dict(table)
        self.sources = dict(sources)

    def __call__(self, images: torch.Tensor, region: object) -> torch.Tensor:
        """Normalise [T,C,H,W] images of one sample from the given region."""
        if str(region) not in self.table:
            raise KeyError(f"no normalisation stats for region '{region}'")
        return zscore(images, self.table[str(region)])


def _self_stats(root: Path, meta: pd.DataFrame, region_column: str, region: str) -> dict:
    """Stats of every image of one region in this dataset (all splits; labels unused)."""
    return stats_over_files(root, meta.loc[meta[region_column].astype(str) == region, "file"])


def build_region_normalizer(root: str | Path, split_column: str, regions: Iterable[object],
                            spec: Mapping) -> RegionNormalizer:
    """RegionNormalizer covering `regions`, following the explicit policy in `spec` (see module docstring)."""
    missing = [k for k in SPEC_KEYS if k not in spec]
    if missing:
        raise KeyError(f"per_region_norm needs keys {missing}")
    if spec["unseen_region"] not in UNSEEN_POLICIES:
        raise ValueError(f"unseen_region must be one of {UNSEEN_POLICIES}, got '{spec['unseen_region']}'")
    column = spec["region_column"]
    cached = load_or_compute_region_stats(root, split_column, column, spec["stats_path"])
    forced = {str(r) for r in spec["self_normalise_regions"] or []}
    meta = read_meta(root)
    table, sources = {}, {}
    for region in sorted({str(r) for r in regions}):
        if region not in forced and region in cached["regions"]:
            table[region], sources[region] = cached["regions"][region], "train"
        elif region not in forced and spec["unseen_region"] == "global":
            table[region], sources[region] = cached["global"], "global"
        elif region in forced or spec["unseen_region"] == "self":
            table[region], sources[region] = _self_stats(Path(root), meta, column, region), "self"
        else:
            raise ValueError(f"region '{region}' has no train stats and unseen_region is 'error'")
    return RegionNormalizer(column, table, sources)
