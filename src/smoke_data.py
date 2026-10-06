"""Write a SMOKE synthetic BraDD-format dataset: `python -m src.smoke_data --config configs/smoke/synthetic_bradd.yaml`.

The data come from `src.data.synthetic.make_bradd_fixture` (random speckle plus elliptical clearings). They are
not measurements; anything trained on them is SMOKE and reproduces nothing. A `SMOKE.txt` marker is
written next to `meta.csv`. `synthetic.dated: true` writes the Phase 3 dated format instead.
A config with a `target_biome` section instead of `synthetic` writes a Phase 6 target-biome copy of an existing
SMOKE dated set (`configs/smoke/synthetic_target_biome.yaml`, see `make_target_biome`).
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, require
from src.data.samples import EPOCH, load_sample, read_meta
from src.training.run import SMOKE_LABEL
from src.data.synthetic import make_bradd_fixture

BRAZIL_ONLY_KEYS = ("event_date", "burn_month", "prodes_year")  # DETER date, Fogo burn month, PRODES year
MODES = ("synthetic", "target_biome")


def _require_empty(root: Path) -> None:
    """Refuse to write into a non-empty directory."""
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; delete it first to regenerate the SMOKE dataset")


def write_smoke_dataset(root: str | Path, n_per_split: dict, seed: int, dated: bool) -> Path:
    """Generate the dataset under `root` (refuses a non-empty directory) and mark it SMOKE."""
    root = Path(root)
    _require_empty(root)
    make_bradd_fixture(root, n_per_split=n_per_split, dated=dated, seed=seed)
    (root / "SMOKE.txt").write_text(
        f"{SMOKE_LABEL}: synthetic data from src/data/synthetic.py::make_bradd_fixture "
        f"(dated {dated}, seed {seed}, counts {n_per_split}). Not real Sentinel-1 data.\n"
    )
    return root


def first_positive_days(label: torch.Tensor, label_dates: list) -> torch.Tensor:
    """int32 [H,W] days since 1970-01-01 of the first label date at which each pixel is positive, -1 never."""
    days = torch.tensor([(d - EPOCH).days for d in label_dates], dtype=torch.int32)
    positive = label.bool()
    first = days[positive.long().argmax(0)]
    return torch.where(positive.any(0), first, torch.full_like(first, -1))


def target_biome_sample(sample: dict, positive_class: str) -> dict:
    """A dated SMOKE sample as the Phase 6 GEE export stores it: Hansen labels, no Brazil-only keys.

    `label` = `label_hansen` (kept, identical, as in gee/convert.py for Hansen-sourced configs); `ref_day` = first
    Hansen-positive label date (so the cumulative labels rebuild exactly from it); `deter_class` = the configured
    positive class for positives, '' otherwise. DETER date, burn month and PRODES year are dropped; RADD dates,
    event mask and the rest are kept unchanged.
    """
    out = {k: v for k, v in sample.items() if k not in BRAZIL_ONLY_KEYS}
    out["label"] = sample["label_hansen"].clone()
    out["ref_day"] = first_positive_days(out["label"], sample["label_dates"])
    out["deter_class"] = positive_class if sample["deter_class"] else ""
    return out


def make_target_biome(source_root: str | Path, target_root: str | Path, positive_class: str) -> Path:
    """Write a target-biome copy (meta.csv + Samples, no stats file) of a SMOKE dated set and mark it SMOKE.

    The source must carry a SMOKE marker; the target must be empty. meta.csv keeps every row and column, with
    `deter_class` = `positive_class` for positives and `event_date` emptied (no DETER date outside Brazil).
    """
    source, target = Path(source_root), Path(target_root)
    if not (source / "SMOKE.txt").is_file():
        raise ValueError(f"{source} has no SMOKE.txt; target-biome copies are only made from SMOKE data")
    _require_empty(target)
    (target / "Samples").mkdir(parents=True)
    meta = read_meta(source)
    for name in meta["file"]:
        sample = load_sample(source / "Samples" / name)
        torch.save(target_biome_sample(sample, positive_class), target / "Samples" / name)
    meta["deter_class"] = np.where(meta["deter_class"].astype(str) != "", positive_class, "")
    meta["event_date"] = ""
    meta.to_csv(target / "meta.csv")
    (target / "SMOKE.txt").write_text(
        f"{SMOKE_LABEL}: Phase 6 target-biome copy of {source} written by src/smoke_data.py::make_target_biome "
        f"(Hansen labels, keys {list(BRAZIL_ONLY_KEYS)} dropped, deter_class '{positive_class}'). "
        "Not real Sentinel-1 data.\n"
    )
    return target


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if not require(cfg, "smoke"):
        raise ValueError("synthetic data configs must set smoke: true")
    modes = [m for m in MODES if m in cfg]
    if len(modes) != 1:
        raise ValueError(f"a SMOKE data config needs exactly one of {MODES}, found {modes}")
    if modes[0] == "target_biome":
        spec = require(cfg, "target_biome")
        root = make_target_biome(spec["source_root"], spec["root"], spec["positive_class_label"])
        print(f"[{SMOKE_LABEL}] wrote target-biome copy of {spec['source_root']} to {root}")
        return
    spec = require(cfg, "synthetic")
    root = write_smoke_dataset(spec["root"], dict(spec["n_per_split"]), spec["seed"], spec["dated"])
    print(f"[{SMOKE_LABEL}] wrote synthetic BraDD-format dataset {spec['n_per_split']} to {root}")


if __name__ == "__main__":
    main()
