"""Write a SMOKE synthetic BraDD-format dataset: `python -m src.smoke_data --config configs/smoke/synthetic_bradd.yaml`.

The data come from `src.data.synthetic.make_bradd_fixture` (random speckle plus elliptical clearings). They are
not measurements; anything trained on them is SMOKE and reproduces nothing. A `SMOKE.txt` marker is
written next to `meta.csv`. `synthetic.dated: true` writes the Phase 3 dated format instead.
"""
import argparse
from pathlib import Path

from src.config import load_config, require
from src.training.run import SMOKE_LABEL
from src.data.synthetic import make_bradd_fixture


def write_smoke_dataset(root: str | Path, n_per_split: dict, seed: int, dated: bool) -> Path:
    """Generate the dataset under `root` (refuses a non-empty directory) and mark it SMOKE."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"{root} is not empty; delete it first to regenerate the SMOKE dataset")
    make_bradd_fixture(root, n_per_split=n_per_split, dated=dated, seed=seed)
    (root / "SMOKE.txt").write_text(
        f"{SMOKE_LABEL}: synthetic data from src/data/synthetic.py::make_bradd_fixture "
        f"(dated {dated}, seed {seed}, counts {n_per_split}). Not real Sentinel-1 data.\n"
    )
    return root


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    if not require(cfg, "smoke"):
        raise ValueError("synthetic data configs must set smoke: true")
    spec = require(cfg, "synthetic")
    root = write_smoke_dataset(spec["root"], dict(spec["n_per_split"]), spec["seed"], spec["dated"])
    print(f"[{SMOKE_LABEL}] wrote synthetic BraDD-format dataset {spec['n_per_split']} to {root}")


if __name__ == "__main__":
    main()
