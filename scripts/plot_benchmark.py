"""Bar chart of pixel IoU per run from a src.benchmark_table CSV, with the reference row as a line.

`python scripts/plot_benchmark.py --table results/benchmark.csv --out results/benchmark_iou.png`
The title carries the rows' label column (SMOKE for synthetic or shortened runs).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot(table: pd.DataFrame, out: Path) -> Path:
    """Draw pixel IoU bars (with OR rule) per experiment and a dashed line at the reference row."""
    fig, ax = plt.subplots(figsize=(max(4.0, 1.2 * len(table)), 4.0))
    ax.bar(table["experiment_name"], table["pixel_iou"], color="#4c72b0")
    reference = table[table["is_reference"].astype(str) == "True"]
    if len(reference):
        ref = reference.iloc[0]
        ax.axhline(ref["pixel_iou"], color="#c44e52", linestyle="--", label=f"reference: {ref['experiment_name']}")
        ax.legend()
    labels = " ".join(sorted({str(v) for v in table["label"] if str(v)}))
    mmu = "; ".join(sorted(set(table["mmu"].astype(str))))
    ax.set_title(f"{labels + ' ' if labels else ''}Pixel IoU per model ({mmu})", fontsize=9)
    ax.set_ylabel("pixel IoU (label[0] OR prediction)")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(pd.read_csv(args.table, keep_default_na=False), args.out)


if __name__ == "__main__":
    main()
