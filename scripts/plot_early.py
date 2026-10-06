"""Phase 5 plots from src.early_eval outputs: probability vs days since event per DETER stage, latency histograms.

`python scripts/plot_early.py --early-dir <run dir>/early_eval [--eval-config configs/eval/early.yaml] [--out DIR]`
Titles carry the summary's label (SMOKE for synthetic or shortened runs).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402

REFERENCES = ("deter", "burn", "radd")


def prefix_title(summary: dict, text: str) -> str:
    """Title with the run label (SMOKE) and the MMU in front of `text`."""
    label = f"{summary['label']} " if summary.get("label") else ""
    return f"{label}{text} ({summary['experiment_name']}, {summary['mmu']})"


def plot_curves(curves: pd.DataFrame, summary: dict, bin_width: float, out: Path) -> Path:
    """Event-mean probability vs days since the DETER date: thin line per event, binned median per stage."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for i, (stage, part) in enumerate(curves.groupby("stage", sort=True)):
        color = f"C{i}"
        for _, event in part.groupby("index"):
            ax.plot(event["days_since_event"], event["prob_mean"], color=color, alpha=0.25, linewidth=0.8)
        bins = np.floor(part["days_since_event"] / bin_width) * bin_width + bin_width / 2
        median = part.groupby(bins)["prob_mean"].median()
        ax.plot(median.index, median.values, color=color, linewidth=2.0, label=f"{stage} (n={part['index'].nunique()})")
    ax.axvline(0, color="black", linestyle=":", linewidth=1)
    ax.axhline(summary["tau"], color="grey", linestyle="--", linewidth=1, label=f"tau={summary['tau']:.3g}")
    ax.set(xlabel="days since DETER date (cutoff - event)", ylabel="mean change probability in event mask")
    ax.set_title(prefix_title(summary, "Probability vs days since event"), fontsize=9)
    ax.legend(fontsize=7)
    return save(fig, out / "early_probability_vs_days.png")


def plot_latency(events: pd.DataFrame, summary: dict, n_bins: int, out: Path) -> Path:
    """One latency histogram per reference; the title states the missed and no-reference counts."""
    fig, axes = plt.subplots(1, len(REFERENCES), figsize=(4.0 * len(REFERENCES), 3.5))
    for ax, ref in zip(axes, REFERENCES):
        has_ref = events[f"ref_{ref}"].notna()
        latency = events.loc[has_ref, f"latency_{ref}"].dropna()
        if len(latency):
            ax.hist(latency, bins=n_bins, color="#4c72b0")
        missed = int(has_ref.sum() - len(latency))
        ax.axvline(0, color="black", linestyle=":", linewidth=1)
        ax.set_title(f"vs {ref}: n={len(latency)}, missed={missed}, no ref={int((~has_ref).sum())}", fontsize=8)
        ax.set_xlabel("latency (days)")
    fig.suptitle(prefix_title(summary, "Detection latency"), fontsize=9)
    return save(fig, out / "early_latency_hist.png")


def save(fig: plt.Figure, path: Path) -> Path:
    """Save and close a figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--early-dir", type=Path, required=True)
    parser.add_argument("--eval-config", default="configs/eval/early.yaml")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    plots = load_config(args.eval_config)["plots"]
    summary = json.loads((args.early_dir / "summary.json").read_text())
    out = args.out or args.early_dir
    plot_curves(pd.read_csv(args.early_dir / "curves.csv"), summary, float(plots["days_bin_width"]), out)
    plot_latency(pd.read_csv(args.early_dir / "events.csv"), summary, int(plots["latency_hist_bins"]), out)


if __name__ == "__main__":
    main()
