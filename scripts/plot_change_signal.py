"""Dataset-level explainer: the radar signal of new clearing and how the model's probability separates it.

`python scripts/plot_change_signal.py --config <run>/config_resolved.yaml --checkpoint <best.ckpt>
 [--split test] [--n 300] [--seed 0] [--out FILE] [dotted.key=value ...]`

For `--n` seeded samples of the split, every pixel not already deforested at the interval start (prior label 0)
of the interval with the most true change contributes its backscatter change (mean dB of the frames after the
interval midpoint minus mean dB of the frames before it, VV and VH), its model probability and its PRODES truth.
Panels: dB-change histograms per class, probability histograms per class, and the mean probability over the
(dVV, dVH) plane next to the true clearing rate over the same plane. Titles carry SMOKE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.loaders import build_dataset  # noqa: E402
from src.evaluation.common import load_model, run_dir, smoke_header  # noqa: E402
from src.evaluation.explain import busiest_interval, change_features, explain_interval  # noqa: E402
from src.train import data_config  # noqa: E402

CLASSES = (("stable (no new clearing)", 0, "#1f77b4"), ("new clearing (PRODES)", 1, "#d62728"))
DB_RANGE = (-8.0, 8.0)


def collect(dataset: object, model: object, rows: list[int]) -> dict[str, np.ndarray]:
    """Pixel arrays dvv, dvh, prob, true over the chosen rows (pixels with prior label 0 only)."""
    parts: dict[str, list[np.ndarray]] = {"dvv": [], "dvh": [], "prob": [], "true": []}
    for row in rows:
        raw = dataset.load_raw(row)
        item = dataset.apply_transforms(dataset.build_item(raw, row))
        result = explain_interval(model, item, busiest_interval(model, item))
        keep = result["prior"] == 0
        delta = change_features(raw, result, raw["label_dates"])
        parts["dvv"].append(delta["vv"][keep])
        parts["dvh"].append(delta["vh"][keep])
        parts["prob"].append(result["prob"][keep])
        parts["true"].append(result["true"][keep])
    return {key: np.concatenate(values) for key, values in parts.items()}


def class_histograms(ax: plt.Axes, values: np.ndarray, true: np.ndarray, bins: np.ndarray, xlabel: str) -> None:
    """Density histogram of `values` per truth class, with each class median marked."""
    for name, cls, color in CLASSES:
        part = values[true == cls]
        ax.hist(part, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color=color,
                label=f"{name}: n={len(part):,}")
        ax.axvline(np.median(part), color=color, linestyle="--", linewidth=1)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    ax.legend(fontsize=6.5)
    ax.tick_params(labelsize=7)


def plane(ax: plt.Axes, pixels: dict, values: np.ndarray, title: str, cbar_label: str) -> None:
    """Mean of `values` over a (dVV, dVH) grid; cells with fewer than 20 pixels are blank."""
    edges = np.linspace(*DB_RANGE, 33)
    total, _, _ = np.histogram2d(pixels["dvv"], pixels["dvh"], bins=[edges, edges])
    summed, _, _ = np.histogram2d(pixels["dvv"], pixels["dvh"], bins=[edges, edges], weights=values)
    mean = np.where(total >= 20, summed / np.maximum(total, 1), np.nan)
    image = ax.imshow(mean.T, origin="lower", extent=(*DB_RANGE, *DB_RANGE), cmap="magma", vmin=0, vmax=1,
                      aspect="auto")
    ax.axhline(0, color="white", linewidth=0.5)
    ax.axvline(0, color="white", linewidth=0.5)
    ax.set_xlabel("VV change (dB, after - before)", fontsize=8)
    ax.set_ylabel("VH change (dB, after - before)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.02).set_label(cbar_label, fontsize=7)


def auc(score: np.ndarray, true: np.ndarray) -> float:
    """ROC AUC of `score` for truth `true` (rank formula, ties averaged): 0.5 = chance, 1 = perfect."""
    ranks = rankdata(score)
    positives = true == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / max(n_pos * n_neg, 1))


def summary_text(pixels: dict, n_samples: int) -> str:
    """Counts and scores at P >= 0.5, the AUC of each single signal against the model's, and the takeaway."""
    pred, true = pixels["prob"] >= 0.5, pixels["true"]
    tp, fp, fn = int((pred & (true == 1)).sum()), int((pred & (true == 0)).sum()), int((~pred & (true == 1)).sum())
    scores = {"VH drop (-dVH)": -pixels["dvh"], "VV drop (-dVV)": -pixels["dvv"], "VH rise (dVH)": pixels["dvh"],
              "model P": pixels["prob"]}
    lines = "\n".join(f"  {name:<16} {auc(score, true):.3f}" for name, score in scores.items())
    return (f"{n_samples} test samples, {len(pred):,} pixels not deforested at interval start\n\n"
            f"Model at P >= 0.5: hits {tp:,}, false alarms {fp:,}, missed {fn:,}\n"
            f"  IoU {tp / max(tp + fp + fn, 1):.3f}, precision {tp / max(tp + fp, 1):.3f}, "
            f"recall {tp / max(tp + fn, 1):.3f}\n\n"
            f"How well each number alone separates the classes\n(AUC: 0.5 = chance, 1 = perfect):\n{lines}\n\n"
            "A pixel's own before/after radar change is a weak signal:\nspeckle is strong and clearing "
            "happens at different times\nwithin the year. The model does better because it looks\nat the "
            "neighbourhood (shape, texture of the patch) and at\nevery date, with attention picking the "
            "informative ones.")


def draw(cfg: dict, pixels: dict, n_samples: int, out: Path) -> Path:
    """Six-panel figure from the collected pixel arrays."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    db_bins = np.linspace(*DB_RANGE, 81)
    class_histograms(axes[0, 0], pixels["dvh"], pixels["true"], db_bins, "VH change (dB, after - before)")
    axes[0, 0].set_title("1. VH before/after change of one pixel: the classes overlap", fontsize=9)
    class_histograms(axes[0, 1], pixels["dvv"], pixels["true"], db_bins, "VV change (dB, after - before)")
    axes[0, 1].set_title("2. VV before/after change: overlaps too", fontsize=9)
    class_histograms(axes[0, 2], pixels["prob"], pixels["true"], np.linspace(0, 1, 51), "model P(new clearing)")
    axes[0, 2].axvline(0.5, color="black", linewidth=1.2)
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_title("3. The model's probability separates them (black = 0.5 decision)", fontsize=9)
    plane(axes[1, 0], pixels, pixels["true"].astype(float), "4. True clearing rate by (dVV, dVH): no clean region",
          "fraction of pixels truly cleared")
    plane(axes[1, 1], pixels, pixels["prob"], "5. Model's mean P by (dVV, dVH): nearly flat", "mean P(new clearing)")
    axes[1, 2].axis("off")
    axes[1, 2].text(0, 1, summary_text(pixels, n_samples), fontsize=8.5, va="top", transform=axes[1, 2].transAxes)
    header = smoke_header(cfg)
    label = f"{header['label']} " if header["label"] else ""
    fig.suptitle(f"{label}How deforestation shows up in Sentinel-1 and what the model learned "
                 f"({header['experiment_name']})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training (config_resolved.yaml)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    dataset = build_dataset(data_config(cfg), args.split)
    model = load_model(cfg, args.checkpoint)
    count = min(args.n, len(dataset.meta))
    rows = sorted(int(i) for i in np.random.default_rng(args.seed).choice(len(dataset.meta), count, replace=False))
    pixels = collect(dataset, model, rows)
    out = args.out or run_dir(cfg) / "explain" / f"change_signal_{args.split}_n{count}.png"
    print(draw(cfg, pixels, count, out))


if __name__ == "__main__":
    main()
