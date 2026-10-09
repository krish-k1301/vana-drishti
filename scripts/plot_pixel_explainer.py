"""Explainer figure, one per sample: how U-TAE turns a Sentinel-1 time series into a per-pixel clearing decision.

`python scripts/plot_pixel_explainer.py --config <run>/config_resolved.yaml --checkpoint <best.ckpt>
 [--split test] [--n 3] [--seed 0] [--files F1.pt F2.pt] [--frames 8] [--out-dir DIR] [dotted.key=value ...]`

Rows: (1) the radar frames the model sees for one label interval (VV, dB); (2) the change probability with example
pixels marked, the truth/error map and the decision rule; (3) each example pixel's VV and VH backscatter over time
with the dates the temporal attention weighted, read from the 6x6 attention cell that contains the pixel.
Samples are seeded picks of `sampling_type == positive` rows unless `--files` names them. Titles carry SMOKE.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.loaders import build_dataset  # noqa: E402
from src.evaluation.common import load_model, run_dir, smoke_header  # noqa: E402
from src.evaluation.explain import (attention_cell, busiest_interval, explain_interval, neighbourhood_series,  # noqa: E402
                                    pick_pixels)
from src.train import data_config  # noqa: E402

COLORS = {"detected clearing": "#2ca02c", "stable, no change": "#1f77b4", "missed clearing": "#ffbf00",
          "false alarm": "#d62728"}
ERRORS = ListedColormap(["#f2f2f2", "#2ca02c", "#d62728", "#ffbf00"])  # stable, hit, false alarm, missed
STEPS = ("1. input: every Sentinel-1 frame in the interval (+-30 days), VV and VH, z-scored",
         "2. spatial encoder: convolutions per frame, 48x48 -> 6x6 feature maps",
         "3. temporal attention: per 6x6 cell, weights over dates -> one summary per cell",
         "4. decoder: back to 48x48, 2 scores per pixel -> softmax -> P(new clearing)",
         "5. decision: P >= 0.5 -> clearing (argmax of the 2 classes)")


def draw_frames(fig: plt.Figure, grid: object, raw: dict, result: dict, n_frames: int) -> None:
    """Top row: up to `n_frames` evenly spaced VV frames of the interval, shared grey scale."""
    kept = result["kept"]
    shown = kept[np.unique(np.linspace(0, len(kept) - 1, min(n_frames, len(kept))).round().astype(int))]
    stack = raw["image"][shown, 0].numpy()
    low, high = np.percentile(stack, [2, 98])
    sub = grid.subgridspec(1, len(shown), wspace=0.05)
    for j, index in enumerate(shown):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(stack[j], cmap="gray", vmin=low, vmax=high)
        ax.set_title(str(raw["image_dates"][index]), fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.01, grid.get_position(fig).y1 + 0.005,
             f"Radar frames the model sees (VV dB, {len(kept)} dates in the window; {len(shown)} shown)", fontsize=9)


def mark(ax: plt.Axes, picks: dict) -> None:
    """Circle and letter each example pixel."""
    for letter, (name, (row, col)) in zip("ABCD", picks.items()):
        ax.scatter([col], [row], s=90, facecolors="none", edgecolors=COLORS[name], linewidths=2)
        ax.annotate(letter, (col, row), xytext=(4, -4), textcoords="offset points", color=COLORS[name],
                    fontsize=10, fontweight="bold")


def draw_maps(fig: plt.Figure, grid: object, result: dict, picks: dict) -> None:
    """Middle row: probability map, truth/error map, and the pipeline with each example pixel's probability."""
    sub = grid.subgridspec(1, 3, width_ratios=[1, 1, 1.5], wspace=0.15)
    ax = fig.add_subplot(sub[0, 0])
    image = ax.imshow(result["prob"], cmap="magma", vmin=0, vmax=1)
    ax.contour(result["prob"], levels=[0.5], colors="white", linewidths=0.8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    ax.set_title("P(new clearing) per pixel\nwhite line = 0.5 threshold", fontsize=8)
    mark(ax, picks)
    ax = fig.add_subplot(sub[0, 1])
    errors = np.zeros_like(result["true"])
    errors[(result["true"] == 1) & (result["pred"] == 1)] = 1
    errors[(result["true"] == 0) & (result["pred"] == 1)] = 2
    errors[(result["true"] == 1) & (result["pred"] == 0)] = 3
    ax.imshow(errors, cmap=ERRORS, vmin=0, vmax=3, interpolation="nearest")
    ax.set_title("vs PRODES truth: green hit, red false alarm,\nyellow missed, grey stable", fontsize=8)
    mark(ax, picks)
    for a in fig.axes[-3:]:
        a.set_xticks([])
        a.set_yticks([])
    ax = fig.add_subplot(sub[0, 2])
    ax.axis("off")
    lines = [*STEPS, ""] + [f"{letter} ({name}): P = {result['prob'][rc]:.2f} -> "
                            f"{'clearing' if result['pred'][rc] else 'no clearing'}"
                            for letter, (name, rc) in zip("ABCD", picks.items())]
    for i, line in enumerate(lines):
        name = next((n for n in COLORS if f"({n})" in line), None)
        ax.text(0, 1 - i * 0.1, line, fontsize=7.5, va="top", color=COLORS.get(name, "black"),
                fontweight="bold" if name else "normal", transform=ax.transAxes)


def draw_series(fig: plt.Figure, grid: object, raw: dict, result: dict, picks: dict) -> None:
    """Bottom row: per example pixel, VV/VH dB over all dates, interval shaded, attention weights as bars."""
    sub = grid.subgridspec(1, len(picks), wspace=0.3)
    dates = raw["image_dates"]
    start, end = raw["label_dates"][result["interval"]], raw["label_dates"][result["interval"] + 1]
    size = result["prob"].shape
    for j, (letter, (name, (row, col))) in enumerate(zip("ABCD", picks.items())):
        ax = fig.add_subplot(sub[0, j])
        ax.axvspan(start, end, color="#eeeeee", zorder=0)
        smooth = neighbourhood_series(raw["image"].numpy(), (row, col), radius=2)
        for channel, (band, color) in enumerate((("VV", "C0"), ("VH", "C1"))):
            ax.plot(dates, raw["image"][:, channel, row, col].numpy(), lw=0.6, color=color, alpha=0.3)
            ax.plot(dates, smooth[:, channel], "-o", ms=2.5, lw=1.4, color=color, label=f"{band} (5x5 mean)")
        twin = ax.twinx()
        weights = attention_cell(result["attention"], (row, col), size)
        twin.bar([dates[i] for i in result["kept"]], weights, width=6, color=COLORS[name], alpha=0.45)
        twin.set_ylim(0, max(0.05, float(weights.max()) * 2.5))
        twin.tick_params(labelsize=6)
        ax.set_title(f"{letter}: {name}, pixel ({row},{col}), P = {result['prob'][row, col]:.2f}", fontsize=8,
                     color=COLORS[name])
        ax.set_ylabel("backscatter (dB)", fontsize=7)
        twin.set_ylabel("attention weight", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.tick_params(axis="x", rotation=30)
        if j == 0:
            ax.legend(fontsize=6, loc="lower left")
    fig.text(0.01, grid.get_position(fig).y1 + 0.03,
             f"Pixel time series (grey band = label interval {start} -> {end}; bold = 5x5-pixel mean, "
             "faint = the pixel itself; bars = attention of its 8x8-pixel cell)", fontsize=9)


def explain_sample(cfg: dict, dataset: object, model: object, row: int, n_frames: int, out: Path) -> Path:
    """Build and save the explainer figure for one dataset row."""
    raw = dataset.load_raw(row)
    item = dataset.apply_transforms(dataset.build_item(raw, row))
    result = explain_interval(model, item, busiest_interval(model, item))
    picks = pick_pixels(result)
    fig = plt.figure(figsize=(14, 11.5))
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1.5, 1.5], hspace=0.35, top=0.92, bottom=0.06, left=0.05,
                            right=0.96)
    draw_frames(fig, grid[0], raw, result, n_frames)
    draw_maps(fig, grid[1], result, picks)
    draw_series(fig, grid[2], raw, result, picks)
    meta = dataset.meta.loc[row]
    header = smoke_header(cfg)
    label = f"{header['label']} " if header["label"] else ""
    fig.suptitle(f"{label}How the model decides, pixel by pixel: {meta['file']} ({meta['sampling_type']}, "
                 f"{meta['state']})", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def choose_rows(meta: object, files: list[str] | None, n: int, seed: int) -> list[int]:
    """Rows of the named files, or `n` seeded positive rows."""
    if files:
        found = {file: i for i, file in enumerate(meta["file"])}
        missing = [file for file in files if file not in found]
        if missing:
            raise ValueError(f"files not in this split: {missing}")
        return [found[file] for file in files]
    positives = np.flatnonzero(meta["sampling_type"].to_numpy() == "positive")
    return sorted(int(i) for i in np.random.default_rng(seed).choice(positives, size=n, replace=False))


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training (config_resolved.yaml)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--files", nargs="+", help="sample files (meta.csv `file`) instead of seeded picks")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    dataset = build_dataset(data_config(cfg), args.split)
    model = load_model(cfg, args.checkpoint)
    out_dir = args.out_dir or run_dir(cfg) / "explain"
    for row in choose_rows(dataset.meta, args.files, args.n, args.seed):
        name = Path(dataset.meta.loc[row, "file"]).stem
        print(explain_sample(cfg, dataset, model, row, args.frames, out_dir / f"explain_{name}.png"))


if __name__ == "__main__":
    main()
