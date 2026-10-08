"""Presentation walkthrough of ONE BraDD-S1TS sample: inputs, physical evidence, labels, model pipeline, decision.

`python scripts/plot_sample_walkthrough.py --config <run config_resolved.yaml> --split test
 (--index N | --file NAME.pt | --pick positive) [--checkpoint best.ckpt] [--plot-config configs/eval/walkthrough.yaml]
 --out results/figures/walkthrough [dotted.key=value ...]`

Writes 01_input_timeseries, 02_temporal_profile, 03_labels and 04_pipeline_shapes (no checkpoint needed), plus
05_attention_over_dates and 06_prediction with --checkpoint, and walkthrough.json with every number shown.
Titles carry the sample file and 'SMOKE (synthetic data)' when the data root holds SMOKE.txt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.evaluation.common import load_model  # noqa: E402
from src.evaluation.io import write_json  # noqa: E402
from src.evaluation.walkthrough import (Walkthrough, changed_pixels, load_walkthrough, pipeline_shapes,  # noqa: E402
                                        predict, window_attention, window_keep)
from src.models.build_model import build_model  # noqa: E402
from scripts.walkthrough_figures import COLORS, plot_pipeline  # noqa: E402

CHANNELS = ("VV", "VH")


def stamp(walk: Walkthrough, text: str) -> str:
    """Figure title: SMOKE prefix, text and the sample file name."""
    return f"{walk.prefix}{text}\nsample {walk.file}"


def mark_dates(ax: plt.Axes, walk: Walkthrough) -> None:
    """Vertical lines at the two label dates and the meta.csv alert date."""
    for i, day in enumerate(walk.label_dates):
        ax.axvline(day, color=COLORS["label_date"], linestyle="--", linewidth=2,
                   label="label dates" if i == 0 else None)
    if walk.alert is not None:
        ax.axvline(walk.alert, color=COLORS["alert"], linestyle=":", linewidth=2.5, label="alert date (meta.csv)")


def date_axis(ax: plt.Axes) -> None:
    """YYYY-MM-DD ticks, rotated."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


def frame_indices(walk: Walkthrough, n_frames: int) -> tuple[list[int], set[int]]:
    """Evenly spaced acquisition indices plus the frames nearest each label date (returned separately)."""
    dates = walk.raw["image_dates"]
    even = set(np.linspace(0, len(dates) - 1, n_frames).round().astype(int).tolist())
    nearest = {int(np.argmin([abs((d - day).days) for d in dates])) for day in walk.label_dates}
    return sorted(even | nearest), nearest


def plot_inputs(walk: Walkthrough, plot_cfg: dict, out: Path) -> Path:
    """VV / VH dB frames at evenly spaced dates; frames nearest the label dates get a coloured border."""
    images = walk.raw["image"].numpy()
    picks, nearest = frame_indices(walk, int(plot_cfg["n_frames"]))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.3 * len(picks) + 1.5, 6.2), squeeze=False)
    for c, name in enumerate(CHANNELS):
        low, high = np.percentile(images[:, c], plot_cfg["db_percentiles"])
        for j, k in enumerate(picks):
            ax = axes[c, j]
            shown = ax.imshow(images[k, c], cmap="gray", vmin=low, vmax=high)
            ax.set_xticks([])
            ax.set_yticks([])
            border = COLORS["label_date"] if k in nearest else "white"
            for spine in ax.spines.values():
                spine.set(edgecolor=border, linewidth=5)
            if c == 1:
                ax.set_xlabel(walk.raw["image_dates"][k].isoformat(), fontsize="small")
        axes[c, 0].set_ylabel(name, fontsize="large")
        fig.colorbar(shown, ax=axes[c].tolist(), fraction=0.02, pad=0.01, label=f"{name} (dB)")
    labels = ", ".join(d.isoformat() for d in walk.label_dates)
    fig.suptitle(stamp(walk, f"Sentinel-1 input (raw dB), {len(images)} dates; orange border = nearest to label "
                             f"date ({labels})"))
    return save(fig, out / "01_input_timeseries.png", plot_cfg, tight=False)


def plot_profile(walk: Walkthrough, window_dates: list, plot_cfg: dict, out: Path) -> Path:
    """Mean +/- std dB over time for 0->1 pixels vs 0->0 pixels; label, alert and model-window dates marked."""
    images, (first, last) = walk.raw["image"].numpy(), walk.label_pair.numpy()
    groups = {"deforested 0→1": (first == 0) & (last == 1), "unchanged 0→0": (first == 0) & (last == 0)}
    dates = walk.raw["image_dates"]
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for (name, mask), color in zip(groups.items(), (COLORS["change"], COLORS["stable"])):
        if not mask.any():
            continue
        for c, style in zip(range(2), ("-", "--")):
            values = images[:, c][:, mask]
            mean, std = values.mean(axis=1), values.std(axis=1)
            ax.plot(dates, mean, style, color=color, linewidth=2.5, marker="o", markersize=4,
                    label=f"{CHANNELS[c]}, {name} ({int(mask.sum())} px)")
            ax.fill_between(dates, mean - std, mean + std, color=color, alpha=0.15)
    mark_dates(ax, walk)
    ax.axvspan(window_dates[0], window_dates[-1], color="grey", alpha=0.12, label="dates the model sees")
    ax.set(ylabel="backscatter (dB), mean ± 1 std", xlabel="acquisition date")
    date_axis(ax)
    ax.legend(fontsize="small", ncol=2, loc="lower left")
    ax.set_title(stamp(walk, "Backscatter over time: cleared pixels vs pixels that stay unchanged"))
    return save(fig, out / "02_temporal_profile.png", plot_cfg)


def plot_labels(walk: Walkthrough, plot_cfg: dict, out: Path) -> Path:
    """label[0], label[1] and the change target side by side."""
    first, last = walk.label_pair.numpy()
    panels = [(first, f"label[0] at {walk.label_dates[0].isoformat()}\n(deforested before)", "deforested"),
              (last, f"label[1] at {walk.label_dates[1].isoformat()}\n(deforested by then)", "deforested"),
              ((first != last).astype(int), "change target\nlabel[0] \u2260 label[1]", "change")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), layout="constrained")
    for ax, (mask, title, color) in zip(axes, panels):
        ax.imshow(mask, cmap=ListedColormap([COLORS["background"], COLORS[color]]), vmin=0, vmax=1)
        ax.set(title=title, xlabel=f"{int(mask.sum())} px = 1", xticks=[], yticks=[])
    legend = [Patch(color=COLORS["background"], label="0 = not deforested"),
              Patch(color=COLORS["deforested"], label="1 = deforested"),
              Patch(color=COLORS["change"], label="1 = changed in the interval (what the model predicts)")]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, fontsize="small")
    fig.suptitle(stamp(walk, "PRODES labels and the change-detection target"))
    return save(fig, out / "03_labels.png", plot_cfg, tight=False)


def plot_attention(walk: Walkthrough, att: np.ndarray, window_dates: list, plot_cfg: dict, out: Path) -> Path:
    """Attention weight per date: one thin line per head, the head mean in bold."""
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for h, weights in enumerate(att):
        ax.plot(window_dates, weights, color="grey", alpha=0.45, linewidth=1, label="single head" if h == 0 else None)
    ax.plot(window_dates, att.mean(axis=0), color=COLORS["attention"], linewidth=3, marker="o",
            label=f"mean of {len(att)} heads")
    mark_dates(ax, walk)
    ax.set(ylabel="L-TAE attention weight\n(mean over 6×6 positions)", xlabel="acquisition date", ylim=(0, None))
    date_axis(ax)
    ax.legend(fontsize="small")
    ax.set_title(stamp(walk, f"Which dates the model looks at ({len(window_dates)} dates in its window)"))
    return save(fig, out / "05_attention_over_dates.png", plot_cfg)


def plot_prediction(walk: Walkthrough, result: dict, plot_cfg: dict, out: Path) -> Path:
    """P(change), argmax and the OR-scored map vs label[1] as TP / FP / FN / TN."""
    target, scored = result["target"].numpy(), result["scored"].numpy()
    outcome = np.select([(scored == 1) & (target == 1), (scored == 1) & (target == 0), (scored == 0) & (target == 1)],
                        [0, 1, 2], default=3)
    names = ("TP", "FP", "FN", "TN")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), layout="constrained")
    shown = axes[0].imshow(result["prob"].numpy(), cmap="magma", vmin=0, vmax=1)
    fig.colorbar(shown, ax=axes[0], fraction=0.046, label="P(change) (softmax class 1)")
    axes[1].imshow(result["pred"].numpy(), cmap=ListedColormap([COLORS["background"], COLORS["change"]]), vmin=0, vmax=1)
    axes[2].imshow(outcome, cmap=ListedColormap([COLORS[n] for n in names]), vmin=0, vmax=3)
    titles = ("probability of change", "predicted change (argmax)", "scored: label[0] OR prediction\nvs label[1]")
    for ax, title in zip(axes, titles):
        ax.set(title=title, xticks=[], yticks=[])
    counts = {n: int((outcome == i).sum()) for i, n in enumerate(names)}
    fig.legend(handles=[Patch(color=COLORS[n], label=f"{n} ({counts[n]} px)") for n in names],
               loc="outside lower center", ncol=4, fontsize="small")
    iou = f"pixel IoU {result['with_or']['iou']:.3f} with OR rule, {result['without_or']['iou']:.3f} without"
    fig.suptitle(stamp(walk, f"Model decision ({iou})"))
    return save(fig, out / "06_prediction.png", plot_cfg, tight=False)


def save(fig: plt.Figure, path: Path, plot_cfg: dict, tight: bool = True) -> Path:
    """Save at the configured dpi and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=int(plot_cfg["dpi"]), bbox_inches="tight")
    plt.close(fig)
    return path


def summary(walk: Walkthrough, split: str, shapes: dict, window_dates: list) -> dict:
    """walkthrough.json payload without the checkpoint part."""
    iso = [d.isoformat() for d in walk.raw["image_dates"]]
    return {"label": walk.prefix.strip(), "file": walk.file, "split": split, "meta_row": walk.row,
            "interval": walk.interval, "image_dates": iso, "T_raw": len(iso), "T_window": shapes["T"],
            "window_dates": [d.isoformat() for d in window_dates],
            "label_dates": [d.isoformat() for d in walk.label_dates],
            "alert_date": walk.alert.isoformat() if walk.alert else None,
            "changed_px_0_to_1": changed_pixels(walk.labels, walk.interval),
            "input_shape": list(shapes["input"]), "stage_shapes": {k: list(v) for k, v in shapes["stages"].items()}}


def checkpoint_part(model: object, walk: Walkthrough, window_dates: list, plot_cfg: dict, out: Path) -> dict:
    """Figures 5 and 6 and their numbers for walkthrough.json."""
    att = window_attention(model, walk.batch, walk.interval).numpy()
    plot_attention(walk, att, window_dates, plot_cfg, out)
    result = predict(model, walk)
    plot_prediction(walk, result, plot_cfg, out)
    return {"attention_mean_per_date": att.mean(axis=0).tolist(), "attention_per_head": att.tolist(),
            "pixel_with_or": result["with_or"], "pixel_without_or": result["without_or"],
            "iou_with_or": result["with_or"]["iou"], "iou_without_or": result["without_or"]["iou"]}


def run(cfg: dict, args: argparse.Namespace, plot_cfg: dict) -> dict:
    """Render every figure the inputs allow and write walkthrough.json; return its payload."""
    plt.rcParams.update({"font.size": int(plot_cfg["font_size"])})
    walk = load_walkthrough(cfg, args.split, args.index, args.file, plot_cfg)
    print(f"walkthrough sample: {walk.file} (split {args.split}, meta row {walk.row})")
    model = load_model(cfg, args.checkpoint) if args.checkpoint else build_model(cfg["model"]).eval()
    keep = window_keep(model, walk.batch, walk.interval).tolist()
    window_dates = [d for d, k in zip(walk.dates, keep) if k]
    shapes = pipeline_shapes(model, walk.batch, walk.interval)
    plot_inputs(walk, plot_cfg, args.out)
    plot_profile(walk, window_dates, plot_cfg, args.out)
    plot_labels(walk, plot_cfg, args.out)
    plot_pipeline(shapes, stamp(walk, f"U-TAE on this sample: T = {shapes['T']} dates in the label window"),
                  args.out / "04_pipeline_shapes.png", int(plot_cfg["dpi"]))
    payload = summary(walk, args.split, shapes, window_dates) | {"checkpoint": args.checkpoint}
    if args.checkpoint:
        payload |= checkpoint_part(model, walk, window_dates, plot_cfg, args.out)
    write_json(args.out / "walkthrough.json", payload)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config_resolved.yaml or a training config")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--index", type=int, help="row of the split (meta order)")
    choice.add_argument("--file", help="sample file name, e.g. 0000123_2021-03-01.pt")
    choice.add_argument("--pick", choices=["positive"], help="first sample with a large 0->1 change (default)")
    parser.add_argument("--checkpoint", help="src.train Lightning checkpoint; enables figures 5 and 6")
    parser.add_argument("--plot-config", default="configs/eval/walkthrough.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides), args, load_config(args.plot_config))


if __name__ == "__main__":
    main()
