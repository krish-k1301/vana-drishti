"""Demo figure: Sentinel-1 before/after, true new clearing, predicted change and an error overlay per test sample.

`python scripts/plot_predictions.py --config <run>/config_resolved.yaml --checkpoint <best.ckpt>
 [--split test] [--n-positive 6] [--n-negative 2] [--negative-type forest] [--seed 0] [--out FILE]`

Samples are drawn (seeded) from the split's `sampling_type == positive` rows and from `--negative-type` rows.
For each sample the label interval with the most true change is shown. Overlay colours: green = correctly
detected clearing, red = false alarm, yellow = missed clearing. Titles carry the run label (SMOKE).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.data.collate import collate_batch  # noqa: E402
from src.data.loaders import build_dataset  # noqa: E402
from src.evaluation.common import load_model, run_dir, smoke_header  # noqa: E402
from src.train import data_config  # noqa: E402

OVERLAY = ListedColormap(["#00000000", "#2ca02c", "#d62728", "#ffbf00"])  # none, hit, false alarm, missed
COLUMNS = ("VV before", "VV after", "True new clearing", "Predicted change", "Errors")


def pick_rows(meta: object, n_positive: int, n_negative: int, negative_type: str, seed: int) -> list[int]:
    """Seeded row indices: `n_positive` positive samples then `n_negative` samples of `negative_type`."""
    rng = np.random.default_rng(seed)
    rows = []
    for kind, count in (("positive", n_positive), (negative_type, n_negative)):
        candidates = np.flatnonzero(meta["sampling_type"].to_numpy() == kind)
        if count > len(candidates):
            raise ValueError(f"asked for {count} '{kind}' samples, the split has {len(candidates)}")
        rows += sorted(int(i) for i in rng.choice(candidates, size=count, replace=False))
    return rows


@torch.no_grad()
def predict(model: torch.nn.Module, item: dict) -> dict:
    """Change probability, true change and prior label of the valid interval with the most true change."""
    batch = collate_batch([item])
    logits = model(batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])[0]
    valid = model.interval_mask(batch["ImageDays"], batch["TargetDays"], batch["PadMask"])[0]
    targets = item["Targets"]
    change = (targets[:-1] != targets[1:]).long()
    amount = change.flatten(1).sum(1).masked_fill(~valid, -1)
    k = int(amount.argmax())
    return {"interval": k, "prob": logits[k].softmax(0)[1].numpy(), "pred": logits[k].argmax(0).numpy(),
            "true": change[k].numpy(), "prior": targets[k].numpy()}


def frame_at(raw: dict, label_date: object) -> tuple[np.ndarray, object]:
    """VV channel (dB) of the last image acquired on or before `label_date` (first image if none) and its date."""
    dates = raw["image_dates"]
    before = [i for i, d in enumerate(dates) if d <= label_date]
    index = before[-1] if before else 0
    return raw["image"][index, 0].numpy(), dates[index]


def error_map(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """0 = nothing, 1 = hit, 2 = false alarm, 3 = missed clearing."""
    out = np.zeros_like(true)
    out[(true == 1) & (pred == 1)] = 1
    out[(true == 0) & (pred == 1)] = 2
    out[(true == 1) & (pred == 0)] = 3
    return out


def sample_iou(true: np.ndarray, pred: np.ndarray) -> str:
    """IoU of predicted vs true change for one sample, or a note when neither has any change pixel."""
    union = int(((true == 1) | (pred == 1)).sum())
    if union == 0:
        return "no change, none predicted"
    return f"IoU {int(((true == 1) & (pred == 1)).sum()) / union:.2f}"


def draw_row(axes: np.ndarray, raw: dict, result: dict, name: str) -> None:
    """Fill one figure row for a sample."""
    k = result["interval"]
    before, before_date = frame_at(raw, raw["label_dates"][k])
    after, after_date = frame_at(raw, raw["label_dates"][k + 1])
    low, high = np.percentile(np.stack([before, after]), [2, 98])
    axes[0].imshow(before, cmap="gray", vmin=low, vmax=high)
    axes[0].set_title(f"{name}\nVV {before_date}", fontsize=8)
    axes[1].imshow(after, cmap="gray", vmin=low, vmax=high)
    axes[1].set_title(f"VV {after_date}", fontsize=8)
    axes[2].imshow(result["true"], cmap="Greens", vmin=0, vmax=1)
    axes[2].set_title(f"true change ({int(result['true'].sum())} px)", fontsize=8)
    axes[3].imshow(result["prob"], cmap="magma", vmin=0, vmax=1)
    axes[3].set_title(f"predicted probability\n{sample_iou(result['true'], result['pred'])}", fontsize=8)
    axes[4].imshow(after, cmap="gray", vmin=low, vmax=high)
    axes[4].imshow(error_map(result["true"], result["pred"]), cmap=OVERLAY, vmin=0, vmax=3, interpolation="nearest")
    axes[4].set_title("green hit, red false alarm,\nyellow missed", fontsize=8)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])


def plot(cfg: dict, checkpoint: str, split: str, rows: list[int], out: Path) -> Path:
    """Run the model on the chosen rows of `split` and save the figure to `out`."""
    dataset = build_dataset(data_config(cfg), split)
    model = load_model(cfg, checkpoint)
    fig, axes = plt.subplots(len(rows), len(COLUMNS), figsize=(2.3 * len(COLUMNS), 2.5 * len(rows)), squeeze=False)
    for ax_row, row in zip(axes, rows):
        raw = dataset.load_raw(row)
        item = dataset.apply_transforms(dataset.build_item(raw, row))
        meta = dataset.meta.loc[row]
        draw_row(ax_row, raw, predict(model, item), f"{meta['sampling_type']} | {meta['state']} | {meta['file']}")
    header = smoke_header(cfg)
    label = f"{header['label']} " if header["label"] else ""
    fig.suptitle(f"{label}{header['experiment_name']}: {split} samples, checkpoint {Path(checkpoint).name}",
                 fontsize=10)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training (config_resolved.yaml)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--n-positive", type=int, default=6)
    parser.add_argument("--n-negative", type=int, default=2)
    parser.add_argument("--negative-type", default="forest", help="sampling_type of the no-new-clearing rows")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    meta = build_dataset(data_config(cfg), args.split).meta
    rows = pick_rows(meta, args.n_positive, args.n_negative, args.negative_type, args.seed)
    out = args.out or run_dir(cfg) / f"predictions_{args.split}_seed{args.seed}.png"
    print(plot(cfg, args.checkpoint, args.split, rows, out))


if __name__ == "__main__":
    main()
