"""Shared colours and the U-TAE shape schematic for scripts/plot_sample_walkthrough.py."""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

COLORS = {
    "background": "#e6e6e6", "deforested": "#8c564b", "change": "#d62728", "stable": "#2ca02c",
    "label_date": "#ff7f0e", "alert": "#17becf", "attention": "#6a3d9a",
    "TP": "#d62728", "FP": "#ffbf00", "FN": "#1f77b4", "TN": "#e6e6e6",
    "input": "#8c8c8c", "encoder": "#4c72b0", "temporal": "#dd8452", "decoder": "#55a868",
}
CAPTIONS = {"in_conv": "input conv\n(each date)", "down": "encoder {}\n(each date)",
            "temporal_encoder": "L-TAE\nattention over dates", "up": "decoder {}", "out_conv": "output conv"}


def shape_text(shape: tuple[int, ...] | list[int], with_t: int | None = None) -> str:
    """'T x C x H x W' text, with the date axis prepended when `with_t` is given."""
    dims = ([with_t] if with_t is not None else []) + list(shape)
    return " × ".join(str(d) for d in dims)


def pipeline_boxes(shapes: dict) -> list[tuple[str, str, str, int]]:
    """(caption, shape text, colour key, spatial size) per stage, input first, in forward order."""
    t, (_, c, h, w) = shapes["T"], shapes["input"]
    boxes = [(f"input\n{t} dates", shape_text((c, h, w), t), "input", h)]
    for name, shape in shapes["stages"].items():
        stage, _, number = name.partition("_") if name.startswith(("down_", "up_")) else (name, "", "")
        caption = CAPTIONS[stage].format(number)
        per_date = stage in ("in_conv", "down")
        kind = "encoder" if per_date else "temporal" if stage == "temporal_encoder" else "decoder"
        boxes.append((caption, shape_text(shape, t if per_date else None), kind, shape[-1]))
    return boxes


def box_positions(boxes: list[tuple[str, str, str, int]]) -> list[tuple[float, float]]:
    """U-shaped layout: x in forward order, y one level down per halving of the spatial size."""
    top = boxes[0][3]
    return [(2.7 * i, -1.6 * math.log2(top / size)) for i, (*_, size) in enumerate(boxes)]


def plot_pipeline(shapes: dict, title: str, path: Path, dpi: int) -> Path:
    """Draw the stage boxes with their real shapes, forward arrows and attention-weighted skip connections."""
    boxes = pipeline_boxes(shapes)
    xy = box_positions(boxes)
    fig, ax = plt.subplots(figsize=(2.7 * len(boxes) + 1, 9))
    for (caption, shape, kind, _), (x, y) in zip(boxes, xy):
        ax.add_patch(FancyBboxPatch((x - 1.25, y - 0.6), 2.5, 1.2, boxstyle="round,pad=0.05",
                                    facecolor=COLORS[kind], alpha=0.85, edgecolor="black"))
        ax.text(x, y + 0.18, caption, ha="center", va="center", color="white", fontsize=14, weight="bold")
        ax.text(x, y - 0.38, shape, ha="center", va="center", color="white", fontsize=14)
    for (x0, y0), (x1, y1) in zip(xy, xy[1:]):
        ax.annotate("", xy=(x1 - 1.3, y1), xytext=(x0 + 1.3, y0), arrowprops={"arrowstyle": "-|>", "lw": 2})
    temporal = next(i for i, b in enumerate(boxes) if b[2] == "temporal")
    for i in range(1, temporal - 1):
        j = len(boxes) - 1 - i
        ax.annotate("", xy=(xy[j][0], xy[j][1] + 0.65), xytext=(xy[i][0], xy[i][1] + 0.65),
                    arrowprops={"arrowstyle": "-|>", "lw": 1.5, "linestyle": "--", "color": COLORS["temporal"],
                                "connectionstyle": "arc3,rad=-0.12"})
    ax.text(xy[temporal][0], xy[temporal][1] - 1.1, "one attention weight per date and head,\nre-used to merge the "
            "skip connections (dashed)", ha="center", va="top", fontsize=14, color=COLORS["temporal"])
    ax.set_xlim(min(x for x, _ in xy) - 1.5, max(x for x, _ in xy) + 1.5)
    ax.set_ylim(min(y for _, y in xy) - 2.4, 1.8)
    ax.axis("off")
    ax.set_title(f"{title}\nshapes per stage: [dates ×] channels × height × width (from forward hooks)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path
