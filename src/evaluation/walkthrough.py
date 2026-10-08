"""One-sample walkthrough data for the presentation figures of scripts/plot_sample_walkthrough.py.

Everything the model sees comes from the evaluation path: the run config's dataset (normalisation stats,
test-time transforms), `collate_batch` (pad mask) and the network's own `SegmentNetwork.window` / `_select`
for the dates of one label interval. Raw dB values and dates come from the sample file on disk.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from src.config import require
from src.data.collate import collate_batch
from src.data.loaders import build_dataset
from src.data.samples import sample_origin
from src.metrics.segmentation import change_detection_outputs, confusion_matrix, scores_from_matrix
from src.models.inspection import describe_utae_shapes
from src.models.segment_network import SegmentNetwork
from src.train import data_config

SMOKE_MARKER = "SMOKE.txt"


@dataclass
class Walkthrough:
    """One sample: raw on-disk record, the collated model input and the dates that go with it."""

    file: str
    row: int
    interval: int
    raw: dict
    labels: torch.Tensor
    batch: dict
    dates: list[dt.date]
    alert: dt.date | None
    prefix: str

    @property
    def label_dates(self) -> list[dt.date]:
        """The two label dates that bound the explained interval."""
        return list(self.raw["label_dates"][self.interval:self.interval + 2])

    @property
    def label_pair(self) -> torch.Tensor:
        """Labels [2,H,W] at the start and end of the explained interval."""
        return self.labels[self.interval:self.interval + 2]


def changed_pixels(labels: torch.Tensor, interval: int = 0) -> int:
    """Number of pixels that go 0 -> 1 between label[interval] and label[interval + 1]."""
    return int(((labels[interval] == 0) & (labels[interval + 1] == 1)).sum())


def smoke_prefix(cfg: dict) -> str:
    """'SMOKE (synthetic data) ' if the data root holds SMOKE.txt, 'SMOKE ' for a SMOKE run, else ''."""
    if (Path(require(cfg, "data.root")) / SMOKE_MARKER).is_file():
        return "SMOKE (synthetic data) "
    return "SMOKE " if bool(cfg.get("smoke")) else ""


def pick_positive(dataset: torch.utils.data.Dataset, min_px: int, interval: int) -> int:
    """First meta row whose label interval has at least `min_px` pixels changing 0 -> 1."""
    for row in range(len(dataset.meta)):
        if changed_pixels(dataset.load_raw(row)[dataset.label_key], interval) >= min_px:
            return row
    raise ValueError(f"no sample in this split has >= {min_px} changed (0->1) pixels; lower pick_min_changed_px")


def select_row(dataset: torch.utils.data.Dataset, index: int | None, file: str | None, min_px: int,
               interval: int) -> int:
    """Meta row chosen by --index, --file, or else the first clearly positive sample."""
    if file is not None:
        hits = dataset.meta.index[dataset.meta["file"] == file].tolist()
        if not hits:
            raise ValueError(f"file {file} is not in this split")
        return int(hits[0])
    if index is not None:
        if not 0 <= index < len(dataset.meta):
            raise IndexError(f"--index {index} outside 0..{len(dataset.meta) - 1}")
        return index
    return pick_positive(dataset, min_px, interval)


def load_walkthrough(cfg: dict, split: str, index: int | None, file: str | None, plot_cfg: dict) -> Walkthrough:
    """Build the evaluation dataset of the run config and load one sample through it."""
    dataset = build_dataset(data_config(cfg), split)
    if dataset.tiles_per_sample != 1:
        raise ValueError("the walkthrough shows whole patches; set data.crop.enabled=false")
    interval = int(plot_cfg["interval"])
    row = select_row(dataset, index, file, int(plot_cfg["pick_min_changed_px"]), interval)
    raw = dataset.load_raw(row)
    item = dataset.apply_transforms(dataset.build_item(raw, row))
    origin = sample_origin(raw)
    dates = [origin + dt.timedelta(days=int(d) - 1) for d in item["ImageDays"]]
    meta = dataset.meta.loc[row]
    alert = dt.date.fromisoformat(meta["date"]) if meta.get("date") else None
    return Walkthrough(file=str(meta["file"]), row=row, interval=interval, raw=raw, labels=raw[dataset.label_key],
                       batch=collate_batch([item]), dates=dates, alert=alert, prefix=smoke_prefix(cfg))


def _args(batch: dict) -> tuple[torch.Tensor, ...]:
    """Model arguments (images, days, target days, pad mask) of a collated batch."""
    return batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"]


def window_keep(model: SegmentNetwork, batch: dict, interval: int) -> torch.Tensor:
    """[T] bool: the dates SegmentNetwork feeds to the backbone for this label interval."""
    _, days, target_days, pad_mask = _args(batch)
    if not bool(model.interval_mask(days, target_days, pad_mask)[0, interval]):
        raise ValueError(f"label interval {interval} has no image in the model's date window")
    return model.window(days, target_days, ~pad_mask, interval)[0]


def pipeline_shapes(model: SegmentNetwork, batch: dict, interval: int) -> dict:
    """Input shape, window length T and per-stage [C,H,W] of the real U-TAE (forward hooks)."""
    backbone = model.backbone
    if not hasattr(backbone, "temporal_encoder"):
        raise ValueError("the pipeline and attention figures need a U-TAE backbone")
    num_dates = int(window_keep(model, batch, interval).sum())
    _, _, channels, h, w = batch["Images"].shape
    stages = describe_utae_shapes(backbone, h, w, num_dates=num_dates)
    return {"T": num_dates, "input": (num_dates, channels, h, w), "stages": stages}


@torch.no_grad()
def window_attention(model: SegmentNetwork, batch: dict, interval: int) -> torch.Tensor:
    """L-TAE attention [n_head, T_window], mean over the bottleneck positions, for the model's own window."""
    images, days, _, pad_mask = _args(batch)
    images = model._apply_pad_mask(images, pad_mask)
    keep = window_keep(model, batch, interval)[None]
    window_images, window_days = model._select(images, days, keep)
    _, att = model.backbone(window_images, window_days, return_att=True)
    return att[:, 0].mean(dim=(-2, -1))


def _scores(logits: torch.Tensor, targets: torch.Tensor, use_or: bool) -> tuple[torch.Tensor, dict]:
    """Scored map [H,W] and pixel scores of one interval, upstream rule with or without the OR."""
    out = change_detection_outputs(logits, targets, use_or=use_or)
    scores = scores_from_matrix(confusion_matrix(out["score_pred"], out["score_target"], 2), 1)
    return out["score_pred"][0], scores


@torch.no_grad()
def predict(model: nn.Module, walk: Walkthrough) -> dict:
    """P(change), argmax prediction, OR-scored map and per-sample IoU with and without the OR rule."""
    logits = model(*_args(walk.batch))[:, walk.interval:walk.interval + 1]
    targets = walk.batch["Targets"][:, walk.interval:walk.interval + 2]
    scored, with_or = _scores(logits, targets, use_or=True)
    _, without_or = _scores(logits, targets, use_or=False)
    return {"prob": logits.softmax(dim=2)[0, 0, 1], "pred": logits.argmax(dim=2)[0, 0], "scored": scored,
            "target": targets[0, 1], "with_or": with_or, "without_or": without_or}
