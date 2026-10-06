"""Read-only helpers for the Phase 0 inspection of a BraDD-S1TS archive (PRD Phase 0)."""
from __future__ import annotations

import datetime as dt
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.samples import load_sample

CHANNELS = ("VV", "VH")
VARIABLE_LENGTH_KEYS = ("image", "image_dates")  # first dim is T, varies per sample


def find_dataset_root(top: Path) -> Path:
    """Directory containing meta.csv (the archive top itself or the shallowest match below it)."""
    matches = sorted(Path(top).rglob("meta.csv"), key=lambda p: len(p.parts))
    if not matches:
        raise FileNotFoundError(f"no meta.csv under {top}")
    return matches[0].parent


def list_archive(top: Path) -> pd.DataFrame:
    """Every file under `top` with relative path, directory, extension and size."""
    rows = [{"path": str(p.relative_to(top)), "directory": str(p.parent.relative_to(top)),
             "extension": p.suffix.lower() or "(none)", "size_bytes": p.stat().st_size}
            for p in sorted(Path(top).rglob("*")) if p.is_file()]
    return pd.DataFrame(rows, columns=["path", "directory", "extension", "size_bytes"])


def count_by(files: pd.DataFrame, column: str) -> pd.DataFrame:
    """File count and total size grouped by one column."""
    grouped = files.groupby(column)["size_bytes"].agg(["count", "sum"]).reset_index()
    return grouped.rename(columns={"sum": "size_bytes"}).sort_values("count", ascending=False)


def non_sample_files(files: pd.DataFrame, samples_dir: str = "Samples") -> pd.DataFrame:
    """Files that are not inside a `Samples` directory (listed by name in the report)."""
    inside = files["directory"].map(lambda d: samples_dir in Path(d).parts)
    return files[~inside].reset_index(drop=True)


def geolocation_hits(meta_columns: Sequence[str], sample_keys: Sequence[str], files: pd.DataFrame,
                     pattern: str, extensions: Sequence[str]) -> pd.DataFrame:
    """Meta columns, sample keys and side files that could geolocate patches."""
    regex = re.compile(pattern, re.IGNORECASE)
    rows = [{"where": "meta.csv column", "name": c} for c in meta_columns if regex.search(c)]
    rows += [{"where": "sample key", "name": k} for k in sample_keys if regex.search(k)]
    side = files[files["extension"].isin(extensions) & (files["path"].map(lambda p: Path(p).name) != "meta.csv")]
    rows += [{"where": "file", "name": p} for p in side["path"]]
    return pd.DataFrame(rows, columns=["where", "name"])


def _as_date(value: object) -> dt.date:
    """Coerce a date-like value (date, datetime, ISO string) to datetime.date."""
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def key_schema(sample: Mapping) -> list[str]:
    """One 'key: type dtype shape' string per entry of a sample dict."""
    out = []
    for key, value in sample.items():
        first = "T" if key in VARIABLE_LENGTH_KEYS else None
        if isinstance(value, torch.Tensor):
            shape = tuple(value.shape) if first is None else (first, *value.shape[1:])
            out.append(f"{key}: Tensor {value.dtype} {shape}")
        elif isinstance(value, (list, tuple)):
            inner = type(value[0]).__name__ if value else "empty"
            out.append(f"{key}: {type(value).__name__}[{inner}] len={first or len(value)}")
        else:
            out.append(f"{key}: {type(value).__name__}")
    return out


def _date_offsets(sample: Mapping, alert: dt.date) -> dict:
    """Label/image date positions relative to the meta alert date, in days."""
    images = [_as_date(d) for d in sample["image_dates"]]
    labels = [_as_date(d) for d in sample["label_dates"]]
    gaps = np.diff([d.toordinal() for d in images])
    return {"T": len(images), "n_label_dates": len(labels),
            "first_label_minus_alert": (labels[0] - alert).days, "last_label_minus_alert": (labels[-1] - alert).days,
            "first_image_minus_alert": (images[0] - alert).days, "last_image_minus_alert": (images[-1] - alert).days,
            "image_span_days": (images[-1] - images[0]).days,
            "median_revisit_days": float(np.median(gaps)) if len(gaps) else float("nan")}


def _value_stats(image: torch.Tensor) -> dict:
    """Per-channel dB min/max/mean/1st/99th percentile and the non-finite count."""
    row = {"nonfinite": int((~torch.isfinite(image)).sum())}
    for c, name in enumerate(CHANNELS[: image.shape[1]]):
        x = image[:, c].flatten().double()
        x = x[torch.isfinite(x)]
        q = torch.quantile(x, torch.tensor([0.01, 0.99], dtype=torch.float64))
        row.update({f"{name}_min": x.min().item(), f"{name}_max": x.max().item(), f"{name}_mean": x.mean().item(),
                    f"{name}_p01": q[0].item(), f"{name}_p99": q[1].item()})
    return row


def _transitions(label: torch.Tensor) -> dict:
    """Pixel transition counts between the first and last label mask."""
    codes = torch.bincount((label[-1].long() + 2 * label[0].long()).flatten(), minlength=4)
    return dict(zip(("n_0to0", "n_0to1", "n_1to0", "n_1to1"), codes.tolist()))


def inspect_samples(root: Path, meta: pd.DataFrame, n_samples: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-sample facts for a seeded subset (all if n_samples < 0) and the key-schema counts."""
    rows_idx = np.arange(len(meta)) if n_samples < 0 or n_samples >= len(meta) else np.sort(
        np.random.default_rng(seed).choice(len(meta), n_samples, replace=False))
    rows, schemas = [], []
    for i in rows_idx:
        record = meta.iloc[int(i)]
        sample = load_sample(Path(root) / "Samples" / record["file"])
        schemas.extend(key_schema(sample))
        row = {"file": record["file"], "sampling_type": record.get("sampling_type"),
               "image_dtype": str(sample["image"].dtype), "label_dtype": str(sample["label"].dtype),
               "image_shape": tuple(sample["image"].shape), "label_shape": tuple(sample["label"].shape)}
        row.update(_date_offsets(sample, _as_date(record["date"])))
        row.update(_value_stats(sample["image"]))
        row.update(_transitions(sample["label"]))
        rows.append(row)
    schema = pd.Series(schemas).value_counts().rename_axis("key_schema").reset_index(name="n_samples")
    return pd.DataFrame(rows), schema


def split_counts(meta: pd.DataFrame, split_column: str, paper: Mapping[str, int],
                 notebook: Mapping[str, int]) -> pd.DataFrame:
    """Counts per split value next to the paper's and the upstream notebook's numbers."""
    counts = meta[split_column].value_counts()
    names = sorted(set(counts.index) | set(paper) | set(notebook))
    return pd.DataFrame({"split": names, "meta_csv": [int(counts.get(n, 0)) for n in names],
                         "paper": [paper.get(n) for n in names], "upstream_notebook": [notebook.get(n) for n in names]})


def markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependency)."""
    shown = df if max_rows is None else df.head(max_rows)
    lines = ["| " + " | ".join(map(str, shown.columns)) + " |", "|" + "---|" * len(shown.columns)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in shown.itertuples(index=False)]
    if max_rows is not None and len(df) > max_rows:
        lines.append(f"\n({len(df) - max_rows} more rows in the CSV)")
    return "\n".join(lines)
