"""Collect evaluated runs into one benchmark table (PRD 7 Phase 2). Only numbers read from files.

`python -m src.benchmark_table --runs RUN_DIR [RUN_DIR ...] --out results/benchmark.csv
 [--eval-name eval_test] [--reference-model utae]`

Per run it reads `config_resolved.yaml` (model, SMOKE flag), `<eval-name>/metrics.json` (written by src.evaluate:
pixel/patch scores, parameter count, MMU) and `csv/version_0/metrics.csv` (Lightning CSV log: mean
`train/epoch_time_s`, max `train/peak_gpu_mem_mb`, blank when trained on CPU). The reference model's rows come
first and every row gets its pixel-IoU difference to the (first) reference row.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.io import write_csv

LOG_PATH = Path("csv") / "version_0" / "metrics.csv"
EPOCH_TIME = "train/epoch_time_s"
PEAK_MEM = "train/peak_gpu_mem_mb"


def log_stats(run: Path) -> dict:
    """Mean epoch time and peak GPU memory from the run's Lightning CSV log (blank if absent)."""
    path = run / LOG_PATH
    log = pd.read_csv(path) if path.is_file() else pd.DataFrame()
    stats = {}
    for column, key, reduce in ((EPOCH_TIME, "epoch_time_s", "mean"), (PEAK_MEM, "peak_gpu_mem_mb", "max")):
        values = log[column].dropna() if column in log else pd.Series(dtype=float)
        stats[key] = float(getattr(values, reduce)()) if len(values) else ""
    return stats


def run_row(run: Path, eval_name: str) -> dict:
    """One table row for a run directory."""
    cfg = yaml.safe_load((run / "config_resolved.yaml").read_text())
    metrics = json.loads((run / eval_name / "metrics.json").read_text())
    row = {"experiment_name": cfg["experiment_name"], "model": cfg["model"]["name"], "smoke": bool(cfg["smoke"]),
           "label": metrics["label"], "split": metrics["split"], "n_samples": metrics["n_samples"]}
    for level in ("pixel", "patch"):
        for key in ("iou", "precision", "recall", "f1"):
            row[f"{level}_{key}"] = metrics[level]["with_or"][key]
        row[f"{level}_iou_without_or"] = metrics[level]["without_or"]["iou"]
    row["params"] = metrics["params"]
    return {**row, **log_stats(run), "mmu": metrics["mmu"]}


def build_table(runs: list[Path], eval_name: str, reference_model: str) -> list[dict]:
    """Rows for all runs, reference model first, with `delta_pixel_iou_vs_reference`."""
    rows = [run_row(run, eval_name) for run in runs]
    rows.sort(key=lambda r: r["model"] != reference_model)
    reference = next((r for r in rows if r["model"] == reference_model), None)
    for row in rows:
        row["is_reference"] = reference is not None and row is reference
        row["delta_pixel_iou_vs_reference"] = row["pixel_iou"] - reference["pixel_iou"] if reference else ""
    return rows


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-name", default="eval_test", help="subfolder written by src.evaluate")
    parser.add_argument("--reference-model", default="utae", help="model.name of the reference row (PRD 6)")
    args = parser.parse_args()
    write_csv(args.out, build_table(args.runs, args.eval_name, args.reference_model))


if __name__ == "__main__":
    main()
