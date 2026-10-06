"""Evaluate a trained checkpoint on one split (PRD 7 Phases 1, 2, 6; metrics PRD 9).

`python -m src.evaluate --config <run yaml> --checkpoint <ckpt> --split test [--eval-config configs/eval/evaluate.yaml]
 [--temporal-subsample mode=last_only num_dates=1] [--out DIR] [dotted.key=value ...]`

Writes `metrics.json` and `metrics.csv` to `--out` (default `<runs_dir>/<experiment_name>/eval_<split>[_ts-...]`):
pixel and patch IoU/P/R/F1 with and without the `label[0] OR prediction` rule, OR-rule counts, per size bin,
edge/interior/outer ring, parameter count and the minimum mapping unit. SMOKE propagates from the run config.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import load_config, require
from src.data.bradd_dataset import LABEL_KEYS
from src.data.collate import collate_batch
from src.data.loaders import build_dataset
from src.evaluation.common import load_model, run_dir, smoke_header
from src.evaluation.io import write_csv, write_json
from src.evaluation.strata_eval import StrataMeters
from src.metrics.strata import minimum_mapping_unit
from src.models.inspection import count_parameters
from src.train import data_config
from src.training.meters import PhaseMeters, score_rows, select_intervals


def parse_subsample(items: list[str] | None) -> dict | None:
    """Turn ['mode=uniform', 'num_dates=5'] into a test-time temporal subsample spec (Phase 2 ablation)."""
    if not items:
        return None
    spec = dict(item.split("=", 1) for item in items)
    if set(spec) - {"mode", "num_dates"} or "mode" not in spec:
        raise ValueError(f"--temporal-subsample takes mode=... [num_dates=...], got {items}")
    return {"mode": spec["mode"], "num_dates": int(spec.get("num_dates", -1)), "at_test": True}


def eval_loader(data_cfg: dict, split: str, eval_cfg: dict) -> DataLoader:
    """Unshuffled DataLoader over `split` of the given data config."""
    dataset = build_dataset(data_cfg, split)
    return DataLoader(dataset, batch_size=int(eval_cfg["batch_size"]), shuffle=False,
                      num_workers=int(eval_cfg["num_workers"]), collate_fn=collate_batch)


def run_data_config(cfg: dict, subsample: dict | None) -> dict:
    """The run's data config, with the optional test-time temporal subsample."""
    data_cfg = data_config(cfg)
    if subsample is not None:
        data_cfg["temporal_subsample"] = subsample
    return data_cfg


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, eval_cfg: dict) -> tuple[dict, list[dict]]:
    """Run the model over the loader; return headline results and stratified rows.

    Intervals the model marks invalid (padded label days, no image in the window) are dropped and counted.
    """
    meters, strata = PhaseMeters(), StrataMeters(eval_cfg)
    for batch in loader:
        args = (batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])
        valid = model.interval_mask(*args[1:])
        logits, targets = select_intervals(model(*args), batch["Targets"], valid)
        meters.update(logits, targets, None, skipped=int((~valid).sum()))
        strata.update(logits, targets)
    results = meters.compute()
    results.pop("loss")
    return results, strata.rows()


def label_source(dataset: torch.utils.data.Dataset) -> str:
    """Label source ('prodes' / 'hansen') the dataset actually scores against."""
    return {key: source for source, key in LABEL_KEYS.items()}[dataset.label_key]


def default_out(cfg: dict, split: str, subsample: dict | None) -> Path:
    """`<run dir>/eval_<split>` plus a temporal-subsample suffix when one is applied."""
    suffix = f"_ts-{subsample['mode']}-{subsample['num_dates']}" if subsample else ""
    return run_dir(cfg) / f"eval_{split}{suffix}"


def run(cfg: dict, checkpoint: str, split: str, eval_cfg: dict, subsample: dict | None, out: Path | None) -> dict:
    """Evaluate and write metrics.json / metrics.csv; return the JSON payload."""
    model = load_model(cfg, checkpoint)
    loader = eval_loader(run_data_config(cfg, subsample), split, eval_cfg)
    results, strata_rows = evaluate(model, loader, eval_cfg)
    mmu = minimum_mapping_unit(float(eval_cfg["pixel_size_m"]), int(eval_cfg["min_component_px"]))
    header = smoke_header(cfg)
    payload = {**header, "source": "src.evaluate", "checkpoint": str(checkpoint), "split": split,
               "model_name": require(cfg, "model.name"), "n_samples": len(loader.dataset),
               "data_root": str(require(cfg, "data.root")), "label_source": label_source(loader.dataset),
               "temporal_subsample": subsample, "params": count_parameters(model), "mmu": mmu,
               **results, "strata": strata_rows}
    out = out or default_out(cfg, split, subsample)
    write_json(out / "metrics.json", payload)
    rows = [{"stratum_type": "all", "stratum": "all", **row} for row in score_rows(results)] + strata_rows
    write_csv(out / "metrics.csv", [{**row, "mmu": mmu} for row in rows], header)
    return payload


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--eval-config", default="configs/eval/evaluate.yaml")
    parser.add_argument("--temporal-subsample", nargs="+", metavar="KEY=VALUE")
    parser.add_argument("--out", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    cfg = load_config(args.config, args.overrides)
    run(cfg, args.checkpoint, args.split, load_config(args.eval_config),
        parse_subsample(args.temporal_subsample), args.out)


if __name__ == "__main__":
    main()
