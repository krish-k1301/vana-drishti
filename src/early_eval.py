"""Phase 5 early-detection evaluation on the dated set (PRD 3, 7 Phase 5, 9).

`python -m src.early_eval --config <run yaml> --checkpoint <ckpt> [--eval-config configs/eval/early.yaml]
 [--out DIR] [dotted.key=value ...]`

1. tau: chosen on VALIDATION stable-forest negatives at the configured false-alarm budget (never on test).
2. Every test event: sliding-prefix inference at each cutoff t_c (no image after t_c is ever passed in),
   per-pixel and per-event probability curves, detection = first cutoff >= tau (optionally persistent).
3. Tables: latency vs DETER / burn / RADD, recall at DETER + offsets, split by DETER stage, size bin and
   edge vs interior; false alarms per km^2 per month on validation and test negatives.
Outputs in `--out` (default `<run dir>/early_eval`): curves.csv, events.csv, latency.csv, recall.csv,
false_alarms.csv, summary.json (SMOKE propagates from the run config).
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from src.config import load_config, require
from src.data.loaders import build_dataset
from src.evaluation.common import load_model, run_dir, smoke_header
from src.evaluation.early_records import EarlyCollector
from src.evaluation.early_tables import latency_table, recall_table
from src.evaluation.false_alarm_eval import choose_tau, false_alarm_row, negative_maps, negative_rows
from src.evaluation.io import pixel_area_ha, write_csv, write_json
from src.evaluation.prefix_inference import cutoff_days, prefix_from_config, prefix_probabilities
from src.metrics.strata import minimum_mapping_unit
from src.train import data_config


class PrefixScorer:
    """Callable row index -> (cutoffs [K], change probabilities [K, H, W]) for one dataset."""

    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, cfg: dict, eval_cfg: dict) -> None:
        """Bind the model, dataset, window margin and prefix settings."""
        self.model, self.dataset = model, dataset
        self.prefix = prefix_from_config(eval_data_config(cfg))
        self.before = int(require(cfg, "model.error_days_before"))
        self.min_dates = int(eval_cfg["min_dates"])
        self.batch_size = int(eval_cfg["cutoff_batch_size"])

    def scores(self, item: dict) -> tuple[np.ndarray, np.ndarray]:
        """Cutoffs and probabilities for an already loaded item."""
        cutoffs = cutoff_days(item, self.before, self.min_dates)
        if len(cutoffs) == 0:
            return cutoffs, np.zeros((0, *item["Targets"].shape[-2:]), dtype=np.float32)
        return cutoffs, prefix_probabilities(self.model, item, cutoffs, self.prefix, self.batch_size)

    def __call__(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """Cutoffs and probabilities for dataset row `row`."""
        return self.scores(self.dataset[row])


def eval_data_config(cfg: dict) -> dict:
    """Run data config with prefix truncation disabled in every phase (Phase 5 applies prefixes itself)."""
    data_cfg = copy.deepcopy(data_config(cfg))
    if require(data_cfg, "dataset") != "dated":
        raise ValueError("early evaluation needs data.dataset = dated")
    data_cfg["prefix_truncation"]["phases"] = []
    return data_cfg


def collect_events(scorer: PrefixScorer, collector: EarlyCollector, eval_cfg: dict, out: Path) -> dict:
    """Run every test event through the scorer into the collector; optionally save pixel probabilities."""
    meta = scorer.dataset.meta
    counts = {"n_events": 0, "n_events_without_cutoffs": 0}
    for row in meta.index[meta["deter_class"].astype(str) != ""]:
        item = scorer.dataset[int(row)]
        if int(item["EventDay"]) < 0 or not bool(item["EventMask"].any()):
            continue
        cutoffs, probs = scorer.scores(item)
        if len(cutoffs) == 0:
            counts["n_events_without_cutoffs"] += 1
            continue
        counts["n_events"] += 1
        collector.add(item, meta.loc[row], cutoffs, probs)
        if eval_cfg["save_pixel_probs"]:
            (out / "pixel_probs").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out / "pixel_probs" / f"{Path(meta.loc[row, 'file']).stem}.npz",
                                cutoffs=cutoffs, probs=probs.astype(np.float16))
    return counts


def run(cfg: dict, checkpoint: str, eval_cfg: dict, out: Path | None) -> dict:
    """Choose tau on validation, evaluate test events and negatives, write the tables; return the summary."""
    out = out or run_dir(cfg) / "early_eval"
    model, data_cfg = load_model(cfg, checkpoint), eval_data_config(cfg)
    fa_cfg, area = eval_cfg["false_alarms"], pixel_area_ha(float(eval_cfg["pixel_size_m"]))
    validation = build_dataset(data_cfg, "validation")
    val_neg = negative_maps(negative_rows(validation.meta, fa_cfg["negative_sampling_types"]),
                            PrefixScorer(model, validation, cfg, eval_cfg))
    tau = choose_tau(val_neg, area, fa_cfg)
    test = build_dataset(data_cfg, "test")
    scorer, collector = PrefixScorer(model, test, cfg, eval_cfg), EarlyCollector(eval_cfg, tau)
    counts = collect_events(scorer, collector, eval_cfg, out)
    test_neg = negative_maps(negative_rows(test.meta, fa_cfg["negative_sampling_types"]), scorer)
    events, pixels = collector.events(), collector.pixels()
    header = smoke_header(cfg)
    offsets = list(eval_cfg["recall_offsets_days"])
    alarms = [false_alarm_row(s, n, tau, area, fa_cfg) for s, n in (("validation", val_neg), ("test", test_neg))]
    write_csv(out / "curves.csv", collector.curve_rows, header)
    write_csv(out / "events.csv", events.to_dict("records"), header)
    write_csv(out / "latency.csv", latency_table(events, pixels), header)
    write_csv(out / "recall.csv", recall_table(events, collector.event_curves, pixels, offsets, tau,
                                               int(eval_cfg["persist_k"])), header)
    write_csv(out / "false_alarms.csv", alarms, header)
    summary = {**header, "source": "src.early_eval", "checkpoint": str(checkpoint), "tau": tau,
               "tau_split": "validation", "false_alarms": alarms, **counts,
               "interval": "[first label day, t_c]", "trailing_margin_days": require(cfg, "model.error_days_after"),
               "event_curve": eval_cfg["event_curve"], "persist_k": eval_cfg["persist_k"],
               "mmu": minimum_mapping_unit(float(eval_cfg["pixel_size_m"]), int(eval_cfg["min_component_px"]))}
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="run config used for training (dated data)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-config", default="configs/eval/early.yaml")
    parser.add_argument("--out", type=Path)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides of the run config")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides), args.checkpoint, load_config(args.eval_config), args.out)


if __name__ == "__main__":
    main()
