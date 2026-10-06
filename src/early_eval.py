"""Phase 5 early-detection evaluation on the dated set (PRD 3, 7 Phase 5, 9).

`python -m src.early_eval --config <run yaml> --checkpoint <ckpt> [--eval-config configs/eval/early.yaml]
 [--out DIR] [dotted.key=value ...]`

1. tau: chosen on VALIDATION stable-forest negatives at the configured false-alarm budget (never on test).
2. Every test event: sliding-prefix inference at each cutoff t_c over the training label-day grid ending at
   t_c (no image after t_c is ever passed in), per-interval change probabilities combined into a cumulative
   per-pixel probability (`interval_combiner`), per-event curves, detection = first cutoff >= tau.
3. Tables: latency vs the event date (named by `event_reference`: `deter` on the Amazon set) / burn / RADD,
   recall at `recall_reference` (one of those names) + offsets, split by DETER stage, size
   bin and edge vs interior; false alarms per km^2 per month on validation and test negatives.
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
from src.evaluation.early_records import EarlyCollector, event_mask
from src.evaluation.early_tables import latency_table, recall_table
from src.evaluation.false_alarm_eval import choose_tau, false_alarm_row, negative_maps, negative_rows
from src.evaluation.io import pixel_area_ha, write_csv, write_json
from src.evaluation.prefix_inference import cumulative_probabilities, prefix_from_config, valid_cutoffs
from src.metrics.strata import minimum_mapping_unit
from src.train import data_config

MODEL_WINDOW_KEYS = ("error_days_before", "error_days_after", "inclusive_end")


class PrefixScorer:
    """Callable row index -> (cutoffs [K], cumulative change probabilities [K, H, W]) for one dataset."""

    def __init__(self, model: torch.nn.Module, dataset: torch.utils.data.Dataset, data_cfg: dict,
                 eval_cfg: dict) -> None:
        """Bind the model, dataset, prefix grid and inference settings."""
        self.model, self.dataset = model, dataset
        self.prefix = prefix_from_config(data_cfg)
        self.min_dates = int(eval_cfg["min_dates"])
        self.batch_size = int(eval_cfg["cutoff_batch_size"])
        self.combiner = eval_cfg["interval_combiner"]

    def scores(self, item: dict) -> tuple[np.ndarray, np.ndarray]:
        """Cutoffs and probabilities for an already loaded item."""
        cutoffs = valid_cutoffs(self.model, item, self.prefix, self.min_dates)
        if len(cutoffs) == 0:
            return cutoffs, np.zeros((0, *item["Targets"].shape[-2:]), dtype=np.float32)
        return cutoffs, cumulative_probabilities(self.model, item, cutoffs, self.prefix, self.batch_size,
                                                 self.combiner)

    def __call__(self, row: int) -> tuple[np.ndarray, np.ndarray]:
        """Cutoffs and probabilities for dataset row `row`."""
        return self.scores(self.dataset[row])


def model_window(cfg: dict) -> dict:
    """Window settings of the run's model section; each key is required (no silent default)."""
    return {key: require(cfg, f"model.{key}") for key in MODEL_WINDOW_KEYS}


def eval_data_config(cfg: dict) -> dict:
    """Run data config with prefix truncation disabled in every phase (Phase 5 applies prefixes itself)."""
    data_cfg = copy.deepcopy(data_config(cfg))
    if require(data_cfg, "dataset") != "dated":
        raise ValueError("early evaluation needs data.dataset = dated")
    data_cfg["prefix_truncation"]["phases"] = []
    return data_cfg


def collect_events(scorer: PrefixScorer, collector: EarlyCollector, eval_cfg: dict, out: Path) -> dict:
    """Run every test sample with event pixels through the scorer; optionally save pixel probabilities."""
    meta = scorer.dataset.meta
    counts = {"n_events": 0, "n_events_without_cutoffs": 0}
    for row in meta.index:
        item = scorer.dataset[int(row)]
        if not event_mask(item).any():
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


def validation_tau(model: torch.nn.Module, data_cfg: dict, eval_cfg: dict) -> tuple[float, dict]:
    """tau at the false-alarm budget on VALIDATION negatives, and those negatives."""
    fa_cfg = eval_cfg["false_alarms"]
    validation = build_dataset(data_cfg, "validation")
    negatives = negative_maps(negative_rows(validation.meta, fa_cfg["negative_sampling_types"]),
                              PrefixScorer(model, validation, data_cfg, eval_cfg))
    return choose_tau(negatives, pixel_area_ha(float(eval_cfg["pixel_size_m"])), fa_cfg), negatives


def write_tables(out: Path, collector: EarlyCollector, alarms: list[dict], eval_cfg: dict, header: dict) -> None:
    """Write curves, events, latency, recall and false-alarm CSVs."""
    events, pixels = collector.events(), collector.pixels()
    write_csv(out / "curves.csv", collector.curve_rows, header)
    write_csv(out / "events.csv", events.to_dict("records"), header)
    write_csv(out / "latency.csv", latency_table(events, pixels, collector.references), header)
    write_csv(out / "recall.csv", recall_table(events, collector.event_curves, pixels,
                                               list(eval_cfg["recall_offsets_days"]), collector.tau,
                                               int(eval_cfg["persist_k"]), eval_cfg["recall_reference"]), header)
    write_csv(out / "false_alarms.csv", alarms, header)


def settings(cfg: dict, data_cfg: dict, eval_cfg: dict) -> dict:
    """Inference and detection settings recorded in summary.json."""
    return {"interval_grid": "prefix_at label days ending at t_c",
            "label_interval_days": data_cfg["prefix_truncation"]["label_interval_days"],
            "interval_combiner": eval_cfg["interval_combiner"],
            **{f"model_{k}": v for k, v in model_window(cfg).items()},
            **{k: eval_cfg[k] for k in ("min_dates", "event_curve", "persist_k", "event_reference",
                                        "recall_reference")},
            "mmu": minimum_mapping_unit(float(eval_cfg["pixel_size_m"]), int(eval_cfg["min_component_px"]))}


def evaluate_early(model: torch.nn.Module, cfg: dict, data_cfg: dict, eval_cfg: dict, out: Path,
                   provenance: dict, tau: float | None = None) -> dict:
    """Full Phase 5 evaluation on `data_cfg`'s test split; tau from validation unless given; returns the summary.

    `provenance` (source, checkpoint, and `tau_source` when tau is given) is copied into summary.json.
    """
    fields = settings(cfg, data_cfg, eval_cfg)
    val_neg = None
    if tau is None:
        tau, val_neg = validation_tau(model, data_cfg, eval_cfg)
        provenance = {**provenance, "tau_source": "validation"}
    fa_cfg, area = eval_cfg["false_alarms"], pixel_area_ha(float(eval_cfg["pixel_size_m"]))
    test = build_dataset(data_cfg, "test")
    scorer, collector = PrefixScorer(model, test, data_cfg, eval_cfg), EarlyCollector(eval_cfg, tau)
    counts = collect_events(scorer, collector, eval_cfg, out)
    test_neg = negative_maps(negative_rows(test.meta, fa_cfg["negative_sampling_types"]), scorer)
    splits = (("validation", val_neg), ("test", test_neg)) if val_neg is not None else (("test", test_neg),)
    alarms = [false_alarm_row(s, n, tau, area, fa_cfg) for s, n in splits]
    header = smoke_header(cfg)
    write_tables(out, collector, alarms, eval_cfg, header)
    summary = {**header, **provenance, "tau": tau, "false_alarms": alarms, **counts, **fields}
    write_json(out / "summary.json", summary)
    return summary


def run(cfg: dict, checkpoint: str, eval_cfg: dict, out: Path | None) -> dict:
    """Phase 5 on the run's own dated data; writes into `out` (default `<run dir>/early_eval`)."""
    model = load_model(cfg, checkpoint)
    return evaluate_early(model, cfg, eval_data_config(cfg), eval_cfg, out or run_dir(cfg) / "early_eval",
                          {"source": "src.early_eval", "checkpoint": str(checkpoint)})


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
