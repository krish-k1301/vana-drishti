"""Phase 6 cross-biome transfer: Amazon-trained weights applied unchanged to a Congo or Borneo dated set.

`python -m src.cross_biome --config <Amazon run yaml> --checkpoint <ckpt> [--eval-config configs/eval/cross_biome.yaml]
 [--tau-from <Amazon early_eval/summary.json>] [--out DIR] [--eval-overrides target_data.root=... ...]`

1. Segmentation on the target test split against its stored (Hansen) labels: pixel/patch IoU/F1 and IoU/F1 per
   size bin, edge/interior (`segmentation.json`, `segmentation.csv`).
2. Early detection with the Phase 5 machinery (`early/`): latency vs RADD and vs the event date, which is
   named `hansen_year_end` (eval-config `event_reference`; 31 Dec of the Hansen loss year, never DETER); burn is
   absent (reported as no-reference); recall at RADD + offsets; false alarms on target negatives.
   tau comes from the Amazon early evaluation (`--tau-from`, fully unchanged transfer) or, without it, from the
   target's validation negatives at the configured budget (recorded as `tau_source`).
Normalisation keeps the run's Amazon stats file; it must exist. SMOKE propagates from the run config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.config import load_config, require
from src.early_eval import eval_data_config, evaluate_early
from src.evaluate import evaluate, eval_loader
from src.evaluation.common import load_model, run_dir, smoke_header
from src.evaluation.io import write_csv, write_json
from src.metrics.strata import minimum_mapping_unit
from src.training.meters import score_rows


def target_data_config(cfg: dict, eval_cfg: dict) -> dict:
    """The run's dated data config pointed at the target biome, keeping the Amazon normalisation stats."""
    data_cfg = eval_data_config(cfg)
    data_cfg.update(require(eval_cfg, "target_data"))
    stats = data_cfg.get("stats_path")
    if data_cfg.get("normalization") == "zscore" and (stats is None or not Path(stats).is_file()):
        raise FileNotFoundError(f"Amazon stats file {stats} is missing; transfer must reuse the source normalisation")
    return data_cfg


def segmentation(model: torch.nn.Module, data_cfg: dict, eval_cfg: dict, out: Path, header: dict) -> dict:
    """Score the target test split against its stored labels; write JSON and CSV; return headline results."""
    results, strata_rows = evaluate(model, eval_loader(data_cfg, "test", eval_cfg), eval_cfg)
    mmu = minimum_mapping_unit(float(eval_cfg["pixel_size_m"]), int(eval_cfg["min_component_px"]))
    write_json(out / "segmentation.json", {**header, "mmu": mmu, **results, "strata": strata_rows})
    rows = [{"stratum_type": "all", "stratum": "all", **row} for row in score_rows(results)] + strata_rows
    write_csv(out / "segmentation.csv", [{**row, "mmu": mmu} for row in rows], header)
    return results


def run(cfg: dict, checkpoint: str, eval_cfg: dict, tau_from: Path | None, out: Path | None) -> dict:
    """Run both parts and write `summary.json`; return it."""
    out = out or run_dir(cfg) / f"cross_biome_{Path(require(eval_cfg, 'target_data.root')).name}"
    model, header = load_model(cfg, checkpoint), smoke_header(cfg)
    data_cfg = target_data_config(cfg, eval_cfg)
    results = segmentation(model, data_cfg, eval_cfg, out, header)
    provenance = {"source": "src.cross_biome", "checkpoint": str(checkpoint), "target_root": str(data_cfg["root"])}
    tau = None
    if tau_from is not None:
        tau = float(json.loads(Path(tau_from).read_text())["tau"])
        provenance["tau_source"] = str(tau_from)
    early = evaluate_early(model, cfg, data_cfg, eval_cfg, out / "early", provenance, tau)
    summary = {**header, **provenance, "stats_path": data_cfg.get("stats_path"), "tau": early["tau"],
               "tau_source": early["tau_source"], "pixel": results["pixel"], "patch": results["patch"],
               "mmu": early["mmu"], "early_summary": str(out / "early" / "summary.json")}
    write_json(out / "summary.json", summary)
    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Amazon run config used for training")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-config", default="configs/eval/cross_biome.yaml")
    parser.add_argument("--tau-from", type=Path, help="Amazon early_eval summary.json whose tau is reused")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--eval-overrides", nargs="*", default=[], help="dotted.key=value for the eval config")
    args = parser.parse_args()
    eval_cfg = load_config(args.eval_config, args.eval_overrides)
    run(load_config(args.config), args.checkpoint, eval_cfg, args.tau_from, args.out)


if __name__ == "__main__":
    main()
