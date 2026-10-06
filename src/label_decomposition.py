"""Phase 6 label decomposition: Hansen label penalty = IoU(PRODES-trained) - IoU(Hansen-trained), both vs PRODES.

`python -m src.label_decomposition --prodes-run RUN_DIR --hansen-run RUN_DIR --out results/label_decomposition.csv
 [--eval-config configs/eval/label_decomposition.yaml]`

Reads only `<run>/<eval_name>/metrics.json` written by src.evaluate. Both runs must have been evaluated against
the reference label source (`label_source` recorded in metrics.json) and on the same data root and split;
otherwise it refuses. One row per (level, variant, stratum) with both scores and their differences.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.evaluation.io import write_csv

SCORES = ("iou", "f1", "precision", "recall")
SAME_FIELDS = ("data_root", "split", "label_source", "mmu")


def read_metrics(run: Path, eval_name: str) -> dict:
    """The src.evaluate metrics.json of one run."""
    return json.loads((run / eval_name / "metrics.json").read_text())


def check_comparable(prodes: dict, hansen: dict, reference_label_source: str) -> None:
    """Refuse pairs scored on different data, splits, MMUs or labels other than the reference labels."""
    different = [k for k in SAME_FIELDS if prodes.get(k) != hansen.get(k)]
    if different:
        raise ValueError(f"evaluations differ in {different}; both must be scored on the same PRODES test set")
    if prodes.get("label_source") != reference_label_source:
        raise ValueError(f"evaluations were scored against label_source={prodes.get('label_source')!r}, "
                         f"expected {reference_label_source!r}")


def score_sets(metrics: dict) -> dict[tuple[str, str, str, str], dict]:
    """(level, variant, stratum_type, stratum) -> scores, for headline and size-bin rows."""
    out = {(level, variant, "all", "all"): metrics[level][variant]
           for level in ("pixel", "patch") for variant in ("with_or", "without_or")}
    for row in metrics["strata"]:
        if row["stratum_type"] == "size_bin":
            out[(row["level"], row["variant"], row["stratum_type"], row["stratum"])] = row
    return out


def decomposition_rows(prodes: dict, hansen: dict) -> list[dict]:
    """One row per score set: PRODES-trained, Hansen-trained and the Hansen label penalty (difference)."""
    hansen_sets = score_sets(hansen)
    rows = []
    for key, p_scores in score_sets(prodes).items():
        h_scores = hansen_sets[key]
        row = dict(zip(("level", "variant", "stratum_type", "stratum"), key))
        for score in SCORES:
            row[f"{score}_prodes_trained"] = p_scores[score]
            row[f"{score}_hansen_trained"] = h_scores[score]
            row[f"{score}_hansen_penalty"] = p_scores[score] - h_scores[score]
        rows.append({**row, "mmu": prodes["mmu"]})
    return rows


def build(prodes_run: Path, hansen_run: Path, eval_cfg: dict) -> list[dict]:
    """Read both evaluations, check they are comparable, and return the table rows."""
    prodes = read_metrics(prodes_run, eval_cfg["eval_name"])
    hansen = read_metrics(hansen_run, eval_cfg["eval_name"])
    check_comparable(prodes, hansen, eval_cfg["reference_label_source"])
    smoke = bool(prodes["smoke"] or hansen["smoke"])
    label = "SMOKE" if smoke else ""
    names = {"prodes_experiment": prodes["experiment_name"], "hansen_experiment": hansen["experiment_name"]}
    return [{"label": label, "smoke": smoke, **names, **row} for row in decomposition_rows(prodes, hansen)]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prodes-run", type=Path, required=True)
    parser.add_argument("--hansen-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-config", default="configs/eval/label_decomposition.yaml")
    args = parser.parse_args()
    write_csv(args.out, build(args.prodes_run, args.hansen_run, load_config(args.eval_config)))


if __name__ == "__main__":
    main()
