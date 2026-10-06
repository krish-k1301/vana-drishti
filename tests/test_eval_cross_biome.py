"""SMOKE tests for src.cross_biome (Phase 6 transfer) and src.label_decomposition."""
import csv
import json
import shutil
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
import torch

from src import cross_biome, label_decomposition
from src.config import load_config
from src.data.samples import load_sample
from src.data.stats import load_or_compute_stats
from tests.test_eval_early import fixture_setup

REPO = Path(__file__).resolve().parents[1]
SHARED_FIXTURES = (fixture_setup,)  # module fixture "setup" from the early-eval tests
BRAZIL_ONLY_KEYS = ("event_date", "deter_class", "burn_month", "prodes_year")


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    """Run this module's tiny CPU models on one thread (much faster than oversubscribed BLAS threads)."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def as_target_biome(source: Path, target: Path) -> Path:
    """Copy a dated fixture, keep one test event and one test negative (fast on CPU), and strip everything
    Brazil-only (DETER date/class, burn), keeping RADD dates and event masks."""
    shutil.copytree(source, target)
    meta = pd.read_csv(target / "meta.csv", index_col=0, keep_default_na=False)
    test = meta[meta["dated_set"] == "test"]
    keep = [test.index[test["deter_class"] != ""][0], test.index[test["deter_class"] == ""][0]]  # 1 event, 1 negative
    meta = meta[(meta["dated_set"] != "test") | meta.index.isin(keep)].copy()
    meta["deter_class"], meta["event_date"] = "", ""
    meta.to_csv(target / "meta.csv")
    for path in (target / "Samples").glob("*.pt"):
        sample = load_sample(path)
        torch.save({k: v for k, v in sample.items() if k not in BRAZIL_ONLY_KEYS}, path)
    return target


def target_eval_cfg(setup_eval: dict, target: Path) -> dict:
    """The shipped cross-biome config pointed at the target copy, with the fast test settings of `setup`."""
    overrides = [f"target_data.root={target}", f"min_dates={setup_eval['min_dates']}",
                 f"cutoff_batch_size={setup_eval['cutoff_batch_size']}",
                 f"false_alarms.negative_sampling_types={setup_eval['false_alarms']['negative_sampling_types']}"]
    return load_config(REPO / "configs" / "eval" / "cross_biome.yaml", overrides)


def test_cross_biome_end_to_end(setup: tuple[dict, Path, dict], tmp_path: Path) -> None:
    """Amazon weights + Amazon stats on a DETER-free target: size-bin scores and latency vs RADD."""
    cfg, ckpt, early_cfg = setup
    data = cfg["data"]
    load_or_compute_stats(data["root"], data["split_column"], data["stats_path"])
    target = as_target_biome(Path(data["root"]), tmp_path / "congo")
    amazon_summary = tmp_path / "amazon_summary.json"
    amazon_summary.write_text(json.dumps({"tau": 0.5}))
    summary = cross_biome.run(cfg, str(ckpt), target_eval_cfg(early_cfg, target), amazon_summary, tmp_path / "out")
    assert summary["label"] == "SMOKE" and summary["tau"] == 0.5 and summary["tau_source"] == str(amazon_summary)
    assert summary["stats_path"] == data["stats_path"]
    rows = list(csv.DictReader((tmp_path / "out" / "segmentation.csv").open()))
    assert any(r["stratum_type"] == "size_bin" and r["f1"] for r in rows)
    latency = pd.read_csv(tmp_path / "out" / "early" / "latency.csv")
    overall = latency[(latency["level"] == "event") & (latency["stratum_type"] == "all")].set_index("reference")
    assert overall.loc["deter", "n"] == 0 and overall.loc["burn", "n"] == 0 and overall.loc["radd", "n"] > 0
    assert overall.loc["deter", "n_no_reference"] == overall.loc["radd", "n"]
    recall = pd.read_csv(tmp_path / "out" / "early" / "recall.csv")
    assert set(recall["reference"]) == {"radd"}


def test_cross_biome_requires_source_stats(setup: tuple[dict, Path, dict], tmp_path: Path) -> None:
    """Transfer refuses to run when the Amazon normalisation stats are missing (no silent target stats)."""
    cfg, _, early_cfg = setup
    missing = {**cfg, "data": {**cfg["data"], "stats_path": str(tmp_path / "absent.pt")}}
    with pytest.raises(FileNotFoundError, match="normalisation"):
        cross_biome.target_data_config(missing, target_eval_cfg(early_cfg, tmp_path))


def write_metrics(run: Path, iou: float, smoke: bool = True, label_source: str = "prodes") -> Path:
    """A minimal src.evaluate metrics.json with one size-bin row."""
    scores = {"iou": iou, "f1": iou + 0.1, "precision": 0.5, "recall": 0.5}
    level = {"with_or": scores, "without_or": scores}
    payload = {"smoke": smoke, "experiment_name": run.name, "data_root": "/d", "split": "test",
               "label_source": label_source, "mmu": "MMU 0.01 ha", "pixel": level, "patch": level,
               "strata": [{"level": "pixel", "variant": "with_or", "stratum_type": "size_bin",
                           "stratum": "<0.5 ha", **scores},
                          {"level": "pixel", "variant": "with_or", "stratum_type": "edge_interior",
                           "stratum": "edge", **scores}]}
    (run / "eval_test").mkdir(parents=True)
    (run / "eval_test" / "metrics.json").write_text(json.dumps(payload))
    return run


def test_label_decomposition_penalty(tmp_path: Path) -> None:
    """Penalty = PRODES-trained minus Hansen-trained, per headline and size-bin row; mismatches refused."""
    eval_cfg = load_config(REPO / "configs" / "eval" / "label_decomposition.yaml")
    prodes, hansen = write_metrics(tmp_path / "prodes", 0.5), write_metrics(tmp_path / "hansen", 0.4)
    rows = label_decomposition.build(prodes, hansen, eval_cfg)
    assert len(rows) == 5 and all(r["label"] == "SMOKE" for r in rows)
    assert all(r["iou_hansen_penalty"] == pytest.approx(0.1) for r in rows)
    assert {r["stratum"] for r in rows} == {"all", "<0.5 ha"}
    other = write_metrics(tmp_path / "hansen_vs_hansen", 0.4, label_source="hansen")
    with pytest.raises(ValueError, match="label_source"):
        label_decomposition.build(prodes, other, eval_cfg)
