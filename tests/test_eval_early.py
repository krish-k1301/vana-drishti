"""End-to-end SMOKE test of src.early_eval and scripts/plot_early.py on the synthetic dated fixture."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from src import early_eval
from src.config import load_config
from src.evaluation.prefix_inference import cutoff_days, prefix_from_config, two_label_prefix
from src.losses.build_loss import build_loss
from src.models.build_model import build_model
from src.training.module import ChangeDetectionModule
from tests.fixtures import make_bradd_fixture

REPO = Path(__file__).resolve().parents[1]
TINY_UTAE = [
    "model.params.encoder_widths=[8, 8, 8, 16]", "model.params.decoder_widths=[8, 8, 8, 16]",
    "model.params.out_conv=[8, 2]", "model.params.d_model=32", "model.params.n_head=4",
]


def plot_module():
    """Import scripts/plot_early.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_early", REPO / "scripts" / "plot_early.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OUTPUTS = ("curves.csv", "events.csv", "latency.csv", "recall.csv", "false_alarms.csv", "summary.json")


@pytest.fixture(scope="module", name="setup")
def fixture_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path, dict]:
    """Dated fixture with forest negatives in validation and test, a dated run config and a random-weight ckpt."""
    tmp = tmp_path_factory.mktemp("early")
    root = make_bradd_fixture(tmp / "dated", n_per_split={"train": 2, "validation": 4, "test": 4}, dated=True, seed=0)
    cfg = load_config(REPO / "configs" / "smoke" / "utae_ce.yaml", [f"output.runs_dir={tmp / 'runs'}", *TINY_UTAE])
    data = load_config(REPO / "configs" / "data" / "dated_amazon.yaml",
                       [f"root={root}", f"stats_path={tmp / 'stats.pt'}", "num_workers=0"])
    cfg.update(data=data, experiment_name="smoke_early")
    cfg["model"]["error_days_after"] = 0
    torch.manual_seed(0)
    module = ChangeDetectionModule(build_model(cfg["model"]), build_loss(cfg["loss"]), cfg["optim"])
    ckpt = tmp / "random.ckpt"
    torch.save({"state_dict": module.state_dict()}, ckpt)
    eval_cfg = load_config(REPO / "configs" / "eval" / "early.yaml", ["min_dates=28"])  # fewer cutoffs: fast on CPU
    return cfg, ckpt, eval_cfg


def test_prefix_inputs_are_causal(setup: tuple[dict, Path, dict]) -> None:
    """Every prefix passed to the model ends at its cutoff, and cutoffs respect min_dates."""
    cfg, _, eval_cfg = setup
    data_cfg = early_eval.eval_data_config(cfg)
    dataset = early_eval.build_dataset(data_cfg, "test")
    item = dataset[0]
    cutoffs = cutoff_days(item, cfg["model"]["error_days_before"], eval_cfg["min_dates"])
    assert len(cutoffs) > 0 and np.all(cutoffs > int(item["TargetDays"][0]))
    prefix = prefix_from_config(data_cfg)
    for cutoff in cutoffs[:3]:
        inputs = two_label_prefix(item, prefix, int(cutoff))
        assert int(inputs["ImageDays"].max()) == cutoff and inputs["TargetDays"].tolist()[1] == cutoff


def test_early_eval_end_to_end(setup: tuple[dict, Path, dict], tmp_path: Path, monkeypatch) -> None:
    """Outputs exist and are SMOKE-labelled; tau is chosen before any test data is loaded."""
    cfg, ckpt, eval_cfg = setup
    phases, seen_at_tau = [], []
    real_build, real_choose = early_eval.build_dataset, early_eval.choose_tau
    monkeypatch.setattr(early_eval, "build_dataset", lambda d, p: phases.append(p) or real_build(d, p))
    monkeypatch.setattr(early_eval, "choose_tau", lambda *a: seen_at_tau.extend(phases) or real_choose(*a))
    summary = early_eval.run(cfg, str(ckpt), eval_cfg, tmp_path)
    assert seen_at_tau == ["validation"] and summary["tau_split"] == "validation"
    for name in OUTPUTS:
        assert (tmp_path / name).is_file(), name
    assert summary["label"] == "SMOKE" and summary["n_events"] > 0
    latency = pd.read_csv(tmp_path / "latency.csv")
    assert set(latency["reference"]) == {"deter", "burn", "radd"} and (latency["label"] == "SMOKE").all()
    assert {"stage", "size_bin", "edge_interior"} <= set(latency["stratum_type"])
    recall = pd.read_csv(tmp_path / "recall.csv")
    assert set(recall["offset_days"]) == {0, 12, 24, 48, 90}
    alarms = pd.read_csv(tmp_path / "false_alarms.csv")
    assert list(alarms["split"]) == ["validation", "test"]
    val = alarms.iloc[0]
    assert val["rate_per_km2_month"] <= eval_cfg["false_alarms"]["budget_per_km2_month"]
    events = pd.read_csv(tmp_path / "events.csv")
    assert set(events["stage"]) <= {"exposed_soil", "vegetation_remaining", "burn_scar"}
    subprocess.run([sys.executable, "scripts/plot_early.py", "--early-dir", str(tmp_path)], cwd=REPO, check=True)
    for png in ("early_probability_vs_days.png", "early_latency_hist.png"):
        assert (tmp_path / png).stat().st_size > 0
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["mmu"].startswith("MMU") and plot_module().prefix_title(saved, "x").startswith("SMOKE x")


def test_early_eval_refuses_bradd_config(setup: tuple[dict, Path, dict]) -> None:
    """A BraDD (undated) run config is rejected."""
    cfg, _, _ = setup
    bradd = {**cfg, "data": {**cfg["data"], "dataset": "bradd"}}
    with pytest.raises(ValueError, match="dated"):
        early_eval.eval_data_config(bradd)
