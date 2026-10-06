"""End-to-end SMOKE test of src.early_eval and scripts/plot_early.py on the synthetic dated fixture."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest
import torch

from src import early_eval
from src.config import load_config
from src.data.samples import load_sample
from src.evaluation.early_records import event_mask, references
from src.evaluation.prefix_inference import combine_intervals, prefix_from_config, prefix_inputs, valid_cutoffs
from src.losses.build_loss import build_loss
from src.models.build_model import build_model
from src.training.module import ChangeDetectionModule
from tests.fixtures import make_bradd_fixture

REPO = Path(__file__).resolve().parents[1]
TINY_UTAE = [
    "model.params.encoder_widths=[8, 8, 8, 16]", "model.params.decoder_widths=[8, 8, 8, 16]",
    "model.params.out_conv=[8, 2]", "model.params.d_model=32", "model.params.n_head=4",
]


def plot_module() -> ModuleType:
    """Import scripts/plot_early.py (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("plot_early", REPO / "scripts" / "plot_early.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OUTPUTS = ("curves.csv", "events.csv", "latency.csv", "recall.csv", "false_alarms.csv", "summary.json")


@pytest.fixture(scope="module", name="setup")
def fixture_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path, dict]:
    """Dated fixture, a dated run config and a random-weight checkpoint in ChangeDetectionModule layout."""
    tmp = tmp_path_factory.mktemp("early")
    root = make_bradd_fixture(tmp / "dated", n_per_split={"train": 2, "validation": 2, "test": 3}, dated=True, seed=0)
    cfg = load_config(REPO / "configs" / "smoke" / "utae_ce.yaml", [f"output.runs_dir={tmp / 'runs'}", *TINY_UTAE])
    data = load_config(REPO / "configs" / "data" / "dated_amazon.yaml",
                       [f"root={root}", f"stats_path={tmp / 'stats.pt'}", "num_workers=0"])
    cfg.update(data=data, experiment_name="smoke_early")
    cfg["model"].update(error_days_after=0, inclusive_end=True)
    torch.manual_seed(0)
    module = ChangeDetectionModule(build_model(cfg["model"]), build_loss(cfg["loss"]), cfg["optim"])
    ckpt = tmp / "random.ckpt"
    torch.save({"state_dict": module.state_dict()}, ckpt)
    # every fixture split has a non-event sample at position 1; SMOKE plumbing only, not a stable-forest claim
    negatives = "false_alarms.negative_sampling_types=[forest, oldDeforest, herbaceous]"
    overrides = [f"min_dates={few_cutoffs(root)}", negatives, "cutoff_batch_size=16"]
    return cfg, ckpt, load_config(REPO / "configs" / "eval" / "early.yaml", overrides)


def few_cutoffs(root: Path, n_cutoffs: int = 2) -> int:
    """min_dates leaving about `n_cutoffs` cutoffs for the shortest sample (keeps the CPU test fast)."""
    shortest = min(len(load_sample(path)["image_dates"]) for path in (root / "Samples").glob("*.pt"))
    return shortest - n_cutoffs + 1


def test_prefix_inputs_are_causal(setup: tuple[dict, Path, dict]) -> None:
    """Every prefix ends at its cutoff with the training label grid (every L days back from t_c)."""
    cfg, ckpt, eval_cfg = setup
    data_cfg = early_eval.eval_data_config(cfg)
    item = early_eval.build_dataset(data_cfg, "test")[0]
    model = early_eval.load_model(cfg, ckpt)
    prefix = prefix_from_config(data_cfg)
    cutoffs = valid_cutoffs(model, item, prefix, eval_cfg["min_dates"])
    assert len(cutoffs) > 0 and int((item["ImageDays"] <= cutoffs[0]).sum()) >= eval_cfg["min_dates"]
    step = data_cfg["prefix_truncation"]["label_interval_days"]
    for cutoff in cutoffs[:3]:
        inputs = prefix_inputs(item, prefix, int(cutoff))
        days = inputs["TargetDays"].tolist()
        assert int(inputs["ImageDays"].max()) == cutoff == days[-1] and len(days) >= 2
        assert all(b - a == step for a, b in zip(days[1:-1], days[2:]))


def test_combine_intervals_noisy_or_and_max() -> None:
    """noisy-OR = 1 - prod(1 - p); max = max p; invalid intervals contribute nothing."""
    probs = torch.tensor([0.5, 0.5, 0.9]).view(1, 3, 1, 1)
    valid = torch.tensor([[1.0, 1.0, 0.0]])
    assert float(combine_intervals(probs, valid, "noisy_or")) == pytest.approx(0.75)
    assert float(combine_intervals(probs, valid, "max")) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        combine_intervals(probs, valid, "mean")


def test_event_mask_excludes_prior_clearing() -> None:
    """Pixels already cleared at label[0] are not part of the event."""
    targets = torch.zeros((2, 2, 2), dtype=torch.long)
    targets[0, 0, 0] = 1
    item = {"EventMask": torch.ones((2, 2), dtype=torch.uint8), "Targets": targets}
    assert event_mask(item).tolist() == [[False, True], [True, True]]


def test_missing_inclusive_end_fails_clearly(setup: tuple[dict, Path, dict]) -> None:
    """The window settings are read from the run config; a missing key is an error, not a default."""
    cfg, _, _ = setup
    model_cfg = {k: v for k, v in cfg["model"].items() if k != "inclusive_end"}
    with pytest.raises(KeyError, match="inclusive_end"):
        early_eval.model_window({**cfg, "model": model_cfg})


def test_early_eval_end_to_end(setup: tuple[dict, Path, dict], tmp_path: Path, monkeypatch) -> None:
    """Outputs exist and are SMOKE-labelled; tau is chosen before any test data is loaded."""
    cfg, ckpt, eval_cfg = setup
    phases, seen_at_tau = [], []
    real_build, real_choose = early_eval.build_dataset, early_eval.choose_tau
    monkeypatch.setattr(early_eval, "build_dataset", lambda d, p: phases.append(p) or real_build(d, p))
    monkeypatch.setattr(early_eval, "choose_tau", lambda *a: seen_at_tau.extend(phases) or real_choose(*a))
    summary = early_eval.run(cfg, str(ckpt), eval_cfg, tmp_path)
    assert seen_at_tau == ["validation"] and summary["tau_source"] == "validation"
    assert summary["interval_combiner"] == "noisy_or" and summary["model_inclusive_end"] is True
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


def test_event_reference_is_required_and_checked() -> None:
    """The event-date reference name comes from the eval config; recall must use one of the reference names."""
    eval_cfg = load_config(REPO / "configs" / "eval" / "early.yaml")
    assert references(eval_cfg) == ("deter", "burn", "radd")
    with pytest.raises(KeyError, match="event_reference"):
        references({k: v for k, v in eval_cfg.items() if k != "event_reference"})
    with pytest.raises(ValueError, match="recall_reference"):
        references({**eval_cfg, "event_reference": "hansen_year_end"})
    with pytest.raises(ValueError, match="clashes"):
        references({**eval_cfg, "event_reference": "radd"})


def test_early_eval_refuses_bradd_config(setup: tuple[dict, Path, dict]) -> None:
    """A BraDD (undated) run config is rejected."""
    cfg, _, _ = setup
    bradd = {**cfg, "data": {**cfg["data"], "dataset": "bradd"}}
    with pytest.raises(ValueError, match="dated"):
        early_eval.eval_data_config(bradd)
