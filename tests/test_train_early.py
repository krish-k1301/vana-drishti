"""Phase 4 training: dated prefix-truncation fit (SMOKE), fixed validation cutoffs, fine-tune init."""
import json
from pathlib import Path

import pytest
import torch

from src import train
from src.config import load_config
from src.data.loaders import build_dataset
from src.models.build_model import build_model
from src.smoke_data import write_smoke_dataset
from src.training.module import load_model_weights
from src.training.validation_prefix import FixedCutoffPrefix, validation_loader
from tests.test_train_smoke import TINY_UTAE

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", name="dated_tiny")
def fixture_dated_tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """6/4/4-sample SMOKE dated dataset written through the src.smoke_data entry point."""
    counts = {"train": 6, "validation": 4, "test": 4}
    return write_smoke_dataset(tmp_path_factory.mktemp("dated") / "data", counts, 0, dated=True)


def early_config(anchor: str, root: Path, runs: Path, extra: tuple[str, ...] = ()) -> dict:
    """The shipped dated smoke config pointed at the tiny dataset, tiny U-TAE, 1 epoch, no workers."""
    return load_config(REPO / "configs" / "smoke" / f"early_{anchor}.yaml", [
        f"data.root={root}", f"data.stats_path={root / 'stats.pt'}", f"output.runs_dir={runs}",
        "trainer.max_epochs=1", "data.num_workers=0", *TINY_UTAE, *extra,
    ])


@pytest.mark.parametrize("anchor", ["deter", "burn"])
def test_early_configs_keep_architecture_and_causal_window(anchor: str) -> None:
    """Early configs: baseline focal recipe, dated data with the right anchor, trailing margin 0, inclusive end."""
    cfg = load_config(REPO / "configs" / "early" / f"utae_prefix_{anchor}.yaml")
    focal = load_config(REPO / "configs" / "baseline_utae_focal.yaml")
    assert cfg["model"]["params"] == focal["model"]["params"] and cfg["loss"] == focal["loss"]
    assert (cfg["model"]["error_days_after"], cfg["model"]["inclusive_end"]) == (0, True)
    assert cfg["data"]["dataset"] == "dated" and cfg["data"]["prefix_truncation"]["anchor"] == anchor
    assert "inspect" not in cfg["data"]  # whole data section replaced, nothing left over from bradd.yaml
    assert cfg["train"]["validation_prefix"] == "fixed_per_sample"


def test_fixed_validation_cutoffs_are_deterministic(dated_tiny: Path, tmp_path: Path) -> None:
    """Every read of a validation sample gives the same prefix; the data path with random cutoffs is refused."""
    cfg = train.data_config(early_config("deter", dated_tiny, tmp_path))
    loader = validation_loader(cfg, "fixed_per_sample")
    assert isinstance(loader.dataset, FixedCutoffPrefix)
    first, second = loader.dataset[1], loader.dataset[1]
    assert torch.equal(first["ImageDays"], second["ImageDays"]) and torch.equal(first["Targets"], second["Targets"])
    full = build_dataset(cfg, "validation")[1]
    assert int(first["ImageDays"].max()) <= int(full["ImageDays"].max())
    assert int(first["TargetDays"][-1]) == int(first["ImageDays"].max())
    assert validation_loader(cfg, "none") is None
    with pytest.raises(ValueError, match="phases"):
        bad = {**cfg, "prefix_truncation": {**cfg["prefix_truncation"], "phases": ["train", "validation"]}}
        validation_loader(bad, "fixed_per_sample")


def test_early_fit_and_fine_tune(dated_tiny: Path, tmp_path: Path) -> None:
    """A SMOKE prefix fit runs end to end; its checkpoint initialises a second run strictly."""
    cfg = early_config("burn", dated_tiny, tmp_path / "a")
    train.run(cfg)
    run_dir = tmp_path / "a" / cfg["experiment_name"]
    metrics = json.loads((run_dir / "metrics_test.json").read_text())
    assert metrics["label"] == "SMOKE" and "skipped_intervals" in metrics
    checkpoint = metrics["checkpoint"]
    model = build_model(cfg["model"])
    load_model_weights(model, checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    assert torch.equal(model.state_dict()["backbone.in_conv.conv.conv.0.weight"],
                       state["model.backbone.in_conv.conv.conv.0.weight"])
    tuned = early_config("burn", dated_tiny, tmp_path / "b", (f"train.init_checkpoint={checkpoint}",))
    train.run(tuned)
    log = (tmp_path / "b" / cfg["experiment_name"] / "run.log").read_text()
    assert f"initialised model weights from {checkpoint}" in log
    wrong = build_model({**cfg["model"], "params": {**cfg["model"]["params"], "d_model": 64}})
    with pytest.raises(RuntimeError):
        load_model_weights(wrong, checkpoint)
