"""SMOKE tests for scripts/plot_sample_walkthrough.py and src.evaluation.walkthrough on the synthetic fixture."""
import argparse
import json
from pathlib import Path

import pytest
import torch

from scripts import plot_sample_walkthrough as walkthrough_script
from src.config import load_config
from src.evaluation.common import load_model
from src.evaluation.walkthrough import changed_pixels, load_walkthrough, window_attention, window_keep
from src.losses.build_loss import build_loss
from src.models.build_model import build_model
from src.training.module import ChangeDetectionModule
from tests.fixtures import make_bradd_fixture

REPO = Path(__file__).resolve().parents[1]
TINY_UTAE = [
    "model.params.encoder_widths=[8, 8, 8, 16]", "model.params.decoder_widths=[8, 8, 8, 16]",
    "model.params.out_conv=[8, 2]", "model.params.d_model=32", "model.params.n_head=4",
]
FIGURES = ("01_input_timeseries", "02_temporal_profile", "03_labels", "04_pipeline_shapes")
MODEL_FIGURES = ("05_attention_over_dates", "06_prediction")


@pytest.fixture(scope="module", name="setup")
def fixture_setup(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path, dict]:
    """Tiny U-TAE run config on a SMOKE-marked fixture, a random-weight ChangeDetectionModule checkpoint, the plot config."""
    tmp = tmp_path_factory.mktemp("walkthrough")
    root = make_bradd_fixture(tmp / "bradd", n_per_split={"train": 2, "validation": 1, "test": 3})
    (root / "SMOKE.txt").write_text("SMOKE: synthetic test fixture")
    cfg = load_config(REPO / "configs" / "smoke" / "utae_ce.yaml", [
        f"data.root={root}", f"data.stats_path={tmp / 'stats.pt'}", f"output.runs_dir={tmp / 'runs'}",
        "data.num_workers=0", *TINY_UTAE,
    ])
    torch.manual_seed(0)
    module = ChangeDetectionModule(build_model(cfg["model"]), build_loss(cfg["loss"]), cfg["optim"])
    ckpt = tmp / "random.ckpt"
    torch.save({"state_dict": module.state_dict()}, ckpt)
    plot_cfg = load_config(REPO / "configs" / "eval" / "walkthrough.yaml", ["pick_min_changed_px=20", "dpi=60"])
    return cfg, ckpt, plot_cfg


def cli_args(out: Path, checkpoint: Path | None, index: int | None = None) -> argparse.Namespace:
    """The namespace main() would build for a test-split run."""
    return argparse.Namespace(split="test", index=index, file=None, pick=None,
                              checkpoint=str(checkpoint) if checkpoint else None, out=out)


def test_all_figures_and_json_with_checkpoint(setup: tuple[dict, Path, dict], tmp_path: Path) -> None:
    """With a checkpoint all six PNGs and walkthrough.json are written; IoUs and SMOKE label are recorded."""
    cfg, ckpt, plot_cfg = setup
    payload = walkthrough_script.run(cfg, cli_args(tmp_path, ckpt), plot_cfg)
    for name in FIGURES + MODEL_FIGURES:
        assert (tmp_path / f"{name}.png").stat().st_size > 0, name
    stored = json.loads((tmp_path / "walkthrough.json").read_text())
    assert stored["file"] == payload["file"] and stored["label"] == "SMOKE (synthetic data)"
    assert 0.0 <= stored["iou_with_or"] <= 1.0 and 0.0 <= stored["iou_without_or"] <= 1.0
    assert stored["T_window"] == len(stored["window_dates"]) == len(stored["attention_mean_per_date"])
    assert stored["input_shape"][0] == stored["T_window"] and stored["stage_shapes"]["out_conv"] == [2, 48, 48]
    assert len(stored["label_dates"]) == 2 and stored["alert_date"]


def test_without_checkpoint_only_first_four(setup: tuple[dict, Path, dict], tmp_path: Path) -> None:
    """Without a checkpoint figures 1-4 and the JSON are written, but no attention or prediction."""
    cfg, _, plot_cfg = setup
    payload = walkthrough_script.run(cfg, cli_args(tmp_path, None, index=1), plot_cfg)
    assert all((tmp_path / f"{name}.png").is_file() for name in FIGURES)
    assert not any((tmp_path / f"{name}.png").exists() for name in MODEL_FIGURES)
    assert payload["checkpoint"] is None and "iou_with_or" not in payload and payload["meta_row"] == 1


def test_pick_positive_has_change(setup: tuple[dict, Path, dict]) -> None:
    """The default pick is a sample with at least pick_min_changed_px pixels changing 0 -> 1."""
    cfg, _, plot_cfg = setup
    walk = load_walkthrough(cfg, "test", None, None, plot_cfg)
    assert changed_pixels(walk.labels, walk.interval) >= int(plot_cfg["pick_min_changed_px"])
    same = load_walkthrough(cfg, "test", None, walk.file, plot_cfg)
    assert same.row == walk.row


def test_attention_sums_to_one_per_head(setup: tuple[dict, Path, dict]) -> None:
    """L-TAE weights over the model's window dates sum to ~1 for every head."""
    cfg, ckpt, plot_cfg = setup
    model = load_model(cfg, ckpt)
    walk = load_walkthrough(cfg, "test", None, None, plot_cfg)
    att = window_attention(model, walk.batch, walk.interval)
    assert att.shape == (4, int(window_keep(model, walk.batch, walk.interval).sum()))
    assert torch.allclose(att.sum(dim=1), torch.ones(4), atol=1e-4)
