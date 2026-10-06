"""Tiny end-to-end fits (SMOKE) of our pipeline and of the upstream reference launcher."""
import json
from pathlib import Path

import pytest

from src import reference_run, train
from src.config import load_config
from src.smoke_data import write_smoke_dataset

REPO = Path(__file__).resolve().parents[1]
TINY_UTAE = [  # small widths keep the CPU fit fast; the architecture (3 downsamples) is unchanged
    "model.params.encoder_widths=[8, 8, 8, 16]", "model.params.decoder_widths=[8, 8, 8, 16]",
    "model.params.out_conv=[8, 2]", "model.params.d_model=32", "model.params.n_head=4",
]
RUN_FILES = ("metrics_test.json", "metrics_test.csv", "config_resolved.yaml", "run.log")


@pytest.fixture(scope="module", name="tiny_root")
def fixture_tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """8/4/4-sample SMOKE dataset written through the src.smoke_data entry point."""
    counts = {"train": 8, "validation": 4, "test": 4}
    return write_smoke_dataset(tmp_path_factory.mktemp("tiny") / "data", counts, 0, dated=False)


def tiny_config(config: str, root: Path, runs: Path) -> dict:
    """A shipped smoke config pointed at the tiny dataset, tiny U-TAE, 1 epoch, no workers."""
    return load_config(REPO / "configs" / "smoke" / config, [
        f"data.root={root}", f"data.stats_path={root / 'stats.pt'}", f"output.runs_dir={runs}",
        "trainer.max_epochs=1", "data.num_workers=0", *TINY_UTAE,
    ])


def check_outputs(run_dir: Path) -> dict:
    """Assert the run files exist and return the parsed metrics JSON."""
    for name in RUN_FILES:
        assert (run_dir / name).is_file(), name
    assert list((run_dir / "checkpoints").glob("*.ckpt"))
    assert (run_dir / "csv" / "version_0" / "metrics.csv").is_file()
    metrics = json.loads((run_dir / "metrics_test.json").read_text())
    assert metrics["label"] == "SMOKE" and metrics["smoke"] is True
    return metrics


def test_our_pipeline_end_to_end(tiny_root: Path, tmp_path: Path) -> None:
    """src.train fits, tests the best checkpoint and writes pixel/patch, with/without OR metrics."""
    cfg = tiny_config("utae_ce.yaml", tiny_root, tmp_path)
    train.run(cfg)
    metrics = check_outputs(tmp_path / cfg["experiment_name"])
    for level in ("pixel", "patch"):
        for variant in ("with_or", "without_or"):
            assert set(metrics[level][variant]) >= {"tp", "fp", "fn", "tn", "iou", "precision", "recall", "f1"}
    assert {"iou_delta", "n_forced_fp", "n_forced_tp"} <= set(metrics["or_rule"])
    log = (tmp_path / cfg["experiment_name"] / "run.log").read_text()
    assert "U-TAE stage down_3" in log and "(16, 6, 6)" in log
    header = (tmp_path / cfg["experiment_name"] / "csv" / "version_0" / "metrics.csv").read_text().splitlines()[0]
    assert "validation/pixel_iou" in header and "train/epoch_time_s" in header and "train/lr" in header


def test_reference_launcher_end_to_end(tiny_root: Path, tmp_path: Path) -> None:
    """src.reference_run trains upstream code (focal) and writes upstream whole-set test scores."""
    cfg = tiny_config("reference_utae_focal.yaml", tiny_root, tmp_path)
    reference_run.run(cfg)
    metrics = check_outputs(tmp_path / cfg["experiment_name"])
    assert set(metrics["pixel"]["with_or"]) == {"Accuracy", "Recall", "Precision", "IoU", "F1"}
    assert (tiny_root / "close_stats.pt").is_file()


def test_seed_single_source_and_upstream_guard(tiny_root: Path, tmp_path: Path) -> None:
    """data.seed is refused, and the reference launcher refuses settings upstream hard-codes."""
    cfg = tiny_config("utae_ce.yaml", tiny_root, tmp_path)
    with pytest.raises(ValueError, match="single source"):
        train.data_config({**cfg, "data": {**cfg["data"], "seed": 1}})
    bad = tiny_config("reference_utae_ce.yaml", tiny_root, tmp_path)
    bad["model"]["error_days_after"] = 0
    with pytest.raises(ValueError, match="hard-codes"):
        reference_run.run(bad)
