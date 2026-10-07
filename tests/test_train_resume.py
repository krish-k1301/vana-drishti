"""Resume from `last.ckpt` (SMOKE): both launchers continue at the next epoch with state and CSV rows intact."""
import csv
import filecmp
import json
import logging
from pathlib import Path
from types import ModuleType

import pytest
import torch
import yaml

from src import reference_run, train
from src.data.stats import compute_train_stats
from src.smoke_data import write_smoke_dataset
from src.training.resume import LAST_CHECKPOINT, AppendingExperimentWriter, config_differences
from tests.test_train_smoke import tiny_config

STEPS_PER_EPOCH = 2  # 8 train samples, batch size 4


@pytest.fixture(scope="module", name="tiny_root")
def fixture_tiny_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """8/4/4-sample SMOKE dataset (as in test_train_smoke)."""
    counts = {"train": 8, "validation": 4, "test": 4}
    return write_smoke_dataset(tmp_path_factory.mktemp("tiny_resume") / "data", counts, 0, dated=False)


def session(launcher: ModuleType, config: str, root: Path, runs: Path, *overrides: str) -> dict:
    """Run one launcher session on the tiny dataset and return its resolved config."""
    cfg = tiny_config(config, root, runs)
    for override in overrides:
        key, value = override.split("=", 1)
        *parents, leaf = key.split(".")
        node = cfg
        for part in parents:
            node = node[part]
        node[leaf] = yaml.safe_load(value)
    launcher.run(cfg)
    return cfg


def csv_rows(run_dir: Path) -> list[dict]:
    """Rows of the run's Lightning metrics.csv."""
    with (run_dir / "csv" / "version_0" / "metrics.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def epoch_rows(rows: list[dict], column: str) -> dict[int, dict]:
    """Rows with a value in `column`, keyed by epoch (asserts one row per epoch)."""
    picked = [row for row in rows if row.get(column, "") != ""]
    by_epoch = {int(row["epoch"]): row for row in picked}
    assert len(by_epoch) == len(picked), f"duplicate epoch rows for {column}"
    return by_epoch


def checkpoint(run_dir: Path) -> dict:
    """The run's last.ckpt contents."""
    return torch.load(run_dir / LAST_CHECKPOINT, map_location="cpu", weights_only=False)


def optimizer_step(state: dict) -> int:
    """AdamW step count of the first parameter in a checkpoint."""
    return int(state["optimizer_states"][0]["state"][0]["step"])


def check_resumed(run_dir: Path, monitor: str, before: list[dict]) -> dict[int, dict]:
    """Shared assertions after a 1-epoch session plus a resumed 2-epoch session; returns monitor rows by epoch."""
    log = (run_dir / "run.log").read_text()
    assert "starting fresh at epoch 0" in log and "training continues at epoch 1" in log
    state = checkpoint(run_dir)
    assert state["epoch"] == 1 and state["global_step"] == 2 * STEPS_PER_EPOCH
    assert optimizer_step(state) == 2 * STEPS_PER_EPOCH
    rows = csv_rows(run_dir)
    monitored = epoch_rows(rows, monitor)
    assert sorted(monitored) == [0, 1]
    assert monitored[0] == epoch_rows(before, monitor)[0] | {k: "" for k in rows[0] if k not in before[0]}
    metrics = json.loads((run_dir / "metrics_test.json").read_text())
    assert metrics["epochs_trained"] == 2
    best = Path(metrics["checkpoint"])
    best_epoch = int(best.stem.removeprefix("best-epoch"))
    scores = {epoch: float(row[monitor]) for epoch, row in monitored.items()}
    assert best.is_file() and best_epoch == max(scores, key=lambda e: (scores[e], -e))
    return monitored


def test_our_pipeline_resumes(tiny_root: Path, tmp_path: Path) -> None:
    """src.train: epoch 1 follows the checkpointed epoch 0, optimizer/best state restored, CSV rows kept."""
    cfg = session(train, "utae_ce.yaml", tiny_root, tmp_path, "train.resume=true")
    run_dir = tmp_path / cfg["experiment_name"]
    assert checkpoint(run_dir)["epoch"] == 0 and optimizer_step(checkpoint(run_dir)) == STEPS_PER_EPOCH
    before = csv_rows(run_dir)
    session(train, "utae_ce.yaml", tiny_root, tmp_path, "train.resume=true", "trainer.max_epochs=2")
    check_resumed(run_dir, "validation/pixel_iou", before)
    assert sorted(epoch_rows(csv_rows(run_dir), "train/epoch_time_s")) == [0, 1]
    assert epoch_rows(csv_rows(run_dir), "test/pixel_iou")


def test_reference_launcher_resumes(tiny_root: Path, tmp_path: Path) -> None:
    """src.reference_run: same continuation, and StoppingScore/Best continues from the restored best score."""
    cfg = session(reference_run, "reference_utae_ce.yaml", tiny_root, tmp_path, "train.resume=true")
    run_dir = tmp_path / cfg["experiment_name"]
    before = csv_rows(run_dir)
    session(reference_run, "reference_utae_ce.yaml", tiny_root, tmp_path, "train.resume=true", "trainer.max_epochs=2")
    monitored = check_resumed(run_dir, "StoppingScore/Epoch", before)
    scores = [float(monitored[e]["StoppingScore/Epoch"]) for e in (0, 1)]
    assert float(monitored[1]["StoppingScore/Best"]) == pytest.approx(max(scores))
    assert "restored upstream StoppingScore/Best" in (run_dir / "run.log").read_text()


def test_resume_false_starts_fresh_and_mismatch_is_refused(tiny_root: Path, tmp_path: Path) -> None:
    """resume=false retrains from epoch 0 with a new CSV; a changed config refuses to resume."""
    cfg = session(train, "utae_ce.yaml", tiny_root, tmp_path, "train.resume=true")
    run_dir = tmp_path / cfg["experiment_name"]
    saved = (run_dir / "config_resolved.yaml").read_text()
    with pytest.raises(ValueError, match=r"refusing to resume.*optim\.lr.*experiment_name"):
        session(train, "utae_ce.yaml", tiny_root, tmp_path, "train.resume=true", "optim.lr=0.5")
    assert (run_dir / "config_resolved.yaml").read_text() == saved
    session(train, "utae_ce.yaml", tiny_root, tmp_path, "train.resume=false", "optim.lr=0.5")
    assert "starting fresh at epoch 0 (train.resume=false)" in (run_dir / "run.log").read_text()
    state = checkpoint(run_dir)
    assert state["epoch"] == 0 and optimizer_step(state) == STEPS_PER_EPOCH
    assert sorted(epoch_rows(csv_rows(run_dir), "validation/pixel_iou")) == [0]


def test_early_stopped_run_stays_stopped(tiny_root: Path, tmp_path: Path) -> None:
    """Rerunning a run that early-stopped (patience 0) tests again without training another epoch."""
    stop = ("train.resume=true", "trainer.max_epochs=5", "trainer.early_stopping.patience=0",
            "trainer.early_stopping.min_delta=10.0")
    cfg = session(train, "utae_ce.yaml", tiny_root, tmp_path, *stop)
    run_dir = tmp_path / cfg["experiment_name"]
    assert checkpoint(run_dir)["epoch"] == 1
    session(train, "utae_ce.yaml", tiny_root, tmp_path, *stop)
    assert checkpoint(run_dir)["epoch"] == 1
    assert sorted(epoch_rows(csv_rows(run_dir), "validation/pixel_iou")) == [0, 1]


def test_config_differences_allow_list() -> None:
    """Session-only keys may differ; anything else (including a missing key) is reported."""
    saved = {"train": {"resume": True, "init_checkpoint": None}, "data": {"num_workers": 4, "root": "a"}}
    current = {"train": {"resume": False, "init_checkpoint": None}, "data": {"num_workers": 0, "root": "b"},
               "extra": 1}
    assert config_differences(saved, current) == ["data.root", "extra"]


def test_csv_trimmed_to_checkpoint_then_appended(tmp_path: Path) -> None:
    """Rows after the checkpoint (epoch > 0, or epoch-less step >= global step 2) are dropped; new rows append."""
    log_dir = tmp_path / "version_0"
    log_dir.mkdir()
    old = "epoch,lr,step,val\n0,,1,0.5\n,0.1,1,\n,0.1,2,\n1,,2,0.9\n"
    (log_dir / "metrics.csv").write_text(old)
    writer = AppendingExperimentWriter(str(log_dir), keep_until=(0, 2))
    writer.log_metrics({"epoch": 1, "val": 0.7, "new": 1.0}, step=3)
    writer.save()
    with (log_dir / "metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(r["epoch"], r["step"], r["val"], r["new"]) for r in rows] == [
        ("0", "1", "0.5", ""), ("", "1", "", ""), ("1", "3", "0.7", "1.0")]


def test_reference_stats_copy_rule(tmp_path: Path) -> None:
    """close_stats.pt: copied from data.stats_path when that matches the split, computed otherwise, never replaced."""
    root = write_smoke_dataset(tmp_path / "data", {"train": 4, "validation": 2, "test": 2}, 0, dated=False)
    ours = root / "ours.pt"
    torch.save(compute_train_stats(root, "close_set"), ours)
    data_cfg = {"root": str(root), "stats_path": str(ours), "split_column": "close_set"}
    log = logging.getLogger("test_stats")
    reference_run.ensure_upstream_stats(data_cfg, "close", log)
    target = root / "close_stats.pt"
    assert filecmp.cmp(ours, target, shallow=False)
    torch.save({"mean": torch.zeros(2), "std": torch.ones(2), "min": torch.zeros(2), "max": torch.ones(2)}, target)
    kept = target.read_bytes()
    reference_run.ensure_upstream_stats(data_cfg, "close", log)
    assert target.read_bytes() == kept
    target.unlink()
    reference_run.ensure_upstream_stats({**data_cfg, "split_column": "other_set"}, "close", log)
    computed = torch.load(target, weights_only=False)
    expected = compute_train_stats(root, "close_set")
    assert all(torch.equal(computed[k], expected[k]) for k in expected)
