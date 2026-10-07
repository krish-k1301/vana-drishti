"""Resume-from-checkpoint support for `src.train` and `src.reference_run` (Colab sessions disconnect).

With `train.resume: true` and `<run_dir>/checkpoints/last.ckpt` present, a launcher calls
`trainer.fit(..., ckpt_path=last.ckpt)`: Lightning restores weights, optimizer, LR scheduler, EarlyStopping and
ModelCheckpoint (best path/score) state and continues at the next epoch in the same run directory.

Safety: the run dir's `config_resolved.yaml` must equal the current resolved config except for the keys in
`RESUME_ALLOWED_DIFFS` (settings that change how a session runs, not what is trained).

CSV continuity: Lightning's CSVLogger deletes an existing `csv/version_0/metrics.csv` when it starts. On resume
`AppendingCSVLogger` instead keeps the rows logged up to the checkpoint (epoch <= checkpoint epoch and
step < checkpoint global step: epoch-end rows carry step = global step - 1, per-step rows such as the LR monitor's
carry no epoch; later rows belong to an epoch that is re-run) and appends the new ones, so every reader of that
file sees each epoch once. Not bit-identical to an uninterrupted run: data-order RNG is not checkpointed.
"""
import csv
from pathlib import Path

import torch
import yaml
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.csv_logs import ExperimentWriter
from lightning.pytorch.loggers.logger import rank_zero_experiment

from src.config import require

LAST_CHECKPOINT = Path("checkpoints") / "last.ckpt"
RESUME_ALLOWED_DIFFS = (
    "train.resume",                  # the switch itself
    "data.num_workers",              # CPU count differs between Colab sessions
    "trainer.enable_progress_bar",   # display only
    "trainer.max_epochs",            # extend training of an unfinished/finished run
    "trainer.max_time",              # per-session wall-clock budget
    "output.runs_dir",               # run dir is the same folder; only the mount spelling may differ
)


def flatten(node: object, prefix: str = "") -> dict:
    """Dotted-key view of a nested config (lists and scalars are leaves)."""
    if not isinstance(node, dict):
        return {prefix: node}
    if not node and prefix:
        return {prefix: node}
    out = {}
    for key, value in node.items():
        out.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    return out


def config_differences(saved: dict, current: dict) -> list[str]:
    """Dotted keys whose values differ between two configs, ignoring `RESUME_ALLOWED_DIFFS`."""
    old = flatten(saved)
    new = flatten(yaml.safe_load(yaml.safe_dump(current, sort_keys=False)))
    keys = sorted(set(old) | set(new))
    return [k for k in keys if old.get(k, "<missing>") != new.get(k, "<missing>") and k not in RESUME_ALLOWED_DIFFS]


def check_resume_config(run_dir: Path, cfg: dict) -> None:
    """Raise if the run dir's saved config differs from `cfg` beyond the allow-list."""
    path = run_dir / "config_resolved.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"cannot resume {run_dir}: {path.name} is missing")
    diffs = config_differences(yaml.safe_load(path.read_text()), cfg)
    if diffs:
        raise ValueError(
            f"refusing to resume {run_dir}: config differs from the saved {path.name} in {diffs} "
            f"(only {list(RESUME_ALLOWED_DIFFS)} may change between sessions). Change `experiment_name` "
            f"to start a new run, or delete the run directory to restart this one.")


def resume_checkpoint(cfg: dict, run_dir: Path) -> Path | None:
    """`last.ckpt` to resume from when `train.resume` is true and it exists (config checked); else None."""
    resume = require(cfg, "train.resume")
    if not isinstance(resume, bool):
        raise ValueError(f"train.resume must be true or false, got {resume!r}")
    path = run_dir / LAST_CHECKPOINT
    if not resume or not path.is_file():
        return None
    check_resume_config(run_dir, cfg)
    return path


def checkpoint_state(path: Path) -> dict:
    """The Lightning checkpoint dict at `path` (CPU)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def read_rows(path: Path) -> tuple[list[str], list[dict]]:
    """Header and rows of a CSV file."""
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def row_before(row: dict, epoch: int, step: int) -> bool:
    """True when a metrics.csv row was logged before the checkpoint (`epoch`, global `step`) was saved."""
    row_epoch, row_step = row.get("epoch", ""), row.get("step", "")
    return (row_epoch == "" or int(float(row_epoch)) <= epoch) and (row_step == "" or int(float(row_step)) < step)


class AppendingExperimentWriter(ExperimentWriter):
    """ExperimentWriter that keeps the rows of an existing metrics.csv up to the resume point and appends."""

    def __init__(self, log_dir: str, keep_until: tuple[int, int] | None) -> None:
        """`keep_until` = (checkpoint epoch, global step); None behaves like Lightning (fresh file)."""
        self.keep_until = keep_until
        super().__init__(log_dir=log_dir)

    def _check_log_dir_exists(self) -> None:
        """Trim the existing file to rows up to the checkpoint and adopt its header (instead of deleting it)."""
        path = Path(self.metrics_file_path)
        if self.keep_until is None or not path.is_file():
            super()._check_log_dir_exists()
            return
        header, rows = read_rows(path)
        kept = [row for row in rows if row_before(row, *self.keep_until)]
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(kept)
        self.metrics_keys = header


class AppendingCSVLogger(CSVLogger):
    """CSVLogger at `<run_dir>/csv/version_0` whose writer continues an earlier session's metrics.csv."""

    def __init__(self, run_dir: Path, keep_until: tuple[int, int] | None) -> None:
        """`keep_until` from `resume_point`, or None for a fresh run."""
        super().__init__(save_dir=run_dir, name="csv", version=0)
        self.keep_until = keep_until

    @property
    @rank_zero_experiment
    def experiment(self) -> AppendingExperimentWriter:
        """The (lazily created) appending writer."""
        if self._experiment is None:
            self._fs.makedirs(self.root_dir, exist_ok=True)
            self._experiment = AppendingExperimentWriter(self.log_dir, self.keep_until)
        return self._experiment


def resume_point(state: dict | None) -> tuple[int, int] | None:
    """(epoch, global_step) stored in a checkpoint dict, or None for a fresh run."""
    return None if state is None else (int(state["epoch"]), int(state["global_step"]))
