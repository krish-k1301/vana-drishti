"""Run plumbing shared by `src.train` and `src.reference_run`: run dir, log file, trainer, metrics files."""
import csv
import json
import logging
import sys
from pathlib import Path
from typing import NamedTuple

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

from src.config import require
from src.training.resume import AppendingCSVLogger, checkpoint_state, resume_checkpoint, resume_point

SMOKE_LABEL = "SMOKE"


class RunStart(NamedTuple):
    """What a launcher needs to begin or continue a run."""

    run_dir: Path
    logger: logging.Logger
    checkpoint: str | None   # last.ckpt to pass as trainer.fit(ckpt_path=...), None = fresh start
    state: dict | None       # that checkpoint's contents (epoch, global_step, callback states)


def run_dir_for(cfg: dict) -> Path:
    """`<output.runs_dir>/<experiment_name>`, created."""
    run_dir = Path(require(cfg, "output.runs_dir")) / require(cfg, "experiment_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def start_run(cfg: dict, name: str) -> RunStart:
    """Run dir, resume decision (config checked before it is overwritten), saved config and logger."""
    run_dir = run_dir_for(cfg)
    checkpoint = resume_checkpoint(cfg, run_dir)
    with (run_dir / "config_resolved.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    logger = run_logger(run_dir, name)
    if require(cfg, "smoke"):
        logger.info("%s run: synthetic or shortened data, results reproduce nothing", SMOKE_LABEL)
    if checkpoint is None:
        reason = "no checkpoints/last.ckpt in the run dir" if require(cfg, "train.resume") else "train.resume=false"
        logger.info("starting fresh at epoch 0 (%s)", reason)
        return RunStart(run_dir, logger, None, None)
    state = checkpoint_state(checkpoint)
    logger.info("RESUMING from %s: checkpoint epoch %d (global step %d), training continues at epoch %d",
                checkpoint, state["epoch"], state["global_step"], state["epoch"] + 1)
    return RunStart(run_dir, logger, str(checkpoint), state)


def run_logger(run_dir: Path, name: str) -> logging.Logger:
    """Logger writing to stdout and `<run_dir>/run.log`."""
    logger = logging.getLogger(f"{name}.{run_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "run.log")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_and_precision(cfg: dict) -> None:
    """Seed everything from the top-level seed and set the float32 matmul precision."""
    pl.seed_everything(require(cfg, "seed"), workers=require(cfg, "seed_workers"))
    torch.set_float32_matmul_precision(require(cfg, "trainer.float32_matmul_precision"))


class StayStopped(Callback):
    """Resuming an early-stopped run must not train another epoch (Lightning does not checkpoint should_stop)."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Re-raise the stop flag when the restored EarlyStopping state had already fired."""
        stop = next(c for c in trainer.callbacks if isinstance(c, EarlyStopping))
        if trainer.current_epoch > 0 and stop.wait_count >= stop.patience:
            trainer.should_stop = True


def monitored_callbacks(trainer_cfg: dict, run_dir: Path, monitor: str) -> list[Callback]:
    """EarlyStopping and ModelCheckpoint(save_top_k best + last.ckpt) on `monitor`, settings from `trainer`."""
    stop, ckpt = trainer_cfg["early_stopping"], trainer_cfg["checkpoint"]
    return [
        EarlyStopping(monitor=monitor, mode=stop["mode"], patience=stop["patience"], min_delta=stop["min_delta"]),
        ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor=monitor, mode=ckpt["mode"],
                        save_top_k=ckpt["save_top_k"], filename="best-epoch{epoch}", auto_insert_metric_name=False,
                        save_last=True),
        StayStopped(),
    ]


def build_trainer(cfg: dict, start: RunStart, callbacks: list[Callback]) -> pl.Trainer:
    """Lightning Trainer with a CSV logger in `<run_dir>/csv` (continued on resume), settings from config."""
    trainer_cfg = require(cfg, "trainer")
    logging_cfg = require(cfg, "logging")
    if logging_cfg["wandb"] or logging_cfg["tensorboard"] or not logging_cfg["csv"]:
        raise ValueError("only logging.csv=true (W&B and TensorBoard off) is implemented")
    return pl.Trainer(
        default_root_dir=start.run_dir,
        logger=AppendingCSVLogger(start.run_dir, resume_point(start.state)),
        callbacks=callbacks,
        max_epochs=trainer_cfg["max_epochs"],
        max_time=trainer_cfg["max_time"],
        precision=trainer_cfg["precision"],
        accelerator=trainer_cfg["accelerator"],
        deterministic=trainer_cfg["deterministic"],
        num_sanity_val_steps=trainer_cfg["num_sanity_val_steps"],
        enable_progress_bar=trainer_cfg["enable_progress_bar"],
        log_every_n_steps=trainer_cfg["log_every_n_steps"],
    )


def best_checkpoint(callbacks: list[Callback]) -> str:
    """Path of the best checkpoint kept by the ModelCheckpoint callback."""
    ckpt = next(c for c in callbacks if isinstance(c, ModelCheckpoint))
    if not ckpt.best_model_path:
        raise RuntimeError("no checkpoint was saved; cannot test the best model")
    return ckpt.best_model_path


def write_metrics(run_dir: Path, cfg: dict, payload: dict, rows: list[dict]) -> Path:
    """Write `metrics_test.json` (labelled SMOKE when cfg.smoke) and `metrics_test.csv` into `run_dir`."""
    smoke = bool(require(cfg, "smoke"))
    header = {"label": SMOKE_LABEL if smoke else "", "smoke": smoke,
              "experiment_name": require(cfg, "experiment_name")}
    json_path = run_dir / "metrics_test.json"
    json_path.write_text(json.dumps({**header, **payload}, indent=2, default=float))
    with (run_dir / "metrics_test.csv").open("w", newline="") as handle:
        fields = ["label", "experiment_name", *rows[0].keys()]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"label": header["label"], "experiment_name": header["experiment_name"], **row})
    return json_path
