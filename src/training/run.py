"""Run plumbing shared by `src.train` and `src.reference_run`: run dir, log file, trainer, metrics files."""
import csv
import json
import logging
import sys
from pathlib import Path

import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from src.config import require

SMOKE_LABEL = "SMOKE"


def run_dir_for(cfg: dict) -> Path:
    """`<output.runs_dir>/<experiment_name>`, created, with the resolved config saved inside."""
    run_dir = Path(require(cfg, "output.runs_dir")) / require(cfg, "experiment_name")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config_resolved.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    return run_dir


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


def monitored_callbacks(trainer_cfg: dict, run_dir: Path, monitor: str) -> list[Callback]:
    """EarlyStopping and ModelCheckpoint(save_top_k) on `monitor`, settings from the trainer section."""
    stop, ckpt = trainer_cfg["early_stopping"], trainer_cfg["checkpoint"]
    return [
        EarlyStopping(monitor=monitor, mode=stop["mode"], patience=stop["patience"], min_delta=stop["min_delta"]),
        ModelCheckpoint(dirpath=run_dir / "checkpoints", monitor=monitor, mode=ckpt["mode"],
                        save_top_k=ckpt["save_top_k"], filename="best-epoch{epoch}", auto_insert_metric_name=False),
    ]


def build_trainer(cfg: dict, run_dir: Path, callbacks: list[Callback]) -> pl.Trainer:
    """Lightning Trainer with a CSV logger in `<run_dir>/csv`, every setting from `trainer`/`logging`."""
    trainer_cfg = require(cfg, "trainer")
    logging_cfg = require(cfg, "logging")
    if logging_cfg["wandb"] or logging_cfg["tensorboard"] or not logging_cfg["csv"]:
        raise ValueError("only logging.csv=true (W&B and TensorBoard off) is implemented")
    return pl.Trainer(
        default_root_dir=run_dir,
        logger=CSVLogger(save_dir=run_dir, name="csv", version=0),
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
