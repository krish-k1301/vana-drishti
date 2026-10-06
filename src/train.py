"""Our training pipeline: `python -m src.train --config <yaml> [dotted.key=value ...]`.

Trains with early stopping on `validation/pixel_iou`, tests the best checkpoint and writes
`metrics_test.json`, `metrics_test.csv`, `config_resolved.yaml`, `run.log` and `csv/` into
`<output.runs_dir>/<experiment_name>`.
"""
import argparse
import logging

import torch
from torch.utils.data import DataLoader

from src.config import load_config, require
from src.data.loaders import build_dataloaders
from src.losses.build_loss import build_loss
from src.models.build_model import build_model
from src.models.inspection import count_parameters, describe_utae_shapes
from src.training.meters import score_rows
from src.training.module import ChangeDetectionModule
from src.training.run import (SMOKE_LABEL, best_checkpoint, build_trainer, monitored_callbacks, run_dir_for,
                              run_logger, seed_and_precision, write_metrics)

UTAE_NAMES = ("utae", "utae_seq2seq")


def data_config(cfg: dict) -> dict:
    """The `data` section with the top-level `seed` injected (single source of the seed)."""
    data_cfg = dict(require(cfg, "data"))
    if "seed" in data_cfg:
        raise ValueError("data.seed is not allowed; the top-level `seed` is the single source")
    return {**data_cfg, "seed": require(cfg, "seed")}


def monitored_key(cfg: dict) -> str:
    """The single metric that drives the scheduler, early stopping and checkpointing."""
    keys = {require(cfg, k) for k in ("optim.scheduler.monitor", "trainer.early_stopping.monitor",
                                      "trainer.checkpoint.monitor")}
    if len(keys) != 1:
        raise ValueError(f"scheduler, early stopping and checkpoint must monitor the same key, got {keys}")
    return keys.pop()


def log_model_summary(logger: logging.Logger, cfg: dict, model: torch.nn.Module, loader: DataLoader) -> None:
    """Log the parameter count and, for U-TAE, the per-stage shapes (PRD 4.2 sanity check)."""
    logger.info("model %s: %d trainable parameters", cfg["model"]["name"], count_parameters(model))
    if cfg["model"]["name"] in UTAE_NAMES:
        height, width = loader.dataset.load_raw(0)["image"].shape[-2:]
        for stage, shape in describe_utae_shapes(model.backbone, int(height), int(width)).items():
            logger.info("U-TAE stage %-16s %s", stage, shape)


def run(cfg: dict) -> dict:
    """Fit, test the best checkpoint, write the metrics files and return the test results."""
    run_dir = run_dir_for(cfg)
    logger = run_logger(run_dir, "train")
    if require(cfg, "smoke"):
        logger.info("%s run: synthetic or shortened data, results reproduce nothing", SMOKE_LABEL)
    seed_and_precision(cfg)
    loaders = build_dataloaders(data_config(cfg))
    model = build_model(require(cfg, "model"))
    log_model_summary(logger, cfg, model, loaders["train"])
    module = ChangeDetectionModule(model, build_loss(require(cfg, "loss")), require(cfg, "optim"))
    monitor = monitored_key(cfg)
    callbacks = monitored_callbacks(require(cfg, "trainer"), run_dir, monitor)
    trainer = build_trainer(cfg, run_dir, callbacks)
    trainer.fit(module, train_dataloaders=loaders["train"], val_dataloaders=loaders["validation"])
    checkpoint = best_checkpoint(callbacks)
    logger.info("testing best checkpoint %s", checkpoint)
    trainer.test(module, dataloaders=loaders["test"], ckpt_path=checkpoint)
    results = module.test_results
    payload = {"source": "src.train", "checkpoint": checkpoint, "epochs_trained": trainer.current_epoch,
               "monitor": monitor, **results}
    path = write_metrics(run_dir, cfg, payload, score_rows(results))
    logger.info("test pixel IoU (with OR) %.4f -> %s", results["pixel"]["with_or"]["iou"], path)
    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
