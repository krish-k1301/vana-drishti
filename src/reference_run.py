"""Reference run: thin launcher around the UPSTREAM BraDD-S1TS code (PRD 7, Phase 1 track 1).

`python -m src.reference_run --config <yaml> [dotted.key=value ...]`

Builds upstream MultiEarthDataModule, Network('UTAE', 'segment'), LossFunction, SegmentationScores,
ForwardFunction('ChangeDetection') and Experiment exactly like upstream `run_via_parser.py`, but with every
PRD 4.2 value passed explicitly and a CSV logger only. No upstream logic is reimplemented. The one
work-around: upstream's stats bootstrap crashes when `<root>/<split>_stats.pt` is missing (PRD 4.4
issue 1), so that file is precomputed from the train split with our `compute_train_stats`.
"""
import argparse
import logging
from pathlib import Path
from types import ModuleType

import torch
from lightning.pytorch.callbacks import LearningRateMonitor

from src.config import load_config, require
from src.data.stats import compute_train_stats
from src.models.build_model import POSITIONAL_PERIOD_KEY
from src.models.vendor import import_upstream
from src.training.run import (SMOKE_LABEL, best_checkpoint, build_trainer, monitored_callbacks, run_dir_for,
                              run_logger, seed_and_precision, write_metrics)

UPSTREAM_MONITOR = "StoppingScore/Epoch"  # run_via_parser.py: EarlyStopping / ModelCheckpoint monitor
UPSTREAM_FIXED = {  # values hard-coded upstream (Network, LTAE2d, Experiment); config must agree
    "model.error_days_before": 30, "model.error_days_after": 30,
    f"model.params.{POSITIONAL_PERIOD_KEY}": 1000, "optim.name": "adamw", "optim.betas": [0.9, 0.999],
    "optim.eps": 1.0e-8, "optim.scheduler.mode": "max", "optim.scheduler.factor": 0.1,
    "optim.scheduler.threshold": 1.0e-4, "optim.scheduler.threshold_mode": "rel",
    "optim.scheduler.cooldown": 0, "optim.scheduler.min_lr": 0.0, "model.name": "utae",
}
UPSTREAM_LOSS_NAMES = {"cross_entropy": "CrossEntropy", "focal": "FocalLoss"}


def check_upstream_compatible(cfg: dict) -> None:
    """Refuse configs asking for something the upstream code cannot do."""
    wrong = {k: require(cfg, k) for k, v in UPSTREAM_FIXED.items() if require(cfg, k) != v}
    if wrong:
        raise ValueError(f"upstream hard-codes {[(k, UPSTREAM_FIXED[k]) for k in wrong]}; config has {wrong}")


def ensure_upstream_stats(data_cfg: dict, split: str, logger: logging.Logger) -> None:
    """Write `<root>/<split>_stats.pt` in upstream's format (mean/std/min/max) if it is missing."""
    path = Path(data_cfg["root"]) / f"{split}_stats.pt"
    if path.exists():
        logger.info("using existing upstream stats file %s", path)
        return
    torch.save(compute_train_stats(data_cfg["root"], f"{split}_set"), path)
    logger.info("WORK-AROUND PRD 4.4 issue 1: wrote %s from the train split with compute_train_stats", path)


def upstream_objects(cfg: dict, source: ModuleType, run_dir: Path) -> tuple:
    """Construct the upstream data module and Experiment with every value from config."""
    data_cfg, ref, optim = require(cfg, "data"), require(cfg, "reference"), require(cfg, "optim")
    data_module = source.MultiEarthDataModule(
        batchSize=data_cfg["batch_size"], Dataset_path=str(data_cfg["root"]), Dataset_split=ref["split"],
        Dataset_NormalizationMethod=ref["normalization_method"], Dataset_numCpu=data_cfg["num_workers"],
        Dataset_TemporalDropout_isRandom=ref["temporal_dropout"]["is_random"],
        Dataset_TemporalDropout_numTemporal=ref["temporal_dropout"]["num_temporal"],
    )
    params = {k: v for k, v in require(cfg, "model.params").items() if k != POSITIONAL_PERIOD_KEY}
    network = source.Network("UTAE", require(cfg, "model.forward_type"),
                             **{f"Model_{k}": v for k, v in params.items()})
    loss_cfg = require(cfg, "loss")
    loss_kwargs = {f"LossHyperparameters_{k}": loss_cfg[k] for k in ("alpha", "gamma") if k in loss_cfg}
    loss = source.LossFunction(UPSTREAM_LOSS_NAMES[loss_cfg["name"]], **loss_kwargs)
    scores = {k: source.SegmentationScores(numClass=2, evalScores=True, runningScores=False)
              for k in ("Train", "Validation", "Test")}
    experiment = source.Experiment(
        network=network, loss=loss, score=scores, forwardFunction=source.ForwardFunction("ChangeDetection"),
        learningRate=optim["lr"], weightDecay=optim["weight_decay"], patienceEpoch=optim["scheduler"]["patience"],
        savingFolder=str(run_dir / "Samples") if ref["save_outputs"] else None,
    )
    return data_module, experiment


def run(cfg: dict) -> dict:
    """Fit and test the upstream Experiment; write metrics_test.json/.csv and return the test scores."""
    check_upstream_compatible(cfg)
    run_dir = run_dir_for(cfg)
    logger = run_logger(run_dir, "reference")
    if require(cfg, "smoke"):
        logger.info("%s run: synthetic or shortened data, results reproduce nothing", SMOKE_LABEL)
    source = import_upstream()
    ensure_upstream_stats(require(cfg, "data"), require(cfg, "reference.split"), logger)
    seed_and_precision(cfg)
    data_module, experiment = upstream_objects(cfg, source, run_dir)
    callbacks = monitored_callbacks(require(cfg, "trainer"), run_dir, UPSTREAM_MONITOR)
    callbacks.append(LearningRateMonitor(logging_interval="step"))
    trainer = build_trainer(cfg, run_dir, callbacks)
    trainer.fit(model=experiment, datamodule=data_module)
    checkpoint = best_checkpoint(callbacks)
    logger.info("testing best checkpoint %s", checkpoint)
    (scores,) = trainer.test(model=experiment, datamodule=data_module, ckpt_path=checkpoint)
    epoch_prefix = "TestScores/Epoch/"  # whole-set scores; TestScores/Batch/* are Lightning means of batch scores
    test_scores = {k.removeprefix(epoch_prefix): float(v) for k, v in scores.items() if k.startswith(epoch_prefix)}
    payload = {"source": "upstream (src.reference_run)", "checkpoint": checkpoint,
               "epochs_trained": trainer.current_epoch, "monitor": UPSTREAM_MONITOR,
               "pixel": {"with_or": test_scores}}
    row = {"level": "pixel", "variant": "with_or", **test_scores}
    path = write_metrics(run_dir, cfg, payload, [row])
    logger.info("upstream test pixel IoU %.4f -> %s", test_scores["IoU"], path)
    return test_scores


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    args = parser.parse_args()
    run(load_config(args.config, args.overrides))


if __name__ == "__main__":
    main()
