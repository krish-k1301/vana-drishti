"""Shared evaluation plumbing: model from checkpoint, run directory and the SMOKE header."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import require
from src.models.build_model import build_model
from src.training.module import load_model_weights
from src.training.run import SMOKE_LABEL


def load_model(cfg: dict, checkpoint: str | Path) -> nn.Module:
    """Rebuild the network from `cfg['model']`, load ChangeDetectionModule weights, match training's matmul precision."""
    torch.set_float32_matmul_precision(require(cfg, "trainer.float32_matmul_precision"))
    model = build_model(require(cfg, "model"))
    load_model_weights(model, str(checkpoint))
    return model.eval()


def run_dir(cfg: dict) -> Path:
    """`<output.runs_dir>/<experiment_name>`, the run's directory written by src.train."""
    return Path(require(cfg, "output.runs_dir")) / require(cfg, "experiment_name")


def smoke_header(cfg: dict) -> dict:
    """Label fields every output carries; SMOKE propagates from the run config."""
    smoke = bool(require(cfg, "smoke"))
    return {"label": SMOKE_LABEL if smoke else "", "smoke": smoke, "experiment_name": require(cfg, "experiment_name")}
