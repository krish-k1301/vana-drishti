"""Shared evaluation plumbing: model from checkpoint, run directory and the SMOKE header."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from src.config import require
from src.models.build_model import build_model
from src.training.run import SMOKE_LABEL

MODEL_PREFIX = "model."  # ChangeDetectionModule stores the network as `self.model`


def load_model(cfg: dict, checkpoint: str | Path) -> nn.Module:
    """Rebuild the network from `cfg['model']` and load the weights of a ChangeDetectionModule checkpoint."""
    model = build_model(require(cfg, "model"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    weights = {k[len(MODEL_PREFIX):]: v for k, v in state.items() if k.startswith(MODEL_PREFIX)}
    if not weights:
        raise ValueError(f"checkpoint {checkpoint} has no '{MODEL_PREFIX}*' weights")
    model.load_state_dict(weights, strict=True)
    return model.eval()


def run_dir(cfg: dict) -> Path:
    """`<output.runs_dir>/<experiment_name>`, the run's directory written by src.train."""
    return Path(require(cfg, "output.runs_dir")) / require(cfg, "experiment_name")


def smoke_header(cfg: dict) -> dict:
    """Label fields every output carries; SMOKE propagates from the run config."""
    smoke = bool(require(cfg, "smoke"))
    return {"label": SMOKE_LABEL if smoke else "", "smoke": smoke, "experiment_name": require(cfg, "experiment_name")}
