"""LightningModule for our pipeline: segment-wise model, change-detection loss, whole-set meters."""
import time

import lightning.pytorch as pl
import torch
from torch import nn

from src.metrics.segmentation import change_detection_outputs
from src.training.meters import PhaseMeters, flat_scores

PHASES = ("train", "validation", "test")


class ChangeDetectionModule(pl.LightningModule):
    """Trains `model` on the ChangeDetection target and logs epoch scores per phase (PRD 4.2, 9)."""

    def __init__(self, model: nn.Module, loss: nn.Module, optim_cfg: dict) -> None:
        """Wrap a model from build_model, a loss from build_loss and the `optim` config section."""
        super().__init__()
        self.model = model
        self.loss = loss
        self.optim_cfg = optim_cfg
        self.meters = {phase: PhaseMeters() for phase in PHASES}
        self.test_results: dict | None = None
        self._epoch_start = 0.0

    def forward(self, batch: dict) -> torch.Tensor:
        """Logits [B, t-1, 2, H, W] for a collated batch."""
        return self.model(batch["Images"], batch["ImageDays"], batch["TargetDays"], batch["PadMask"])

    def _step(self, batch: dict, phase: str) -> torch.Tensor:
        """Forward, loss on change targets, and meter update for one batch."""
        logits = self(batch)
        out = change_detection_outputs(logits, batch["Targets"], use_or=True)
        loss = self.loss(out["loss_pred"], out["loss_target"])
        self.meters[phase].update(logits, batch["Targets"], loss)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """One optimisation step."""
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Accumulate validation scores."""
        self._step(batch, "validation")

    def test_step(self, batch: dict, batch_idx: int) -> None:
        """Accumulate test scores."""
        self._step(batch, "test")

    def on_train_epoch_start(self) -> None:
        """Start the epoch timer and reset peak GPU memory."""
        self._epoch_start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def on_train_epoch_end(self) -> None:
        """Log train scores, loss, lr, epoch wall time and peak GPU memory."""
        extra = {"train/epoch_time_s": time.perf_counter() - self._epoch_start,
                 "train/lr": float(self.optimizers().param_groups[0]["lr"])}
        if torch.cuda.is_available():
            extra["train/peak_gpu_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        self._log_phase("train", extra)

    def on_validation_epoch_end(self) -> None:
        """Log validation scores (including the monitored `validation/pixel_iou`)."""
        self._log_phase("validation", {})

    def on_test_epoch_end(self) -> None:
        """Keep the nested test results for the metrics files and log the flat scores."""
        self.test_results = self.meters["test"].compute()
        self._log_phase("test", {})

    def _log_phase(self, phase: str, extra: dict) -> None:
        """Log one phase's epoch scores and reset its meters."""
        scores = flat_scores(self.meters[phase].compute(), phase)
        self.log_dict({**scores, **extra}, on_epoch=True)
        self.meters[phase].reset()

    def configure_optimizers(self) -> dict:
        """AdamW + ReduceLROnPlateau, every value from the `optim` config section."""
        cfg = self.optim_cfg
        if cfg["name"] != "adamw" or cfg["scheduler"]["name"] != "reduce_on_plateau":
            raise ValueError("only optim.name=adamw with scheduler reduce_on_plateau is implemented")
        optimizer = torch.optim.AdamW(self.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                      betas=tuple(cfg["betas"]), eps=cfg["eps"])
        sched = cfg["scheduler"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=sched["mode"], patience=sched["patience"], factor=sched["factor"],
            threshold=sched["threshold"], threshold_mode=sched["threshold_mode"],
            cooldown=sched["cooldown"], min_lr=sched["min_lr"],
        )
        return {"optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": sched["monitor"], "interval": "epoch",
                                 "frequency": 1}}
