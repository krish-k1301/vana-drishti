"""Size-bin and edge/interior meters over change-detection score maps, with and without the OR rule."""
from __future__ import annotations

import torch

from src.evaluation.io import pixel_area_ha
from src.metrics.segmentation import change_detection_outputs
from src.metrics.strata import SizeBinMeter, edge_interior, edge_interior_meter
from src.training.meters import COUNT_KEYS, SCORE_KEYS

VARIANTS = {"with_or": True, "without_or": False}


class StrataMeters:
    """Per size bin (PRD 7 Phase 5/6) and edge/interior/outer-ring (PRD 3 H2) scores of the score maps.

    The reference polygon is the scored target `Targets[:, 1:]`, the prediction is the scored prediction of
    each variant, so per-stratum numbers decompose the headline pixel scores of the same variant.
    """

    def __init__(self, eval_cfg: dict) -> None:
        """Read pixel size, size bins, component hit fraction and edge width from the eval config."""
        self.edge_width = int(eval_cfg["edge_width_px"])
        area = pixel_area_ha(float(eval_cfg["pixel_size_m"]))
        bins = tuple(eval_cfg["size_bins_ha"])
        hit = float(eval_cfg["component_hit_fraction"])
        self.size = {v: SizeBinMeter(hit, pixel_area_ha=area, bins=bins) for v in VARIANTS}
        self.edge = {v: edge_interior_meter(outer_ring=True) for v in VARIANTS}

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Add a batch: logits [B, t-1, 2, H, W], targets [B, t, H, W]."""
        for variant, use_or in VARIANTS.items():
            out = change_detection_outputs(logits.detach(), targets, use_or=use_or)
            pred, ref = out["score_pred"].cpu().numpy(), out["score_target"].cpu().numpy()
            self.size[variant].update(pred, ref)
            for p, r in zip(pred, ref):
                self.edge[variant].update(p, r, edge_interior(r, self.edge_width, outer_ring=True))

    def rows(self) -> list[dict]:
        """Tidy rows: one per (variant, stratum type, stratum) with counts, scores and component recall."""
        out = []
        for variant in VARIANTS:
            for kind, results in (("size_bin", self.size[variant].compute()),
                                  ("edge_interior", self.edge[variant].compute())):
                for stratum, scores in results.items():
                    extra = {k: scores[k] for k in ("n_components", "n_detected", "component_recall") if k in scores}
                    out.append({"level": "pixel", "variant": variant, "stratum_type": kind, "stratum": stratum,
                                **{k: scores[k] for k in COUNT_KEYS + SCORE_KEYS}, **extra})
        return out
