"""Size-bin and edge/interior strata (PRD 3 H2, PRD 7 Phases 5-6, PRD 9) and the minimum mapping unit.

Connected components use 8-connectivity within one map (components never cross patches or intervals).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import ndimage

from src.metrics.regions import RegionMeter, as_maps, as_numpy

EIGHT_CONNECTED = np.ones((3, 3), dtype=bool)
OUTSIDE = -1


def bin_labels(bins: Sequence[float]) -> list[str]:
    """Return labels like ['<0.5 ha', '0.5-2 ha', '>2 ha'] for bin edges in hectares."""
    edges = [f"{b:g}" for b in bins]
    middle = [f"{lo}-{hi} ha" for lo, hi in zip(edges[:-1], edges[1:])]
    return [f"<{edges[0]} ha", *middle, f">{edges[-1]} ha"]


def component_bins(
    mask: np.ndarray, pixel_area_ha: float, bins: Sequence[float]
) -> tuple[np.ndarray, list[dict], np.ndarray]:
    """Return (bin-id map, component table, component labels) for a [H, W] mask; bins are [lo, hi) in ha."""
    labels, n = ndimage.label(as_numpy(mask) != 0, structure=EIGHT_CONNECTED)
    n_pixels = np.bincount(labels.ravel(), minlength=n + 1)[1:]
    areas = n_pixels * pixel_area_ha
    bin_ids = np.digitize(areas, bins, right=False)
    bin_map = np.full(labels.shape, OUTSIDE, dtype=np.int64)
    inside = labels > 0
    bin_map[inside] = bin_ids[labels[inside] - 1]
    table = [
        {"component": i + 1, "n_pixels": int(n_pixels[i]), "area_ha": float(areas[i]), "bin": int(bin_ids[i])}
        for i in range(n)
    ]
    return bin_map, table, labels


def size_bins(
    mask: np.ndarray, pixel_area_ha: float = 0.01, bins: Sequence[float] = (0.5, 2.0)
) -> tuple[np.ndarray, list[dict]]:
    """Per-pixel size-bin id map (-1 outside) and per-component table for a reference mask [H, W].

    Defaults: 10 m Sentinel-1 pixel = 0.01 ha; bins <0.5, 0.5-2, >2 ha from PRD 7 Phase 5.
    """
    bin_map, table, _ = component_bins(mask, pixel_area_ha, bins)
    return bin_map, table


def size_bin_regions(pred: np.ndarray, ref: np.ndarray, pixel_area_ha: float, bins: Sequence[float]) -> np.ndarray:
    """Region map for per-bin pixel metrics on one [H, W] map.

    Reference pixels take their reference component's bin (TP/FN). Predicted pixels outside the reference take
    the bin of their own predicted component (FP), so a false alarm counts against the size it appears as.
    """
    ref_bins, _, _ = component_bins(ref, pixel_area_ha, bins)
    pred_bins, _, _ = component_bins(pred, pixel_area_ha, bins)
    return np.where(as_numpy(ref) != 0, ref_bins, pred_bins)


class SizeBinMeter:
    """Pixel IoU/F1 per size bin plus component-level recall per reference size bin."""

    def __init__(self, hit_fraction: float, pixel_area_ha: float = 0.01, bins: Sequence[float] = (0.5, 2.0)) -> None:
        """`hit_fraction`: a reference component is detected if >= this fraction of its pixels is predicted."""
        self.hit_fraction = hit_fraction
        self.pixel_area_ha = pixel_area_ha
        self.bins = tuple(bins)
        self.labels = bin_labels(self.bins)
        self.pixels = RegionMeter(dict(enumerate(self.labels)))
        self.n_components = np.zeros(len(self.labels), dtype=np.int64)
        self.n_detected = np.zeros(len(self.labels), dtype=np.int64)

    def update(self, pred: np.ndarray, ref: np.ndarray) -> None:
        """Add binary prediction and reference maps, [H, W] or [N, H, W]."""
        for p, r in zip(as_maps(pred), as_maps(ref)):
            self.pixels.update(p, r, size_bin_regions(p, r, self.pixel_area_ha, self.bins))
            _, table, labels = component_bins(r, self.pixel_area_ha, self.bins)
            hits = np.bincount(labels[p != 0], minlength=len(table) + 1)[1:]
            for row, n_hit in zip(table, hits):
                self.n_components[row["bin"]] += 1
                self.n_detected[row["bin"]] += int(n_hit >= self.hit_fraction * row["n_pixels"])

    def compute(self) -> dict:
        """Return {bin label: pixel scores + n_components, n_detected, component_recall}."""
        out = self.pixels.compute()
        for i, label in enumerate(self.labels):
            n = int(self.n_components[i])
            out[label] |= {
                "n_components": n,
                "n_detected": int(self.n_detected[i]),
                "component_recall": int(self.n_detected[i]) / n if n else float("nan"),
            }
        return out


EDGE_NAMES = {1: "interior", 2: "edge", 3: "outer_ring"}


def edge_interior(mask: np.ndarray, width_px: int = 2, outer_ring: bool = False) -> np.ndarray:
    """Map a [H, W] polygon mask to 0 outside, 1 interior, 2 edge, 3 outer ring (if requested).

    Edge = mask pixels within `width_px` (chessboard distance) of the polygon boundary; 2 px is PRD 3 H2.
    The patch border is not a polygon boundary (unknown beyond it), so erosion treats outside the patch as mask.
    """
    inside = as_numpy(mask) != 0
    core = ndimage.binary_erosion(inside, structure=EIGHT_CONNECTED, iterations=width_px, border_value=1)
    region = np.zeros(inside.shape, dtype=np.int64)
    region[inside & ~core] = 2
    region[core] = 1
    if outer_ring:
        grown = ndimage.binary_dilation(inside, structure=EIGHT_CONNECTED, iterations=width_px)
        region[grown & ~inside] = 3
    return region


def edge_interior_meter(outer_ring: bool = False) -> RegionMeter:
    """Return a RegionMeter keyed to `edge_interior` region ids (interior, edge, optional outer ring)."""
    ids = (1, 2, 3) if outer_ring else (1, 2)
    return RegionMeter({i: EDGE_NAMES[i] for i in ids})


def minimum_mapping_unit(pixel_size_m: float, min_component_px: int = 1) -> str:
    """Return the MMU statement every results table must carry (PRD 9)."""
    area_ha = min_component_px * pixel_size_m**2 / 1e4
    return f"MMU {area_ha:g} ha ({min_component_px} px of {pixel_size_m:g} m, 8-connected components)"
