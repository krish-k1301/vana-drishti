"""Tests for src.metrics.strata and src.metrics.regions."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics.strata import (  # noqa: E402
    SizeBinMeter,
    bin_labels,
    edge_interior,
    edge_interior_meter,
    minimum_mapping_unit,
    size_bins,
)


def _three_blobs():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[0:3, 0:3] = 1  # 9 px = 0.09 ha -> bin 0
    mask[10:20, 10:20] = 1  # 100 px = 1.0 ha -> bin 1
    mask[25:40, 25:40] = 1  # 225 px = 2.25 ha -> bin 2
    return mask


def test_bin_labels_default() -> None:
    """bin_labels names the size bins from their edges in hectares."""
    assert bin_labels((0.5, 2.0)) == ["<0.5 ha", "0.5-2 ha", ">2 ha"]


def test_size_bins_components_and_map() -> None:
    """size_bins assigns each connected component and its pixels to a size bin."""
    mask = _three_blobs()
    bin_map, table = size_bins(mask)
    assert sorted((r["n_pixels"], r["bin"]) for r in table) == [(9, 0), (100, 1), (225, 2)]
    assert bin_map[1, 1] == 0 and bin_map[15, 15] == 1 and bin_map[30, 30] == 2 and bin_map[5, 5] == -1


def test_size_bins_eight_connectivity_and_pixel_area() -> None:
    """Diagonal pixels form one 8-connected component whose area uses the pixel area."""
    mask = np.eye(5, dtype=np.uint8)  # diagonal touches -> one component under 8-connectivity
    _, table = size_bins(mask, pixel_area_ha=0.1, bins=(0.5, 2.0))
    assert len(table) == 1 and table[0]["area_ha"] == pytest.approx(0.5) and table[0]["bin"] == 1


def test_size_bin_meter_pixels_and_component_recall() -> None:
    """SizeBinMeter scores pixels and component recall per size bin."""
    ref = _three_blobs()
    pred = np.zeros_like(ref)
    pred[10:20, 10:15] = 1  # half of the 1 ha blob
    pred[25:40, 25:40] = 1  # all of the large blob
    pred[0, 30] = 1  # 1 px false alarm -> bin 0 by its own size
    meter = SizeBinMeter(hit_fraction=0.5)
    meter.update(torch.from_numpy(pred), ref)
    out = meter.compute()
    small, mid, large = out["<0.5 ha"], out["0.5-2 ha"], out[">2 ha"]
    assert (small["tp"], small["fp"], small["fn"]) == (0, 1, 9)
    assert (mid["tp"], mid["fn"], mid["iou"]) == (50, 50, pytest.approx(0.5))
    assert large["iou"] == pytest.approx(1.0)
    assert [b["component_recall"] for b in (small, mid, large)] == [0.0, 1.0, 1.0]
    strict = SizeBinMeter(hit_fraction=0.6)
    strict.update(pred[None], ref[None])
    assert strict.compute()["0.5-2 ha"]["n_detected"] == 0


def test_edge_interior_square() -> None:
    """edge_interior splits a square into interior, edge ring and outer ring."""
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:10, 2:10] = 1  # 8x8 square: 2 px edge ring leaves a 4x4 interior
    region = edge_interior(mask, width_px=2, outer_ring=True)
    assert (region == 1).sum() == 16
    assert (region == 2).sum() == 64 - 16
    assert (region == 3).sum() == 144 - 64  # chessboard 2 px ring fills the rest of the 12x12 patch


def test_edge_interior_patch_border_is_not_boundary() -> None:
    """The patch border is not treated as a clearing boundary."""
    mask = np.ones((6, 6), dtype=np.uint8)
    region = edge_interior(mask, width_px=2)
    assert (region == 1).all()


def test_edge_interior_meter_split() -> None:
    """The edge/interior meter scores edge and interior pixels separately."""
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[2:10, 2:10] = 1
    region = edge_interior(mask, width_px=2)
    pred = (region == 1).astype(np.uint8)  # only the interior is predicted
    meter = edge_interior_meter()
    meter.update(pred, mask, region)
    out = meter.compute()
    assert out["interior"]["recall"] == pytest.approx(1.0)
    assert out["edge"]["tp"] == 0 and out["edge"]["fn"] == 48


def test_minimum_mapping_unit() -> None:
    """minimum_mapping_unit states the MMU in hectares from pixel size and component size."""
    assert minimum_mapping_unit(10.0) == "MMU 0.01 ha (1 px of 10 m, 8-connected components)"
    assert minimum_mapping_unit(10.0, min_component_px=5).startswith("MMU 0.05 ha")
