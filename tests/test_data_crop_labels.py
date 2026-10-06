"""Spatial crops (center/random/tile) and the PRODES/Hansen label-source switch."""
from __future__ import annotations

import pytest
import torch

from src.data import BraDDDataset, DatedDataset, PrefixTruncation, build_dataloaders
from src.data.crop import SPATIAL_KEYS, tile_starts
from tests.fixtures import data_config


def _crop(size, mode):
    """A complete enabled crop block."""
    return {"enabled": True, "size": size, "mode": mode, "input_size": 48}


def test_tile_starts() -> None:
    """48 splits into 3 disjoint 16-px tiles per axis and 2 overlapping 32-px tiles per axis."""
    assert tile_starts(48, 16) == [0, 16, 32] and tile_starts(48, 32) == [0, 16]
    assert tile_starts(48, 48) == [0] and tile_starts(48, 24) == [0, 24]


@pytest.mark.parametrize("size,n_tiles", [(16, 9), (32, 4)])
def test_tiles_cover_the_patch_consistently(dated_root, size, n_tiles) -> None:
    """Every spatial key of every tile equals the same window of the full item; tiles cover all pixels."""
    full = DatedDataset(dated_root, "train", normalization="none")
    tiled = DatedDataset(dated_root, "train", normalization="none", crop=_crop(size, "tile"))
    assert len(tiled) == n_tiles * len(full)
    item = full[0]
    covered = torch.zeros(48, 48, dtype=torch.bool)
    for tile in range(n_tiles):
        part = tiled[tile]
        y, x = part["CropOffset"].tolist()
        for key in SPATIAL_KEYS:
            assert torch.equal(part[key], item[key][..., y:y + size, x:x + size]), key
        assert part["Index"].item() == 0 and torch.equal(part["ImageDays"], item["ImageDays"])
        covered[y:y + size, x:x + size] = True
    assert covered.all() and tiled[n_tiles]["Index"].item() == 1


def test_random_crop_train_only_and_seeded(bradd_root) -> None:
    """Random crops vary and are reproducible in train; other phases use the center crop."""
    a = BraDDDataset(bradd_root, "train", normalization="none", crop=_crop(32, "random"), seed=3)
    b = BraDDDataset(bradd_root, "train", normalization="none", crop=_crop(32, "random"), seed=3)
    offsets = [tuple(a[0]["CropOffset"].tolist()) for _ in range(6)]
    assert offsets == [tuple(b[0]["CropOffset"].tolist()) for _ in range(6)] and len(set(offsets)) > 1
    val = BraDDDataset(bradd_root, "validation", normalization="none", crop=_crop(32, "random"))
    item = val[0]
    assert item["CropOffset"].tolist() == [8, 8] and item["Images"].shape[-2:] == (32, 32)
    assert item["Targets"].shape[-2:] == (32, 32)


def test_crop_config_errors(bradd_root) -> None:
    """Bad modes, sizes and incomplete blocks are rejected."""
    with pytest.raises(ValueError):
        BraDDDataset(bradd_root, "train", normalization="none", crop=_crop(64, "center"))
    with pytest.raises(ValueError):
        BraDDDataset(bradd_root, "train", normalization="none", crop=_crop(16, "grid"))
    with pytest.raises(KeyError):
        BraDDDataset(bradd_root, "train", normalization="none", crop={"enabled": True, "size": 16})


def test_label_source_switch(dated_root, bradd_root) -> None:
    """'hansen' reads label_hansen, 'prodes' reads label; missing stacks and prefix mixing fail."""
    prodes = DatedDataset(dated_root, "train", normalization="none")
    hansen = DatedDataset(dated_root, "train", normalization="none", label_source="hansen")
    raw = prodes.load_raw(0)
    assert torch.equal(prodes[0]["Targets"], raw["label"]) and torch.equal(hansen[0]["Targets"], raw["label_hansen"])
    differs = any(not torch.equal(prodes.load_raw(i)["label"], prodes.load_raw(i)["label_hansen"])
                  for i in range(len(prodes)))
    assert differs
    with pytest.raises(KeyError, match="label_hansen"):
        BraDDDataset(bradd_root, "train", normalization="none", label_source="hansen")[0]
    with pytest.raises(ValueError):
        DatedDataset(dated_root, "train", normalization="none", label_source="hansen",
                     prefix=PrefixTruncation("deter", 2, 30))
    with pytest.raises(ValueError):
        DatedDataset(dated_root, "train", normalization="none", label_source="landsat")


def test_loader_per_phase_label_source_and_crop(dated_root) -> None:
    """Label decomposition config: Hansen in train, PRODES in test; crops flow through collation."""
    cfg = data_config("dated_amazon", root=str(dated_root), normalization="none", batch_size=4, num_workers=0,
                      seed=0, crop=_crop(16, "tile"), prefix_truncation=None,
                      label_source={"train": "hansen", "validation": "hansen", "test": "prodes"})
    loaders = build_dataloaders(cfg)
    assert loaders["train"].dataset.label_key == "label_hansen" and loaders["test"].dataset.label_key == "label"
    batch = next(iter(loaders["test"]))
    assert batch["Images"].shape[-2:] == (16, 16) and batch["CropOffset"].shape == (4, 2)
    assert batch["BurnDay"].shape == (4, 16, 16)
