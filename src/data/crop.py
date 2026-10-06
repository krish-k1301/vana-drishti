"""Spatial crops for the Phase 6 smaller-patch mitigation (32x32, 16x16 out of 48x48)."""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

SPEC_KEYS = ("enabled", "size", "mode", "input_size")
CROP_MODES = ("center", "random", "tile")
SPATIAL_KEYS = ("Images", "Targets", "EventMask", "BurnDay", "RaddDay", "RefDay")


def tile_starts(input_size: int, size: int) -> list[int]:
    """Tile start positions along one axis: ceil(input/size) evenly spaced starts covering every pixel.

    48 by 16 gives [0, 16, 32] (3x3 = 9 disjoint tiles). 48 by 32 cannot be tiled without overlap, so it
    gives [0, 16] (4 corner crops overlapping by 16 px): full coverage with no padding and no lost border.
    Metrics summed over overlapping tiles count the overlap strips twice; use `center` or size 16 when
    that matters.
    """
    n = math.ceil(input_size / size)
    if n == 1:
        return [0]
    return [round(i * (input_size - size) / (n - 1)) for i in range(n)]


class SpatialCrop:
    """Crop every spatial key of an item consistently and record the offset as `CropOffset` [2] (y, x).

    Modes: `center`; `random` (seeded, train only; other phases fall back to center, see build_crop);
    `tile` (each sample yields `num_tiles` items, one per tile from tile_starts on both axes).
    """

    def __init__(self, size: int, mode: str, input_size: int, seed: int) -> None:
        """Validate the crop and seed the generator for `random`."""
        if mode not in CROP_MODES:
            raise ValueError(f"crop mode must be one of {CROP_MODES}, got '{mode}'")
        if not 0 < size <= input_size:
            raise ValueError(f"crop size {size} must be in 1..{input_size}")
        self.size, self.mode, self.input_size = size, mode, input_size
        starts = tile_starts(input_size, size)
        self.tiles = [(y, x) for y in starts for x in starts] if mode == "tile" else [None]
        self._generator = torch.Generator().manual_seed(seed)

    @property
    def num_tiles(self) -> int:
        """Items produced per sample (1 unless mode is tile)."""
        return len(self.tiles)

    def reseed(self, seed: int) -> None:
        """Reset the random generator (called once per DataLoader worker)."""
        self._generator.manual_seed(seed)

    def offset(self, tile: int) -> tuple[int, int]:
        """Top-left corner (y, x) of the crop for this tile index."""
        if self.mode == "tile":
            return self.tiles[tile]
        if self.mode == "center":
            start = (self.input_size - self.size) // 2
            return start, start
        y, x = torch.randint(self.input_size - self.size + 1, (2,), generator=self._generator).tolist()
        return y, x

    def __call__(self, item: dict, tile: int = 0) -> dict:
        """Item with every spatial key cropped to size x size."""
        y, x = self.offset(tile)
        out = dict(item)
        for key in SPATIAL_KEYS:
            if key in item:
                if item[key].shape[-2:] != (self.input_size, self.input_size):
                    raise ValueError(f"{key} is {tuple(item[key].shape[-2:])}, expected input_size {self.input_size}")
                out[key] = item[key][..., y:y + self.size, x:x + self.size]
        out["CropOffset"] = torch.tensor([y, x], dtype=torch.long)
        return out


def build_crop(spec: Mapping | None, phase: str, seed: int) -> SpatialCrop | None:
    """SpatialCrop from a config block (None if absent/disabled); `random` becomes `center` outside train."""
    if spec is None:
        return None
    missing = [k for k in SPEC_KEYS if k not in spec]
    if missing:
        raise KeyError(f"crop needs keys {missing}")
    if not spec["enabled"]:
        return None
    mode = "center" if spec["mode"] == "random" and phase != "train" else spec["mode"]
    return SpatialCrop(int(spec["size"]), mode, int(spec["input_size"]), seed)
