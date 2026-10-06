"""Pure decoding of pulled band arrays (direct numpy pull or batch CSV tables) into per-patch arrays."""
from __future__ import annotations

import numpy as np
import pandas as pd

from gee.dates import NO_DATE, Window, decode_radd_yyddd, first_burn_day, to_days
from gee.grid import PatchGrid, lonlat_to_rowcol
from gee.labels import HANSEN_LOSS, MODIS_BURN, RADD_ALERT, RADD_DATE, fogo_band_name
from gee.s1 import s1_band_names

LON_BAND = "longitude"
LAT_BAND = "latitude"


def structured_to_bands(array: np.ndarray) -> dict[str, np.ndarray]:
    """Convert a structured numpy array from computePixels (fields = band names) into a band dict."""
    return {name: np.asarray(array[name]) for name in array.dtype.names}


def table_to_bands(table: pd.DataFrame, grid: PatchGrid, bands: list[str]) -> dict[str, np.ndarray]:
    """Rebuild [H, W] band arrays from a batch-exported pixel table with longitude/latitude columns."""
    rows, cols = lonlat_to_rowcol(table[LON_BAND].to_numpy(), table[LAT_BAND].to_numpy(), grid)
    inside = (rows >= 0) & (rows < grid.size) & (cols >= 0) & (cols < grid.size)
    out = {}
    for band in bands:
        arr = np.full((grid.size, grid.size), np.nan, dtype=np.float64)
        arr[rows[inside], cols[inside]] = table[band].to_numpy()[inside]
        out[band] = arr
    return out


def s1_series(bands: dict[str, np.ndarray], n_dates: int, polarisations: list[str],
              nodata: float) -> tuple[np.ndarray, list[int]]:
    """Return (image float32 [T', P, H, W], kept date indices), dropping dates with nodata or non-finite pixels."""
    names = s1_band_names(n_dates, polarisations)
    stack = np.stack([bands[n] for n in names]).astype(np.float64)
    stack = stack.reshape(n_dates, len(polarisations), *stack.shape[1:])
    bad = ~np.isfinite(stack) | (stack == nodata)
    keep = [i for i in range(n_dates) if not bad[i].any()]
    return stack[keep].astype(np.float32), keep


def radd_days(bands: dict[str, np.ndarray], window: Window, base_year: int) -> np.ndarray:
    """Per-pixel RADD alert day (days since epoch) for alerts inside the window, -1 otherwise."""
    alert = np.nan_to_num(bands[RADD_ALERT]).astype(np.int64)
    days = decode_radd_yyddd(np.where(alert > 0, np.nan_to_num(bands[RADD_DATE]), 0), base_year)
    in_window = (days >= to_days(window.start)) & (days <= to_days(window.end))
    return np.where(in_window, days, NO_DATE).astype(np.int32)


def burn_days(bands: dict[str, np.ndarray], window: Window, shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel first MapBiomas Fogo burn day in the window (int16), all -1 when Fogo was not pulled."""
    names = {y: fogo_band_name(y) for y in window.years}
    if not all(n in bands for n in names.values()):
        return np.full(shape, NO_DATE, dtype=np.int16)
    monthly = {y: np.nan_to_num(bands[n]).astype(np.int64) for y, n in names.items()}
    return first_burn_day(monthly, window).astype(np.int16)


def hansen_lossyear(bands: dict[str, np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    """Hansen lossyear array (0 = no loss); zeros when Hansen was not pulled."""
    if HANSEN_LOSS not in bands:
        return np.zeros(shape, dtype=np.int32)
    return np.nan_to_num(bands[HANSEN_LOSS]).astype(np.int32)


def modis_flag(bands: dict[str, np.ndarray]) -> int:
    """Patch-level MODIS burn flag: 1 if any pixel burned in the window, 0 if none, -1 if not pulled."""
    if MODIS_BURN not in bands:
        return -1
    return int(np.nan_to_num(bands[MODIS_BURN]).max() > 0)
