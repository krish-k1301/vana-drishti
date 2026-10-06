"""Earth Engine export backends: direct numpy pull (pilots) and batch table exports (full runs), plus raw I/O."""
from __future__ import annotations

import json
import os

import ee
import numpy as np
import pandas as pd

from gee.decode import LAT_BAND, LON_BAND, structured_to_bands, table_to_bands
from gee.grid import PatchGrid
from gee.labels import patch_region
from gee.tasklog import TaskLog

BATCH_STATE_MAP = {"COMPLETED": "COMPLETED", "FAILED": "FAILED", "CANCELLED": "FAILED", "CANCEL_REQUESTED": "FAILED"}


def pull_direct(image: object, grid: PatchGrid) -> dict[str, np.ndarray]:
    """Pull all bands of `image` on the patch grid with ee.data.computePixels as numpy arrays."""
    request = {"expression": image, "fileFormat": "NUMPY_NDARRAY", "grid": grid.compute_pixels_grid()}
    return structured_to_bands(ee.data.computePixels(request))


def pixel_table(image: object, grid: PatchGrid) -> object:
    """Per-pixel FeatureCollection of the image plus pixel-centre lon/lat on the patch grid."""
    stacked = image.addBands(ee.Image.pixelLonLat())
    return stacked.sample(region=patch_region(grid), projection=ee.Projection(grid.crs, grid.affine),
                          dropNulls=False, geometries=False)


def submit_batch(image: object, grid: PatchGrid, patch_id: str, bands: list[str], batch_cfg: dict) -> str:
    """Start a CSV table export of one patch to Cloud Storage or Drive and return its task id."""
    table = pixel_table(image, grid)
    selectors = bands + [LON_BAND, LAT_BAND]
    if batch_cfg["destination"] == "gcs":
        task = ee.batch.Export.table.toCloudStorage(
            collection=table, description=patch_id, bucket=batch_cfg["bucket"],
            fileNamePrefix=f"{batch_cfg['prefix']}/{patch_id}", fileFormat="CSV", selectors=selectors)
    elif batch_cfg["destination"] == "drive":
        task = ee.batch.Export.table.toDrive(
            collection=table, description=patch_id, folder=batch_cfg["drive_folder"],
            fileNamePrefix=patch_id, fileFormat="CSV", selectors=selectors)
    else:
        raise ValueError(f"unknown batch destination {batch_cfg['destination']!r}")
    task.start()
    return task.id


def refresh_batch(log: TaskLog) -> dict[str, str]:
    """Poll pending batch tasks and log terminal states; returns patch id -> new state."""
    pending = log.pending_tasks()
    if not pending:
        return {}
    updates = {}
    for status in ee.data.getTaskStatus(list(pending)):
        state = BATCH_STATE_MAP.get(status.get("state", ""))
        if state is not None:
            patch_id = pending[status["id"]]
            log.append(patch_id, state, "batch", status["id"], status.get("error_message", ""))
            updates[patch_id] = state
    return updates


def save_raw(raw_dir: str, patch_id: str, bands: dict[str, np.ndarray] | None, info: dict) -> None:
    """Write the patch sidecar JSON and, when given, the pulled bands as compressed npz."""
    os.makedirs(raw_dir, exist_ok=True)
    with open(os.path.join(raw_dir, f"{patch_id}.json"), "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=1)
    if bands is not None:
        np.savez_compressed(os.path.join(raw_dir, f"{patch_id}.npz"), **bands)


def load_raw(raw_dir: str, patch_id: str, grid: PatchGrid) -> tuple[dict[str, np.ndarray], dict] | None:
    """Load a pulled patch (npz from direct pulls, or the CSV mirrored from a batch export); None if absent."""
    info_path = os.path.join(raw_dir, f"{patch_id}.json")
    if not os.path.exists(info_path):
        return None
    with open(info_path, encoding="utf-8") as handle:
        info = json.load(handle)
    npz_path = os.path.join(raw_dir, f"{patch_id}.npz")
    csv_path = os.path.join(raw_dir, f"{patch_id}.csv")
    if os.path.exists(npz_path):
        with np.load(npz_path) as data:
            return {k: data[k] for k in data.files}, info
    if os.path.exists(csv_path):
        return table_to_bands(pd.read_csv(csv_path), grid, info["bands"]), info
    return None
