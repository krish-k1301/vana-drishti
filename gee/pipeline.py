"""Per-patch Earth Engine export: S1 query, orbit choice, image stack, then direct pull or batch submission."""
from __future__ import annotations

import ee

from gee import labels, s1
from gee.dates import revisit_stats
from gee.export import pull_direct, refresh_batch, save_raw, submit_batch
from gee.spec import PatchSpec
from gee.tasklog import COMPLETED, FAILED, TaskLog


def label_band_names(labels_cfg: dict, years: list[int]) -> list[str]:
    """Names of the pulled label bands, in the order label_image concatenates them."""
    names = {"radd": [labels.RADD_ALERT, labels.RADD_DATE], "hansen": [labels.HANSEN_LOSS],
             "fogo": [labels.fogo_band_name(y) for y in years], "modis": [labels.MODIS_BURN]}
    return [band for source in labels_cfg["pull"] for band in names[source]]


def build_patch_image(cfg: dict, spec: PatchSpec) -> tuple[object, dict]:
    """Build the per-patch ee.Image (S1 stack + label bands) and the sidecar info describing it."""
    grid, window = spec.grid(cfg["grid"]), spec.window(cfg["window"])
    s1_cfg = cfg["s1"]
    region = labels.patch_region(grid)
    collection = s1.s1_collection(s1_cfg, region, window.start, window.end)
    acquisitions = s1.fetch_acquisitions(collection)
    orbit, orbit_pass = s1.select_orbit(acquisitions, s1_cfg["pass_preference"])
    dates = s1.orbit_dates(acquisitions, orbit, orbit_pass)
    image = s1.s1_stack(collection, orbit, orbit_pass, dates, s1_cfg["polarisations"], s1_cfg["nodata"])
    image = image.addBands(labels.label_image(cfg["labels"], cfg["labels"]["pull"], region, window))
    bands = s1.s1_band_names(len(dates), s1_cfg["polarisations"]) + label_band_names(cfg["labels"], window.years)
    info = {"patch_id": spec.patch_id, "image_dates": [d.isoformat() for d in dates], "relative_orbit": orbit,
            "orbit_pass": orbit_pass, "platforms": s1.orbit_platforms(acquisitions, orbit, orbit_pass),
            "n_scenes_all_orbits": len(acquisitions), "bands": bands, "crs": grid.crs, "affine": grid.affine,
            "window": [window.start.isoformat(), window.end.isoformat()], **revisit_stats(dates)}
    return image, info


def export_patch(cfg: dict, spec: PatchSpec, raw_dir: str, log: TaskLog, backend: str) -> str:
    """Export one patch with the chosen backend, logging every state change; returns the logged state."""
    try:
        image, info = build_patch_image(cfg, spec)
        if backend == "direct":
            log.append(spec.patch_id, "RUNNING", backend)
            save_raw(raw_dir, spec.patch_id, pull_direct(image, spec.grid(cfg["grid"])), info)
            log.append(spec.patch_id, COMPLETED, backend)
            return COMPLETED
        save_raw(raw_dir, spec.patch_id, None, info)
        task_id = submit_batch(image, spec.grid(cfg["grid"]), spec.patch_id, info["bands"], cfg["export"]["batch"])
        log.append(spec.patch_id, "SUBMITTED", backend, task_id)
        return "SUBMITTED"
    except (ee.EEException, ValueError, KeyError) as error:
        log.append(spec.patch_id, FAILED, backend, info=f"{type(error).__name__}: {error}")
        return FAILED


def export_all(cfg: dict, specs: list[PatchSpec], raw_dir: str, log: TaskLog, backend: str) -> dict[str, int]:
    """Export every spec not already done/pending (resume-safe); returns counts per resulting state."""
    if backend == "batch":
        refresh_batch(log)
    limit = cfg["export"]["max_pending_tasks"]
    counts: dict[str, int] = {}
    for spec in specs:
        if log.should_skip(spec.patch_id, cfg["export"]["max_attempts"]):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        if backend == "batch" and len(log.pending_tasks()) >= limit:
            counts["deferred"] = counts.get("deferred", 0) + 1
            continue
        state = export_patch(cfg, spec, raw_dir, log, backend)
        counts[state] = counts.get(state, 0) + 1
    return counts
