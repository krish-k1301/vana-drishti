"""Turn raw pulled patches into BraDD-format samples: S1 series, dated cumulative labels and dated extra keys."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import geopandas as gpd
import numpy as np
from shapely import wkt

from gee.dates import monthly_label_dates, parse_date, revisit_stats
from gee.decode import burn_days, hansen_lossyear, modis_flag, radd_days, s1_series
from gee.grid import PatchGrid, rasterize
from gee.spec import PatchSpec
from gee.export import load_raw
from gee.to_bradd import (build_sample, cumulative_labels, hansen_reference, sample_file_name, save_sample,
                          write_meta)
from gee.vectors import filter_events, prior_mask, prodes_year_of, read_deter, read_prodes, reference_days

ALL_DATES = (dt.date(1900, 1, 1), dt.date(2100, 1, 1))


@dataclass
class Vectors:
    """Local label vectors: DETER events of the label classes and PRODES yearly polygons (None if unused)."""

    events: gpd.GeoDataFrame | None
    prodes: gpd.GeoDataFrame | None


def load_vectors(cfg: dict) -> Vectors:
    """Read the local DETER/PRODES files for DETER-sourced configs; empty for Hansen-sourced configs."""
    if cfg["events"]["source"] != "deter":
        return Vectors(None, None)
    deter = cfg["events"]["deter"]
    events = read_deter(deter["path"], deter["date_attribute"], deter["class_attribute"])
    prodes_cfg = cfg["events"]["prodes"]
    return Vectors(filter_events(events, deter["label_classes"], *ALL_DATES),
                   read_prodes(prodes_cfg["path"], prodes_cfg["year_attribute"]))


def deter_labels(cfg: dict, spec: PatchSpec, grid: PatchGrid, vectors: Vectors) -> tuple:
    """(reference days, prior mask, event mask, PRODES year, area ha) from DETER polygons and PRODES."""
    window = spec.window(cfg["window"])
    prodes_cfg = cfg["events"]["prodes"]
    ref = reference_days(vectors.events, grid)
    prior = prior_mask(vectors.prodes, grid, window.start, prodes_cfg["year_end_month"], prodes_cfg["year_end_day"])
    if not spec.is_positive:
        return ref, prior, np.zeros(prior.shape, dtype=np.uint8), -1, 0.0
    polygon = wkt.loads(spec.event_wkt)
    mask = rasterize([polygon], grid).astype(np.uint8)
    return ref, prior, mask, prodes_year_of(vectors.prodes, polygon), spec.polygon_area_ha


def hansen_labels(cfg: dict, spec: PatchSpec, grid: PatchGrid, loss: np.ndarray) -> tuple:
    """(reference days, prior mask, event mask, PRODES year -1, area ha) from Hansen loss alone (no RADD)."""
    base = cfg["labels"]["hansen"]["base_year"]
    ref = hansen_reference(loss, base)
    prior = np.zeros(ref.shape, dtype=bool)
    if not spec.is_positive:
        return ref, prior, np.zeros(ref.shape, dtype=np.uint8), -1, 0.0
    mask = (loss == spec.event_date.year - base).astype(np.uint8)
    return ref, prior, mask, -1, float(mask.sum()) * grid.pixel_m ** 2 / 10_000.0


def convert_patch(cfg: dict, spec: PatchSpec, bands: dict, info: dict, vectors: Vectors) -> tuple[dict, dict] | str:
    """Return (sample dict, meta row) for one pulled patch, or a rejection reason string."""
    grid, window = spec.grid(cfg["grid"]), spec.window(cfg["window"])
    dates = [parse_date(d) for d in info["image_dates"]]
    image, keep = s1_series(bands, len(dates), cfg["s1"]["polarisations"], cfg["s1"]["nodata"])
    if not keep:
        return "no_complete_s1_dates"
    loss = hansen_lossyear(bands, (grid.size, grid.size))
    if vectors.events is not None:
        ref, prior, mask, prodes_year, area = deter_labels(cfg, spec, grid, vectors)
    else:
        ref, prior, mask, prodes_year, area = hansen_labels(cfg, spec, grid, loss)
    label_dates = monthly_label_dates(window)
    label = cumulative_labels(ref, prior, label_dates)
    if spec.is_positive and not mask.any():
        return "positive_empty_event_mask"
    if not spec.is_positive and (label[-1] != label[0]).any():
        return "negative_has_change"
    hansen_ref = hansen_reference(loss, cfg["labels"]["hansen"]["base_year"])
    extras = {"event_date": spec.event_date if spec.is_positive else None, "deter_class": spec.deter_class,
              "burn_month": burn_days(bands, window, (grid.size, grid.size)),
              "radd_date": radd_days(bands, window, cfg["labels"]["radd"]["base_year"]),
              "prodes_year": prodes_year, "polygon_area_ha": area, "event_mask": mask,
              "region_block": spec.region_block, "ref_day": ref,
              "label_hansen": cumulative_labels(hansen_ref, np.zeros(ref.shape, dtype=bool), label_dates)}
    kept_dates = [dates[i] for i in keep]
    sample = build_sample(image, kept_dates, label_dates, label, extras)
    return sample, meta_row(spec, info, revisit_stats(kept_dates), modis_flag(bands))


def meta_row(spec: PatchSpec, info: dict, stats: dict, modis: int) -> dict:
    """One meta.csv row: BraDD columns (close_set mirrors dated_set), dated columns and provenance extras."""
    return {"alert_idx": spec.alert_idx, "center_idx": spec.center_idx, "date": spec.event_date.isoformat(),
            "sampling_type": spec.sampling_type, "state": spec.state,
            "file": sample_file_name(spec.patch_id, spec.event_date), "close_set": spec.split,
            "event_date": spec.event_date.isoformat() if spec.is_positive else "", "deter_class": spec.deter_class,
            "region_block": spec.region_block, "dated_set": spec.split, "patch_id": spec.patch_id,
            "lon": spec.lon, "lat": spec.lat, "relative_orbit": info["relative_orbit"],
            "orbit_pass": info["orbit_pass"], "platforms": "+".join(info["platforms"]),
            "n_dates": int(stats["n_dates"]), "gap_median_days": stats["gap_median_days"],
            "gap_max_days": stats["gap_max_days"], "modis_burn_flag": modis}


def convert_all(cfg: dict, specs: list[PatchSpec], raw_dir: str, out_root: str) -> dict:
    """Convert every pulled patch into `<out_root>/Samples` + `meta.csv`; returns counts and rejection reasons."""
    vectors = load_vectors(cfg)
    rows, rejected, missing = [], {}, []
    for spec in specs:
        raw = load_raw(raw_dir, spec.patch_id, spec.grid(cfg["grid"]))
        if raw is None:
            missing.append(spec.patch_id)
            continue
        result = convert_patch(cfg, spec, raw[0], raw[1], vectors)
        if isinstance(result, str):
            rejected[spec.patch_id] = result
            continue
        sample, row = result
        save_sample(out_root, row["file"], sample)
        rows.append(row)
    write_meta(out_root, rows)
    return {"converted": len(rows), "rejected": rejected, "missing": missing}
