"""Patch planning: positives (DETER polygons or Hansen+RADD points), BraDD-like negatives, block split."""
from __future__ import annotations

import datetime as dt

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from gee import labels
from gee.dates import Window, parse_date
from gee.grid import area_ha, lonlat_bounds, make_grid
from gee.sampling import (allocate_negatives, assign_block, block_name, names_at, parse_sample_points,
                          sample_negative_dates, split_of_block, thin_by_distance)
from gee.spec import PatchSpec
from gee.vectors import filter_events, read_deter


def period(cfg: dict) -> Window:
    """The event period from the config."""
    return Window(parse_date(cfg["events"]["period_start"]), parse_date(cfg["events"]["period_end"]))


def load_deter(cfg: dict) -> gpd.GeoDataFrame:
    """Read the local DETER file named in the config."""
    deter = cfg["events"]["deter"]
    return read_deter(deter["path"], deter["date_attribute"], deter["class_attribute"])


def deter_positives(cfg: dict, events: gpd.GeoDataFrame, bbox: list[float]) -> list[PatchSpec]:
    """One positive spec per DETER polygon of a positive class in the period and bbox, centred on its centroid."""
    chosen = filter_events(events, cfg["events"]["deter"]["positive_classes"], period(cfg).start, period(cfg).end,
                           bbox)
    specs = []
    for i, row in enumerate(chosen.itertuples()):
        centre = row.geometry.centroid
        specs.append(PatchSpec(f"{cfg['name']}_pos_{i:06d}", "positive", centre.x, centre.y, row.event_date,
                               row.event_class, area_ha(row.geometry), "", "", "", i, i, row.geometry.wkt))
    return specs


def point_specs(cfg: dict, points: list[dict], sampling_type: str, dates: list[dt.date], start_idx: int,
                deter_class: str) -> list[PatchSpec]:
    """Build specs for sampled points of one type with given event dates."""
    return [PatchSpec(f"{cfg['name']}_{sampling_type}_{start_idx + i:06d}", sampling_type, p["lon"], p["lat"], d,
                      deter_class, 0.0, cfg["region_label"], "", "", -1, start_idx + i, "")
            for i, (p, d) in enumerate(zip(points, dates))]


def finalize(cfg: dict, specs: list[PatchSpec]) -> list[PatchSpec]:
    """Assign region blocks and splits, drop patches crossing a block boundary, thin near-duplicate centres."""
    split_cfg = cfg["split"]
    kept = []
    for spec in specs:
        grid = make_grid(spec.lon, spec.lat, cfg["grid"]["patch_size"], cfg["grid"]["pixel_size_m"])
        block = assign_block(lonlat_bounds(grid), split_cfg["block_size_deg"])
        if block is None:
            continue
        spec.region_block = block_name(block)
        spec.split = split_of_block(block, split_cfg["seed"], split_cfg["fractions"])
        kept.append(spec)
    order = thin_by_distance([s.lon for s in kept], [s.lat for s in kept], split_cfg["min_center_distance_m"])
    return [kept[i] for i in order]


def shuffle_cap(specs: list[PatchSpec], limit: int, seed: int) -> list[PatchSpec]:
    """Seeded shuffle then keep at most `limit` specs."""
    order = np.random.default_rng(seed).permutation(len(specs))
    return [specs[int(i)] for i in order[:limit]]


def drop_near_events(specs: list[PatchSpec], events: gpd.GeoDataFrame, cfg: dict) -> list[PatchSpec]:
    """Drop negatives whose footprint touches any labelled DETER event in the period widened by the window."""
    start = period(cfg).start - dt.timedelta(days=cfg["window"]["days_before"])
    end = period(cfg).end + dt.timedelta(days=cfg["window"]["days_after"])
    nearby = filter_events(events, cfg["events"]["deter"]["label_classes"], start, end)
    kept = []
    for spec in specs:
        footprint = box(*lonlat_bounds(spec.grid(cfg["grid"])))
        if len(nearby.sindex.query(footprint, predicate="intersects")) == 0:
            kept.append(spec)
    return kept


def sample_points(cfg: dict, mode: dict, counts: dict[str, int], n_positive: int) -> list[dict]:
    """Stratified-sample class points in Earth Engine (negatives, plus Hansen-loss positives if requested)."""
    neg = cfg["negatives"]
    codes = labels.class_codes(neg)
    request = {codes[k]: v * neg["oversample"] for k, v in counts.items() if v > 0}
    if n_positive > 0:
        request[labels.POSITIVE_CODE] = n_positive * neg["oversample"]
    image = labels.class_image(neg, cfg["labels"], period(cfg), n_positive > 0)
    info = labels.sample_class_points(image, mode["bbox"], request, neg["sample_scale_m"], cfg["split"]["seed"],
                                      neg["tile_scale"])
    return parse_sample_points(info, labels.CLASS_BAND, labels.HANSEN_LOSS)


def hansen_positives(cfg: dict, points: list[dict]) -> list[PatchSpec]:
    """Positive specs from Hansen-loss points; event date = 31 Dec of the centre pixel's loss year (causal-safe)."""
    chosen = [p for p in points if p["code"] == labels.POSITIVE_CODE]
    base = cfg["labels"]["hansen"]["base_year"]
    dates = [dt.date(base + int(p["extra"]), 12, 31) for p in chosen]
    specs = point_specs(cfg, chosen, "positive", dates, 0, cfg["events"]["positive_class_label"])
    for i, spec in enumerate(specs):
        spec.alert_idx = i
    return specs


def assign_states(cfg: dict, specs: list[PatchSpec], bbox: list[float]) -> None:
    """Set every spec's `state` from one polygon layer (same vocabulary for positives and negatives)."""
    states = cfg["states"]
    if states is None:
        for spec in specs:
            spec.state = cfg["region_label"]
        return
    info = labels.state_features(states, bbox)
    names = names_at([s.lon for s in specs], [s.lat for s in specs], info, states["name_property"],
                     cfg["region_label"])
    for spec, name in zip(specs, names):
        spec.state = name


def negatives(cfg: dict, points: list[dict], positives: list[PatchSpec], counts: dict[str, int]) -> list[PatchSpec]:
    """Negative specs per type, with event dates drawn from the positive date distribution."""
    codes = labels.class_codes(cfg["negatives"])
    out: list[PatchSpec] = []
    for offset, (name, n) in enumerate(sorted(counts.items())):
        chosen = [p for p in points if p["code"] == codes[name]]
        dates = sample_negative_dates([s.event_date for s in positives], len(chosen), cfg["split"]["seed"] + offset)
        out += point_specs(cfg, chosen, name, dates, len(out), "")
    return out


def plan_patches(cfg: dict, mode: dict) -> list[PatchSpec]:
    """Plan all patches of a run: positives, negatives (per type), split by region blocks, capped per mode."""
    seed, neg, limit = cfg["split"]["seed"], cfg["negatives"], mode["max_positives"]
    deter = load_deter(cfg) if cfg["events"]["source"] == "deter" else None
    positives = [] if deter is None else shuffle_cap(finalize(cfg, deter_positives(cfg, deter, mode["bbox"])),
                                                     limit, seed)
    n_pos = limit if deter is None else len(positives)
    counts = allocate_negatives(n_pos, neg["ratio"], neg["type_weights"])
    points = sample_points(cfg, mode, counts, n_pos if deter is None else 0)
    if deter is None:
        positives = shuffle_cap(finalize(cfg, hansen_positives(cfg, points)), limit, seed)
    if not positives:
        raise RuntimeError("no positive patches found in the configured period and bbox")
    candidates = finalize(cfg, negatives(cfg, points, positives, counts))
    if deter is not None:
        candidates = drop_near_events(candidates, deter, cfg)
    by_type = [shuffle_cap([s for s in candidates if s.sampling_type == k], n, seed) for k, n in counts.items()]
    chosen = positives + [s for group in by_type for s in group]
    spaced = [chosen[i] for i in thin_by_distance([s.lon for s in chosen], [s.lat for s in chosen],
                                                  cfg["split"]["min_center_distance_m"])]
    assign_states(cfg, spaced, mode["bbox"])
    return spaced
