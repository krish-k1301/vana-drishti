"""Pure sampling logic: region-block split, boundary policy, spacing thinning, negative dates and counts."""
from __future__ import annotations

import datetime as dt
import hashlib
import math

import numpy as np

SPLITS = ("train", "validation", "test")
EARTH_RADIUS_M = 6_371_000.0
DEG_M = 111_000.0


def block_index(lon: float, lat: float, size_deg: float) -> tuple[int, int]:
    """Return the (ix, iy) index of the size_deg x size_deg block containing (lon, lat)."""
    return int(math.floor(lon / size_deg)), int(math.floor(lat / size_deg))


def block_name(index: tuple[int, int]) -> str:
    """Return a stable string name for a block index."""
    return f"b{index[0]}_{index[1]}"


def split_of_block(index: tuple[int, int], seed: int, fractions: dict[str, float]) -> str:
    """Assign a block to train/validation/test by a seeded SHA-256 hash; deterministic across runs."""
    total = sum(fractions[s] for s in SPLITS)
    if total <= 0:
        raise ValueError("split fractions must sum to a positive number")
    digest = hashlib.sha256(f"{seed}:{index[0]}:{index[1]}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    edge = 0.0
    for split in SPLITS:
        edge += fractions[split] / total
        if u < edge:
            return split
    return SPLITS[-1]


def assign_block(bounds: tuple[float, float, float, float], size_deg: float) -> tuple[int, int] | None:
    """Return the block that fully contains a lon/lat footprint, or None if it crosses a block boundary (dropped)."""
    lo = block_index(bounds[0], bounds[1], size_deg)
    hi = block_index(bounds[2], bounds[3], size_deg)
    return lo if lo == hi else None


def thin_by_distance(lons: list[float], lats: list[float], min_dist_m: float) -> list[int]:
    """Greedily keep points (in the given order) at least min_dist_m from every kept point; return kept indices."""
    if min_dist_m <= 0:
        return list(range(len(lons)))
    max_lat = min(max((abs(v) for v in lats), default=0.0), 80.0)
    cell_deg = min_dist_m / (DEG_M * math.cos(math.radians(max_lat)))
    buckets: dict[tuple[int, int], list[int]] = {}
    kept: list[int] = []
    for i, (lon, lat) in enumerate(zip(lons, lats)):
        cx, cy = int(math.floor(lon / cell_deg)), int(math.floor(lat / cell_deg))
        near = [j for dx in (-1, 0, 1) for dy in (-1, 0, 1) for j in buckets.get((cx + dx, cy + dy), [])]
        if all(_distance_m(lon, lat, lons[j], lats[j]) >= min_dist_m for j in near):
            kept.append(i)
            buckets.setdefault((cx, cy), []).append(i)
    return kept


def _distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Equirectangular distance in metres (accurate at patch scale)."""
    x = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    y = math.radians(lat2 - lat1)
    return EARTH_RADIUS_M * math.hypot(x, y)


def sample_negative_dates(positive_dates: list[dt.date], n: int, seed: int) -> list[dt.date]:
    """Draw n negative event dates with replacement from the positive date distribution."""
    if not positive_dates:
        raise ValueError("negative dates are drawn from positives; no positive dates given")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(positive_dates), size=n)
    return [positive_dates[int(i)] for i in picks]


def allocate_negatives(n_positive: int, ratio: float, type_weights: dict[str, float]) -> dict[str, int]:
    """Split round(ratio * n_positive) negatives across sampling types proportionally to their weights."""
    total = int(round(ratio * n_positive))
    weight_sum = sum(type_weights.values())
    counts = {k: int(math.floor(total * w / weight_sum)) for k, w in type_weights.items()}
    remainder = total - sum(counts.values())
    for name in sorted(type_weights, key=lambda k: -type_weights[k])[:remainder]:
        counts[name] += 1
    return counts


def parse_sample_points(info: dict, class_band: str, extra_band: str | None = None) -> list[dict]:
    """Parse a `FeatureCollection.getInfo()` of sampled points into dicts with lon, lat, code (and extra)."""
    points = []
    for feature in info.get("features", []):
        lon, lat = feature["geometry"]["coordinates"][:2]
        props = feature.get("properties", {})
        point = {"lon": float(lon), "lat": float(lat), "code": int(props[class_band])}
        if extra_band is not None:
            point["extra"] = props.get(extra_band, 0)
        points.append(point)
    return points
