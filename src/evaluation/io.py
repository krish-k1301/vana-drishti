"""Lightweight output helpers (no model imports): JSON/CSV writers and pixel area."""
from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json(path: Path, payload: dict) -> Path:
    """Write `payload` as indented JSON (numpy/torch scalars converted to float)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=float))
    return path


def write_csv(path: Path, rows: list[dict], header: dict | None = None) -> Path:
    """Write rows as CSV; `header` fields (e.g. the SMOKE label) are prepended to every row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = {k: header[k] for k in ("label", "experiment_name")} if header else {}
    full = [{**prefix, **row} for row in rows]
    fields = list(dict.fromkeys(k for row in full for k in row)) or list(prefix)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(full)
    return path


def pixel_area_ha(pixel_size_m: float) -> float:
    """Area of one square pixel in hectares (1 ha = 10,000 m^2)."""
    return pixel_size_m**2 / 1e4
