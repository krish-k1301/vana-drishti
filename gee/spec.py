"""Pure per-patch specification (what to export and why) with CSV round-trip and derived grid/window."""
from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, fields

import pandas as pd

from gee.dates import Window, compute_window, parse_date
from gee.grid import PatchGrid, make_grid


@dataclass
class PatchSpec:
    """One planned patch: location, event date, provenance and split."""

    patch_id: str
    sampling_type: str
    lon: float
    lat: float
    event_date: dt.date
    deter_class: str
    polygon_area_ha: float
    state: str
    region_block: str
    split: str
    alert_idx: int
    center_idx: int
    event_wkt: str

    @property
    def is_positive(self) -> bool:
        """True for event-centred patches."""
        return self.sampling_type == "positive"

    def grid(self, grid_cfg: dict) -> PatchGrid:
        """Pixel grid of this patch."""
        return make_grid(self.lon, self.lat, grid_cfg["patch_size"], grid_cfg["pixel_size_m"])

    def window(self, window_cfg: dict) -> Window:
        """Time window of this patch around its (real or assigned) event date."""
        return compute_window(self.event_date, window_cfg["days_before"], window_cfg["days_after"])


def save_specs(specs: list[PatchSpec], path: str) -> None:
    """Write specs to CSV (one row per patch)."""
    frame = pd.DataFrame([asdict(s) for s in specs], columns=[f.name for f in fields(PatchSpec)])
    frame.to_csv(path, index=False)


def load_specs(path: str) -> list[PatchSpec]:
    """Read specs written by save_specs."""
    frame = pd.read_csv(path, keep_default_na=False)
    specs = []
    for row in frame.to_dict("records"):
        row["event_date"] = parse_date(row["event_date"])
        for key in ("lon", "lat", "polygon_area_ha"):
            row[key] = float(row[key])
        for key in ("alert_idx", "center_idx"):
            row[key] = int(row[key])
        for key in ("patch_id", "sampling_type", "deter_class", "state", "region_block", "split", "event_wkt"):
            row[key] = str(row[key])
        specs.append(PatchSpec(**row))
    return specs
