"""Round-trip tests: synthetic pulled bands -> BraDD-format samples + meta.csv -> torch.load / DatedDataset."""
import datetime as dt

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import box

from gee.config import load_config
from gee.convert import Vectors, convert_all, convert_patch, load_vectors
from gee.dates import encode_yyddd, from_days, to_days
from gee.export import save_raw
from gee.grid import lonlat_bounds
from gee.labels import HANSEN_LOSS, MODIS_BURN, RADD_ALERT, RADD_DATE, fogo_band_name
from gee.s1 import s1_band_names
from gee.spec import PatchSpec
from gee.to_bradd import EXTRA_COLUMNS, META_COLUMNS, cumulative_labels

EVENT = dt.date(2022, 6, 1)
HANSEN_EVENT = dt.date(2022, 12, 31)


def make_spec(sampling_type: str, lon: float, lat: float, event: dt.date = EVENT, wkt: str = "") -> PatchSpec:
    """Spec for a synthetic patch."""
    return PatchSpec(f"t_{sampling_type}", sampling_type, lon, lat, event, "hansen_loss", 0.0, "congo",
                     "b50_1", "train", 0, 0, wkt)


def s1_bands(n_dates: int, nodata_date: int | None = None) -> dict:
    """Plausible dB backscatter for n dates; optionally one date with a nodata pixel."""
    rng = np.random.default_rng(1)
    bands = {}
    for name in s1_band_names(n_dates, ["VV", "VH"]):
        bands[name] = rng.normal(-8.0 if name.endswith("VV") else -14.0, 1.0, (48, 48))
    if nodata_date is not None:
        bands[f"t{nodata_date:03d}_VV"][3, 3] = -9999.0
    return bands


def info_for(n_dates: int, start: dt.date = dt.date(2021, 6, 5)) -> dict:
    """Sidecar info with dates every 12 days from `start`."""
    dates = [start + dt.timedelta(days=12 * i) for i in range(n_dates)]
    return {"image_dates": [d.isoformat() for d in dates], "relative_orbit": 50, "orbit_pass": "DESCENDING",
            "platforms": ["A"], "gap_median_days": 12.0, "gap_max_days": 12.0}


def hansen_bands() -> dict:
    """Left half lost in 2022; a RADD alert (2022-06-10) on the top half, also over pixels Hansen never lost."""
    loss = np.zeros((48, 48))
    loss[:, :24] = 22
    alert, date = np.zeros((48, 48)), np.zeros((48, 48))
    alert[:24, :], date[:24, :] = 3, encode_yyddd(dt.date(2022, 6, 10))
    return {HANSEN_LOSS: loss, RADD_ALERT: alert, RADD_DATE: date, MODIS_BURN: np.zeros((48, 48))}


def test_cumulative_labels_include_prior_and_dated_pixels():
    ref = np.array([[-1, to_days(dt.date(2022, 3, 5))]], dtype=np.int32)
    prior = np.array([[True, False]])
    label = cumulative_labels(ref, prior, [dt.date(2022, 3, 1), dt.date(2022, 4, 1)])
    assert label.dtype == np.int64 and label.tolist() == [[[1, 0]], [[1, 1]]]


def test_hansen_positive_is_dated_at_year_end_and_ignores_radd(tmp_path):
    cfg = load_config("configs/gee/congo_pilot.yaml")
    spec = make_spec("positive", 25.3, 0.9, HANSEN_EVENT)
    result = convert_patch(cfg, spec, {**s1_bands(40, nodata_date=5), **hansen_bands()},
                           info_for(40, dt.date(2022, 1, 2)), Vectors(None, None))
    sample, row = result
    torch.save(sample, tmp_path / "s.pt")
    loaded = torch.load(tmp_path / "s.pt", weights_only=False)
    assert loaded["image"].dtype == torch.float32 and loaded["image"].shape == (39, 2, 48, 48)
    assert all(isinstance(d, dt.date) for d in loaded["image_dates"] + loaded["label_dates"])
    assert loaded["label"].dtype == torch.int64 and loaded["label"].shape[0] == len(loaded["label_dates"])
    assert loaded["burn_month"].dtype == torch.int16 and (loaded["burn_month"] == -1).all()
    assert loaded["radd_date"].dtype == torch.int32 and loaded["event_mask"].dtype == torch.uint8
    assert loaded["ref_day"].dtype == torch.int32 and loaded["label_hansen"].dtype == torch.int64
    assert loaded["event_date"] == HANSEN_EVENT and loaded["prodes_year"] == -1
    assert loaded["polygon_area_ha"] == 0.5 * 48 * 48 * 0.01
    year_end = to_days(HANSEN_EVENT)
    assert (loaded["ref_day"][:, :24] == year_end).all() and (loaded["ref_day"][:, 24:] == -1).all()
    for k, day in enumerate(loaded["label_dates"]):
        expected = day >= HANSEN_EVENT
        assert bool(loaded["label"][k][:, :24].all()) == expected and not loaded["label"][k][:, 24:].any()
    assert torch.equal(loaded["label"], loaded["label_hansen"])
    assert from_days(int(loaded["radd_date"][0, 30])) == dt.date(2022, 6, 10)
    assert row["n_dates"] == 39 and row["dated_set"] == row["close_set"] == "train"


def test_hansen_negative_with_change_is_rejected_and_clean_negative_kept():
    cfg = load_config("configs/gee/congo_pilot.yaml")
    changing = make_spec("forest", 25.3, 0.9, dt.date(2023, 3, 1))
    assert convert_patch(cfg, changing, {**s1_bands(10), **hansen_bands()}, info_for(10), Vectors(None, None)) == \
        "negative_has_change"
    clean = {**hansen_bands(), HANSEN_LOSS: np.zeros((48, 48))}
    sample, row = convert_patch(cfg, changing, {**s1_bands(10), **clean}, info_for(10), Vectors(None, None))
    assert sample["event_date"] is None and row["event_date"] == ""


def deter_inputs(tmp_path, cfg: dict, spec: PatchSpec) -> tuple:
    """Write DETER (central + neighbouring polygon) and PRODES files; return (central, neighbour) geometries."""
    lon0, lat0, lon1, lat1 = lonlat_bounds(spec.grid(cfg["grid"]))
    mid_lon, mid_lat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    central = box(lon0 - 0.01, lat0 - 0.01, mid_lon, mid_lat)
    neighbour = box(mid_lon + 1e-4, lat0 - 0.01, lon1 + 0.01, mid_lat)
    deter = gpd.GeoDataFrame({"view_date": [EVENT.isoformat(), "2022-08-15"],
                              "classname": ["DESMATAMENTO_CR", "DESMATAMENTO_VEG"]},
                             geometry=[central, neighbour], crs=4326)
    prodes_box = box(mid_lon - 1e-3, mid_lat - 1e-3, lon1 + 0.01, lat1 + 0.01)
    prodes = gpd.GeoDataFrame({"year": [2019.0]}, geometry=[prodes_box], crs=4326)
    deter.to_file(tmp_path / "deter.geojson")
    prodes.to_file(tmp_path / "prodes.geojson")
    cfg["events"]["deter"]["path"] = str(tmp_path / "deter.geojson")
    cfg["events"]["prodes"]["path"] = str(tmp_path / "prodes.geojson")
    return central, neighbour


def test_deter_path_ref_day_neighbours_label_hansen_and_dated_loader(tmp_path):
    cfg = load_config("configs/gee/amazon_dated.yaml")
    spec = make_spec("positive", -55.4, -7.0)
    central, _ = deter_inputs(tmp_path, cfg, spec)
    spec.event_wkt, spec.deter_class = central.wkt, "DESMATAMENTO_CR"
    fogo = {fogo_band_name(y): np.full((48, 48), 7 if y == 2022 else 0) for y in (2021, 2022)}
    root = tmp_path / "dated"
    save_raw(str(root / "raw"), spec.patch_id, {**s1_bands(20), **hansen_bands(), **fogo}, info_for(20))
    summary = convert_all(cfg, [spec], str(root / "raw"), str(root))
    assert summary["converted"] == 1, summary
    meta = pd.read_csv(root / "meta.csv", index_col=0, keep_default_na=False)
    assert list(meta.columns) == META_COLUMNS + EXTRA_COLUMNS
    sample = torch.load(root / "Samples" / meta.loc[0, "file"], weights_only=False)
    assert sample["prodes_year"] == 2019 and sample["label"][0].sum() > 0
    assert (sample["burn_month"] == to_days(dt.date(2022, 7, 31))).all()
    ref, mask = sample["ref_day"], sample["event_mask"].bool()
    assert (ref[mask] == to_days(EVENT)).all()
    neighbour = (ref == to_days(dt.date(2022, 8, 15)))
    assert neighbour.any() and not (neighbour & mask).any()
    assert sample["label"][-1][neighbour].all() and sample["label"][-1][mask].all()
    assert sample["label_hansen"].shape == sample["label"].shape and sample["label_hansen"].sum() == 0
    assert load_vectors(cfg).events is not None
    from src.data.dated_dataset import DatedDataset
    item = DatedDataset(root, "train", normalization="none")[0]
    assert item["Images"].shape[1:] == (2, 48, 48) and int(item["EventDay"]) > 0
