"""The synthetic fixture matches the BraDD on-disk format of INTERFACES.md."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import torch

from src.data.samples import load_sample, read_meta

BASE_COLUMNS = ["alert_idx", "center_idx", "date", "sampling_type", "state", "file", "close_set"]
DATED_COLUMNS = BASE_COLUMNS + ["event_date", "deter_class", "region_block", "dated_set", "patch_id", "lon",
                                 "lat", "relative_orbit", "orbit_pass", "platforms", "n_dates", "gap_median_days",
                                 "gap_max_days", "modis_burn_flag"]


def test_meta_columns_and_splits(bradd_root):
    """meta.csv has the seven BraDD columns, split counts and file names."""
    meta = pd.read_csv(bradd_root / "meta.csv", index_col=0)
    assert list(meta.columns) == BASE_COLUMNS
    assert meta["close_set"].value_counts().to_dict() == {"train": 8, "validation": 4, "test": 4}
    assert meta["file"].str.match(r"^\d{7}_\d{4}-\d{2}-\d{2}\.pt$").all()


def test_sample_format(bradd_root):
    """Samples have BraDD keys, dtypes, shapes, dB-like values, variable T and no 1->1."""
    meta = read_meta(bradd_root)
    lengths = set()
    for file in meta["file"]:
        s = load_sample(bradd_root / "Samples" / file)
        t = len(s["image_dates"])
        lengths.add(t)
        assert 19 <= t <= 40 and isinstance(s["image_dates"][0], dt.date)
        assert len(s["label_dates"]) == 2
        assert s["image"].dtype == torch.float32 and s["image"].shape == (t, 2, 48, 48)
        assert s["label"].dtype == torch.int64 and s["label"].shape == (2, 48, 48)
        assert not bool((s["label"][0].bool() & s["label"][1].bool()).any()), "no 1->1 pixels"
        vv, vh = s["image"][:, 0].mean().item(), s["image"][:, 1].mean().item()
        assert -12 < vv < -5 and -20 < vh < -11 and vh < vv
    assert len(lengths) > 1


def test_transitions_present(bradd_root):
    """Fixture labels contain 0->0 > 0->1 > 0, some 1->0 and no 1->1 pixels."""
    meta = read_meta(bradd_root)
    counts = torch.zeros(4, dtype=torch.long)
    for file in meta["file"]:
        label = load_sample(bradd_root / "Samples" / file)["label"]
        counts += torch.bincount((label[1] + 2 * label[0]).flatten(), minlength=4)
    assert counts[0] > counts[1] > 0 and counts[2] > 0 and counts[3] == 0


def test_dated_format(dated_root):
    """Dated fixture has extra columns, block-disjoint splits and cumulative monthly labels."""
    meta = pd.read_csv(dated_root / "meta.csv", index_col=0, keep_default_na=False)
    assert list(meta.columns) == DATED_COLUMNS
    assert set(meta.loc[meta["deter_class"] != "", "deter_class"]) <= {
        "DESMATAMENTO_CR", "DESMATAMENTO_VEG", "CICATRIZ_DE_QUEIMADA"}
    blocks = meta.groupby("dated_set")["region_block"].apply(set)
    assert not (blocks["train"] & blocks["test"]) and not (blocks["train"] & blocks["validation"])
    s = load_sample(dated_root / "Samples" / meta.loc[0, "file"])
    assert s["label"].shape[0] > 2 and len(s["label_dates"]) == s["label"].shape[0]
    assert (s["label"][1:] >= s["label"][:-1]).all(), "cumulative labels"
    assert s["burn_month"].dtype == torch.int16 and s["radd_date"].dtype == torch.int32
    assert s["event_mask"].dtype == torch.uint8
    assert s["label_hansen"].dtype == torch.int64 and s["label_hansen"].shape == s["label"].shape
    span = (s["image_dates"][-1] - s["image_dates"][0]).days
    assert 440 < span < 520
