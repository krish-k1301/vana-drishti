"""Tests for pilot QA: good samples pass; NaNs, linear scale, wrong labels and missing keys are caught; pilot gate."""
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from gee.config import load_config
from gee.qa import check_sample, qa_passed, random_qa_sample, run_qa
from gee.qa_report import pilot_passed, write_qa_outputs
from gee.to_bradd import build_sample, save_sample, write_meta

QA = load_config("configs/gee/amazon_dated.yaml")["qa"]


def good_sample(positive: bool = True, n_dates: int = 30) -> dict:
    """A plausible dB sample with a change (positive) or none (negative)."""
    rng = np.random.default_rng(0)
    image = np.stack([rng.normal(-8, 1, (n_dates, 48, 48)), rng.normal(-14, 1, (n_dates, 48, 48))], axis=1)
    dates = [dt.date(2021, 1, 1) + dt.timedelta(days=12 * i) for i in range(n_dates)]
    label = np.zeros((3, 48, 48), dtype=np.int64)
    mask = np.zeros((48, 48), dtype=np.uint8)
    if positive:
        label[2, 10:20, 10:20] = 1
        mask[10:20, 10:20] = 1
    extras = {"event_date": dt.date(2021, 6, 1) if positive else None, "deter_class": "DESMATAMENTO_CR",
              "burn_month": np.full((48, 48), -1), "radd_date": np.full((48, 48), -1), "prodes_year": -1,
              "polygon_area_ha": 1.0, "event_mask": mask, "region_block": "b0_0"}
    return build_sample(image, dates, [dt.date(2021, 1, 1), dt.date(2021, 6, 1), dt.date(2021, 7, 1)], label, extras)


def row(sampling_type: str = "positive", orbit: object = 10, gap: float = 12.0) -> pd.Series:
    """meta.csv-like row."""
    return pd.Series({"sampling_type": sampling_type, "relative_orbit": orbit, "gap_max_days": gap})


def test_good_samples_pass_every_check():
    assert all(check_sample(good_sample(), row(), QA).values())
    assert all(check_sample(good_sample(False), row("forest"), QA).values())


def test_bad_inputs_are_caught():
    nan = good_sample()
    nan["image"][0, 0, 0, 0] = float("nan")
    assert not check_sample(nan, row(), QA)["finite"]
    linear = good_sample()
    linear["image"] = 10 ** (linear["image"] / 10)
    assert not check_sample(linear, row(), QA)["db_scale"]
    bright = good_sample()
    bright["image"][:, 0] += 30
    assert not check_sample(bright, row(), QA)["vv_range"]
    assert not check_sample(good_sample(), row("forest"), QA)["label_change"]
    assert not check_sample(good_sample(False), row("positive"), QA)["label_change"]
    assert not check_sample(good_sample(n_dates=5), row(), QA)["t_range"]
    assert not check_sample(good_sample(), row(gap=60.0), QA)["revisit_gap"]
    for orbit in (float("nan"), "", "10+83"):
        assert not check_sample(good_sample(), row(orbit=orbit), QA)["single_orbit"]
    missing = good_sample()
    del missing["radd_date"]
    assert check_sample(missing, row(), QA) == {"format": False}
    wrong = good_sample()
    wrong["label"] = wrong["label"].int()
    assert not check_sample(wrong, row(), QA)["format"]


def write_dataset(root, samples: list[tuple[dict, str]]) -> None:
    """Write samples + a minimal meta.csv."""
    rows = []
    for i, (sample, kind) in enumerate(samples):
        name = f"p{i}_2021-06-01.pt"
        save_sample(str(root), name, sample)
        rows.append({"patch_id": f"p{i}", "file": name, "sampling_type": kind, "relative_orbit": 10,
                     "gap_max_days": 12.0, "lon": -55.0, "lat": -7.0, "event_date": "2021-06-01",
                     "deter_class": "DESMATAMENTO_CR", "dated_set": "train"})
    write_meta(str(root), rows)


def test_run_qa_report_and_pilot_gate(tmp_path):
    cfg_qa = {**QA, "min_positives": 2}
    write_dataset(tmp_path, [(good_sample(), "positive")] * 3 + [(good_sample(False), "forest")])
    table = run_qa(str(tmp_path), cfg_qa)
    passed, reasons = qa_passed(table, cfg_qa)
    assert passed and not reasons and table["ok"].all()
    spot = random_qa_sample(str(tmp_path), 2, seed=0)
    assert len(spot) == 2 and {"lon", "lat", "event_date"} <= set(spot.columns)
    conversion = {"converted": 4, "rejected": {"x": "negative_has_change"}, "missing": []}
    paths = write_qa_outputs(str(tmp_path / "qa"), table, passed, reasons, conversion, spot, "abc")
    assert "Verdict: PASSED" in Path(paths["qa_report.md"]).read_text()
    assert pilot_passed(str(tmp_path / "qa"), "abc")[0]
    assert not pilot_passed(str(tmp_path / "qa"), "other")[0]
    assert not pilot_passed(str(tmp_path / "nothing"), "abc")[0]


def test_failed_pilot_blocks_full_run(tmp_path):
    bad = good_sample()
    bad["image"] = torch.full_like(bad["image"], float("nan"))
    write_dataset(tmp_path, [(bad, "positive")])
    table = run_qa(str(tmp_path), QA)
    passed, reasons = qa_passed(table, QA)
    assert not passed and any("failed a check" in r for r in reasons)
    write_qa_outputs(str(tmp_path / "qa"), table, passed, reasons, {"converted": 1, "rejected": {}, "missing": []},
                     random_qa_sample(str(tmp_path), 1, 0), "abc")
    ok, reason = pilot_passed(str(tmp_path / "qa"), "abc")
    assert not ok and "did not pass" in reason
