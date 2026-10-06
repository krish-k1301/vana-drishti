"""Tests for resume-safe exports: JSONL task log, skip/retry rules, batch polling and raw round-trips (mocked ee)."""
import datetime as dt
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from gee import export, pipeline
from gee.decode import table_to_bands
from gee.grid import _transformer, lonlat_to_rowcol, make_grid
from gee.spec import PatchSpec
from gee.tasklog import TaskLog


def spec(i: int) -> PatchSpec:
    """Minimal patch spec."""
    return PatchSpec(f"p{i}", "positive", -55.4, -7.0, dt.date(2021, 6, 1), "DESMATAMENTO_CR", 5.0, "PA",
                     "b-111_-15", "train", i, i, "")


def test_completed_and_pending_are_skipped_failed_retried_until_limit(tmp_path) -> None:
    """Completed and pending patches are skipped; failed ones are retried until the attempt limit."""
    log = TaskLog(str(tmp_path / "raw" / "log.jsonl"))
    log.append("done", "COMPLETED", "direct")
    log.append("queued", "SUBMITTED", "batch", "T1")
    log.append("flaky", "FAILED", "direct", info="boom")
    assert log.should_skip("done", 3) and log.should_skip("queued", 3)
    assert not log.should_skip("flaky", 3) and not log.should_skip("new", 3)
    log.append("flaky", "FAILED", "direct")
    log.append("flaky", "FAILED", "direct")
    assert log.should_skip("flaky", 3)
    assert log.pending_tasks() == {"T1": "queued"}


def test_torn_last_line_is_ignored(tmp_path) -> None:
    """A torn last log line is ignored when reading the task log."""
    path = tmp_path / "log.jsonl"
    log = TaskLog(str(path))
    log.append("a", "COMPLETED", "direct")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write('{"patch_id": "b", "sta')
    assert list(log.latest()) == ["a"]


def test_export_all_resumes_without_redoing_finished_patches(tmp_path, monkeypatch) -> None:
    """A second export_all skips every patch the first run completed."""
    calls = []

    def fake_export(cfg, s, raw_dir, log, backend) -> str:
        """Record the patch and log it as completed."""
        calls.append(s.patch_id)
        log.append(s.patch_id, "COMPLETED", backend)
        return "COMPLETED"

    monkeypatch.setattr(pipeline, "export_patch", fake_export)
    cfg = {"export": {"max_pending_tasks": 10, "max_attempts": 2}}
    log = TaskLog(str(tmp_path / "log.jsonl"))
    specs = [spec(i) for i in range(3)]
    assert pipeline.export_all(cfg, specs, str(tmp_path), log, "direct") == {"COMPLETED": 3}
    assert pipeline.export_all(cfg, specs, str(tmp_path), log, "direct") == {"skipped": 3}
    assert calls == ["p0", "p1", "p2"]


def test_refresh_batch_logs_terminal_states(tmp_path, monkeypatch) -> None:
    """refresh_batch logs completed and failed batch tasks and keeps running ones pending."""
    fake = MagicMock()
    fake.data.getTaskStatus.return_value = [{"id": "T1", "state": "COMPLETED"}, {"id": "T2", "state": "RUNNING"},
                                            {"id": "T3", "state": "FAILED", "error_message": "quota"}]
    monkeypatch.setattr(export, "ee", fake)
    log = TaskLog(str(tmp_path / "log.jsonl"))
    for i in (1, 2, 3):
        log.append(f"p{i}", "SUBMITTED", "batch", f"T{i}")
    assert export.refresh_batch(log) == {"p1": "COMPLETED", "p3": "FAILED"}
    assert log.pending_tasks() == {"T2": "p2"}


def test_direct_pull_uses_patch_grid_and_numpy_format(monkeypatch) -> None:
    """pull_direct requests the patch grid in NUMPY_NDARRAY format and returns bands by name."""
    fake = MagicMock()
    fake.data.computePixels.return_value = np.zeros((48, 48), dtype=[("t000_VV", "f4"), ("t000_VH", "f4")])
    monkeypatch.setattr(export, "ee", fake)
    grid = make_grid(-55.4, -7.0, 48, 10.0)
    bands = export.pull_direct("IMAGE", grid)
    request = fake.data.computePixels.call_args.args[0]
    assert request["fileFormat"] == "NUMPY_NDARRAY" and request["grid"]["crsCode"] == grid.crs
    assert set(bands) == {"t000_VV", "t000_VH"} and bands["t000_VV"].shape == (48, 48)


def test_raw_npz_round_trip_and_batch_table_rebuild(tmp_path) -> None:
    """Raw bands round-trip through npz and batch tables rebuild onto the patch grid."""
    grid = make_grid(-55.4, -7.0, 48, 10.0)
    bands = {"t000_VV": np.random.default_rng(0).normal(-8, 1, (48, 48))}
    export.save_raw(str(tmp_path), "p0", bands, {"bands": ["t000_VV"]})
    loaded, info = export.load_raw(str(tmp_path), "p0", grid)
    assert np.allclose(loaded["t000_VV"], bands["t000_VV"]) and info["bands"] == ["t000_VV"]
    assert export.load_raw(str(tmp_path), "absent", grid) is None
    xs, ys = grid.pixel_centres()
    lons, lats = _transformer(grid.epsg, 4326).transform(xs.ravel(), ys.ravel())
    table = pd.DataFrame({"t000_VV": bands["t000_VV"].ravel(), "longitude": lons, "latitude": lats})
    rebuilt = table_to_bands(table, grid, ["t000_VV"])
    assert np.allclose(rebuilt["t000_VV"], bands["t000_VV"])
    rows, cols = lonlat_to_rowcol(lons, lats, grid)
    assert rows.min() == 0 and rows.max() == 47 and cols.max() == 47
