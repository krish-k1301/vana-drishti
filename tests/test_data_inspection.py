"""Phase 0 inspection script on a copy of the synthetic fixture."""
from __future__ import annotations

import shutil

import pandas as pd
import torch
import yaml

from scripts import inspect_bradd
from src.data import compute_train_stats


def test_inspection_report(bradd_root, tmp_path):
    """The inspection script reports files, geolocation hits, dates and split counts."""
    top = tmp_path / "archive"
    shutil.copytree(bradd_root, top / "BraDD-S1TS")
    (top / "extras").mkdir()
    (top / "extras" / "patch_centers.geojson").write_text("{}")
    meta = pd.read_csv(top / "BraDD-S1TS" / "meta.csv", index_col=0)
    meta["lon"] = 0.0
    meta.to_csv(top / "BraDD-S1TS" / "meta.csv")
    cfg = yaml.safe_load(open("configs/data/bradd.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    out = tmp_path / "out"
    assert inspect_bradd.main(["--config", str(cfg_path), "--root", str(top), "--out-dir", str(out),
                               "--n-samples", "5", "--label", "SMOKE"]) == 0
    report = (out / "report.md").read_text()
    assert "SMOKE" in report and "**False**" in report
    hits = pd.read_csv(out / "geolocation_hits.csv")
    assert set(hits["name"]) == {"lon", "extras/patch_centers.geojson"}
    samples = pd.read_csv(out / "samples.csv")
    assert len(samples) == 5 and (samples["last_label_minus_alert"] == 0).all()
    splits = pd.read_csv(out / "split_counts.csv").set_index("split")
    assert splits.loc["train", "meta_csv"] == 8 and splits.loc["validation", "paper"] == 5251
    assert len(pd.read_csv(out / "files.csv")) == 16 + 2
    assert (bradd_root / "meta.csv").read_text() != (top / "BraDD-S1TS" / "meta.csv").read_text()


def test_shipped_stats_compared(bradd_root, tmp_path):
    """A shipped close_stats.pt is reported next to our train stats with per-channel differences."""
    top = tmp_path / "archive"
    shutil.copytree(bradd_root, top)
    ours = compute_train_stats(top, "close_set")
    torch.save({"mean": ours["mean"] + torch.tensor([0.5, 0.0]), "std": ours["std"], "min": -30.0, "max": 5.0},
               top / "close_stats.pt")
    out = tmp_path / "out"
    assert inspect_bradd.main(["--config", "configs/data/bradd.yaml", "--root", str(top), "--out-dir", str(out),
                               "--n-samples", "2", "--label", "SMOKE"]) == 0
    table = pd.read_csv(out / "shipped_stats_comparison.csv").set_index("channel")
    assert abs(table.loc["VV", "abs_diff_mean"] - 0.5) < 1e-5 and table.loc["VH", "abs_diff_std"] < 1e-6
    report = (out / "report.md").read_text()
    assert "**True**" in report and "Max abs difference: 0.5" in report
