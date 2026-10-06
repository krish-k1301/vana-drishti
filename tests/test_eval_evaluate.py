"""End-to-end SMOKE tests for src.evaluate, src.benchmark_table and scripts/plot_benchmark.py."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src import evaluate, train
from src.config import load_config
from src.evaluation.common import load_model

REPO = Path(__file__).resolve().parents[1]
TINY_UTAE = [
    "model.params.encoder_widths=[8, 8, 8, 16]", "model.params.decoder_widths=[8, 8, 8, 16]",
    "model.params.out_conv=[8, 2]", "model.params.d_model=32", "model.params.n_head=4",
]


@pytest.fixture(scope="module", name="trained")
def fixture_trained(bradd_root: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    """A tiny U-TAE trained for one epoch with src.train (real Lightning checkpoint and CSV logs)."""
    runs = tmp_path_factory.mktemp("runs")
    cfg = load_config(REPO / "configs" / "smoke" / "utae_ce.yaml", [
        f"data.root={bradd_root}", f"data.stats_path={runs / 'stats.pt'}", f"output.runs_dir={runs}",
        "trainer.max_epochs=1", "data.num_workers=0", *TINY_UTAE,
    ])
    train.run(cfg)
    ckpt = next((runs / cfg["experiment_name"] / "checkpoints").glob("*.ckpt"))
    return cfg, ckpt


def eval_cfg() -> dict:
    """The shipped evaluation config."""
    return load_config(REPO / "configs" / "eval" / "evaluate.yaml")


def test_checkpoint_weights_round_trip(trained: tuple[dict, Path]) -> None:
    """Loaded weights equal the `model.*` tensors stored by Lightning."""
    cfg, ckpt = trained
    model = load_model(cfg, ckpt)
    stored = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    for name, tensor in model.state_dict().items():
        assert torch.equal(tensor, stored[f"model.{name}"]), name


def test_evaluate_matches_train_test_metrics(trained: tuple[dict, Path], tmp_path: Path) -> None:
    """src.evaluate on the best checkpoint reproduces src.train's own test scores, plus strata and MMU."""
    cfg, ckpt = trained
    payload = evaluate.run(cfg, str(ckpt), "test", eval_cfg(), None, tmp_path)
    reference = json.loads((Path(cfg["output"]["runs_dir"]) / cfg["experiment_name"] / "metrics_test.json").read_text())
    for level in ("pixel", "patch"):
        for variant in ("with_or", "without_or"):
            for key in ("tp", "fp", "fn", "tn"):
                assert payload[level][variant][key] == reference[level][variant][key]
    assert payload["label"] == "SMOKE" and payload["params"] > 0 and payload["mmu"].startswith("MMU 0.01 ha")
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    kinds = {(r["stratum_type"], r["stratum"]) for r in rows}
    assert {("size_bin", "<0.5 ha"), ("edge_interior", "edge"), ("edge_interior", "outer_ring")} <= kinds
    assert all(r["label"] == "SMOKE" and r["mmu"] for r in rows)
    size_tp = sum(int(r["tp"]) for r in rows if r["stratum_type"] == "size_bin" and r["variant"] == "with_or")
    assert size_tp == payload["pixel"]["with_or"]["tp"]


def test_evaluate_temporal_subsample_last_only(trained: tuple[dict, Path]) -> None:
    """The single-date ablation runs at test time and lands in its own output folder."""
    cfg, ckpt = trained
    spec = evaluate.parse_subsample(["mode=last_only", "num_dates=1"])
    payload = evaluate.run(cfg, str(ckpt), "test", eval_cfg(), spec, None)
    out = evaluate.default_out(cfg, "test", spec)
    assert out.name == "eval_test_ts-last_only-1" and (out / "metrics.json").is_file()
    assert payload["temporal_subsample"]["at_test"] is True
    with pytest.raises(ValueError):
        evaluate.parse_subsample(["dates=3"])


def test_benchmark_table_and_plot(trained: tuple[dict, Path], tmp_path: Path) -> None:
    """benchmark_table reads the eval JSON and CSV logs; plot_benchmark renders a SMOKE-titled PNG."""
    cfg, ckpt = trained
    evaluate.run(cfg, str(ckpt), "test", eval_cfg(), None, None)
    run = Path(cfg["output"]["runs_dir"]) / cfg["experiment_name"]
    table = tmp_path / "bench.csv"
    subprocess.run([sys.executable, "-m", "src.benchmark_table", "--runs", str(run), "--out", str(table)],
                   cwd=REPO, check=True)
    row = next(csv.DictReader(table.open()))
    metrics = json.loads((run / "eval_test" / "metrics.json").read_text())
    assert float(row["pixel_iou"]) == pytest.approx(metrics["pixel"]["with_or"]["iou"])
    assert row["is_reference"] == "True" and row["smoke"] == "True" and float(row["epoch_time_s"]) > 0
    assert row["peak_gpu_mem_mb"] == "" and row["mmu"].startswith("MMU")
    png = tmp_path / "bench.png"
    subprocess.run([sys.executable, "scripts/plot_benchmark.py", "--table", str(table), "--out", str(png)],
                   cwd=REPO, check=True)
    assert png.stat().st_size > 0
