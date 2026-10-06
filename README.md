# VanaDrishti

Sentinel-1 deforestation detection with early-stage evaluation. The baseline is U-TAE on BraDD-S1TS
(Karaman et al. 2023). The project measures how early, and at which stage, a SAR model detects clearing, and how it
transfers to the Congo Basin and Borneo. The spec is `PRD.md`; the status and results are in `REPORT.md`.

## Setup

```bash
conda env create -f environment.yml
conda activate vanadrishti
OMP_NUM_THREADS=1 python -m pytest -q tests     # ~1 min on CPU, no real data needed
```

- Python 3.11, torch 2.5.1, lightning 2.4.0. On a GPU machine install the CUDA build of torch 2.5.1.
- The last four pip entries in `environment.yml` (timm, opencv, torch-scatter, detectron2) are needed only for
  the Exchanger+U-Net model. torch-scatter and detectron2 build from source (C++ toolchain; detectron2 needs
  `pip install --no-build-isolation`).
- Every path and hyperparameter comes from YAML. Any command accepts dotted overrides after the config, e.g.
  `data.root=/path trainer.max_epochs=2`.

## Layout

| Path | What |
|---|---|
| `configs/` | `baseline_utae*.yaml` and `reference_baseline_utae*.yaml` (Phase 1), `models/` (Phase 2 challengers), `early/` (Phase 4), `eval/`, `data/`, `gee/`, `smoke/` (CPU runs on synthetic data) |
| `src/data/` | BraDD and dated datasets, explicit pad mask, train-split stats, temporal subsample, prefix truncation, per-region norm, seasonal window, crops, synthetic data generator |
| `src/models/` | `build_model` (U-TAE, U-TAE seq2seq, ConvLSTM, ConvGRU, 3D-UNet, TSViT, Exchanger+U-Net, Galileo, AnySat), segment-wise network |
| `src/losses/`, `src/metrics/`, `src/training/`, `src/evaluation/` | Focal loss, metrics (upstream-exact scoring, OR-rule effect, latency, size bins, edge/interior, false alarms), Lightning module, evaluation helpers |
| `src/train.py`, `src/reference_run.py` | Our training pipeline; thin launcher around the vendored upstream code |
| `src/evaluate.py`, `src/benchmark_table.py`, `src/early_eval.py`, `src/cross_biome.py`, `src/label_decomposition.py` | Evaluation CLIs for Phases 1–6 |
| `gee/` | Earth Engine export pipeline (Phase 3 dated Amazon set, Phase 6 Congo/Borneo pilots) |
| `scripts/` | Downloads (`download_bradd.py`, `download_deter_prodes.py`), BraDD inspection, plots |
| `third_party/` | Vendored upstream code, never edited (each directory has its commit and license) |
| `results/smoke/` | SMOKE results (synthetic data, CPU); they reproduce nothing |

## Data

```bash
# BraDD-S1TS (17.7 GB zip, md5 checked, resumable; deletes the zip after unzip if free space < 45 GB)
python scripts/download_bradd.py --config configs/data/downloads.yaml
python scripts/inspect_bradd.py --config configs/data/bradd.yaml --root data/BraDD-S1TS --out-dir results/phase0

# DETER-B and PRODES from TerraBrasilis (prints DETER class names to check against PRD Section 5)
python scripts/download_deter_prodes.py --config configs/data/downloads.yaml

# Earth Engine exports: set EE_SERVICE_ACCOUNT, EE_KEY_FILE, EE_PROJECT and fill every VERIFY item in configs/gee/
python gee/run_export.py --config configs/gee/amazon_dated.yaml --pilot   # then review the QA report
python gee/run_export.py --config configs/gee/amazon_dated.yaml           # refused until the pilot passed QA
python gee/run_export.py --config configs/gee/congo_pilot.yaml --pilot    # Phase 6; also borneo_pilot.yaml
```

## Train and evaluate

`<BraDD>` is the unzipped dataset directory (default `data/BraDD-S1TS`). Each run writes to
`results/runs/<experiment_name>/` (checkpoints, CSV logs, `metrics_test.json`, `config_resolved.yaml`).

**Phase 1, baseline reproduction** (targets: CE 47.3 ± 2, focal 48.6 ± 2 pixel IoU; ours within 1 point of the
reference):
```bash
python -m src.reference_run --config configs/reference_baseline_utae.yaml       data.root=<BraDD>
python -m src.train         --config configs/baseline_utae.yaml                 data.root=<BraDD>
python -m src.reference_run --config configs/reference_baseline_utae_focal.yaml data.root=<BraDD>
python -m src.train         --config configs/baseline_utae_focal.yaml           data.root=<BraDD>
```

**Phase 2, benchmark** (one per model: `utae_seq2seq`, `convlstm`, `convgru`, `unet3d`, `tsvit`, `exchanger_unet`,
`galileo`, `anysat`; AnySat needs `model.params.checkpoint_path`):
```bash
python -m src.train --config configs/models/<model>.yaml data.root=<BraDD>
python -m src.evaluate --config results/runs/<run>/config_resolved.yaml --checkpoint <best.ckpt> --split test
python -m src.evaluate --config ... --checkpoint ... --split test --temporal-subsample mode=last_only num_dates=1
python -m src.evaluate --config ... --checkpoint ... --split test --temporal-subsample mode=uniform num_dates=5
python -m src.benchmark_table --runs results/runs/<run> [...] --out results/benchmark.csv
python scripts/plot_benchmark.py --table results/benchmark.csv --out results/benchmark.png
```

**Phase 4, prefix-truncation training** (needs the Phase 3 dated set at `data/dated_amazon`):
```bash
python -m src.train --config configs/early/utae_prefix_deter.yaml
python -m src.train --config configs/early/utae_prefix_burn.yaml
```

**Phase 5, early-detection evaluation** (sliding prefix, tau chosen on validation at a false-alarm budget):
```bash
python -m src.early_eval --config results/runs/<early run>/config_resolved.yaml --checkpoint <best.ckpt>
python scripts/plot_early.py --early-dir results/runs/<early run>/early_eval
```

**Phase 6, cross-biome transfer and label decomposition:**
```bash
python -m src.cross_biome --config results/runs/<amazon run>/config_resolved.yaml --checkpoint <best.ckpt> \
    --tau-from results/runs/<amazon run>/early_eval/summary.json --eval-overrides target_data.root=data/dated_congo
python -m src.label_decomposition --prodes-run results/runs/<prodes run> --hansen-run results/runs/<hansen run> \
    --out results/label_decomposition.csv
```

**SMOKE runs on CPU without real data:** `python -m src.smoke_data --config configs/smoke/synthetic_bradd.yaml`
(or `synthetic_dated.yaml`, then `synthetic_target_biome.yaml`), then any `configs/smoke/*.yaml` with `src.train`.
SMOKE outputs carry `"label": "SMOKE"` and never count as results.
