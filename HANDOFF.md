# HANDOFF: continue the VanaDrishti autonomous build locally

Give this file to Claude Code as the first message in a fresh session at the repo root, together with
`PRD.md`. It describes exactly where the cloud build stopped and what is left.

> **Prompt to paste into Claude Code:**
> Read `HANDOFF.md` and `PRD.md` fully. You are resuming an autonomous build of this repo. Follow the
> "Operating rules" and work through "Remaining work" in order. Don't redo anything listed as done.
> Keep `HANDOFF.md` up to date as you go, and delete it at final cleanup once its content is in `REPORT.md`.

---

## 1. Get the code and environment

```bash
git clone https://github.com/krish-k1301/vana-drishti.git
cd vana-drishti
git checkout claude/new-session-uto7x0        # all work is on this branch, nothing merged to main
conda env create -f environment.yml           # Python 3.11, torch 2.5.1, lightning 2.4.0, ...
conda activate vanadrishti
OMP_NUM_THREADS=1 python -m pytest -q tests   # expect: 251 passed, 1 xfailed (~1 min)
```

Notes on the environment:
- The last 4 pip lines in `environment.yml` (timm, opencv, torch-scatter, detectron2) are only needed for the
  Exchanger+U-Net model. torch-scatter and detectron2 build from source (needs a C++ toolchain; detectron2 needs
  `pip install --no-build-isolation`). If they fail, comment them out; only `exchanger_unet` breaks.
- In the cloud build, `python` was `/home/user/venv311/bin/python`. Locally it is just `python` inside the env.
- If you have a GPU, install the CUDA build of torch 2.5.1 matching your driver.
- Commits must be authored as Krish Kubadia <kubu1813@gmail.com> with **no Claude co-author trailer**.

## 2. Operating rules (from the original orchestrator prompt; the owner granted all permissions)

- Work in phases; after each phase passes its checks, commit and push to `claude/new-session-uto7x0`.
- **Never fabricate or estimate a result.** Every number in a report comes from a file written by code that ran.
  Anything run on synthetic data, a subset, fewer epochs, or CPU is labelled `SMOKE` everywhere. A SMOKE run
  never counts as reproducing the paper.
- Never edit `third_party/` (vendored upstream code, verified byte-identical to upstream commits). Wrap in `src/`.
- Code rules: no file over 250 lines, no function over 50 lines, type hints + one-line docstring on every public
  function, no commented-out code, no dead code, every path/hyperparameter from YAML, no notebooks, plots from
  `scripts/`. `tests/test_review_code_rules.py` checks these.
- Before every commit: `OMP_NUM_THREADS=1 python -m pytest -q tests` and check the **exit code**, not just the
  tail of the output (a piped `tail` hid a failure once).
- The original prompt used subagents (data / model / eval / gee / reviewer). Locally you may do the work directly
  or use subagents; keep the independent-review step for anything that touches labels or metrics.

## 3. Where things stand (2026-10-06 ~21:00 UTC; see `git log` for the latest commit)

### Blockers in the cloud environment (probably not blockers on your PC)
| Blocker | Effect |
|---|---|
| zenodo.org blocked by proxy | BraDD-S1TS (17.7 GB) never downloaded. Phase 0 inspection not run on real data. |
| terrabrasilis.dpi.inpe.br blocked | DETER / PRODES never downloaded; class names and WFS layer names unverified. |
| No GPU | Every training run was a 2-epoch CPU SMOKE run on synthetic data. |
| No Earth Engine credentials | No exports (Phases 3, 5, 6 real data). |
| huggingface.co, arxiv.org, CVF blocked | AnySat weights not downloaded; AnySat BraDD numbers and Flores-Anderson 2026 statistic not verified. |

### Phase status
| Phase | State | What exists |
|---|---|---|
| 0 Setup + inspection | PARTIAL (data blocked) | Layout, `environment.yml`, vendoring, `scripts/download_bradd.py` (resumable + md5), `scripts/inspect_bradd.py` (every PRD item incl. geolocation search and shipped `close_stats.pt` comparison), tested on synthetic data only. |
| 1 Baseline reproduction | PARTIAL (no GPU, no data) | Our pipeline `src/train.py`; upstream reference launcher `src/reference_run.py`; 4 SMOKE runs done. |
| 2 Benchmark suite | PARTIAL (SMOKE done, no GPU/data) | Tier A (ConvLSTM, ConvGRU, 3D-UNet, U-TAE seq2seq) and Tier B/C (TSViT, Exchanger+U-Net, Galileo nano WORKING; AnySat code works but weights BLOCKED). SMOKE benchmark of 8 models + U-TAE temporal-depth ablation done: `results/smoke/benchmark_smoke.csv`, `.png`, `ablation_utae_*.csv`. SMOKE configs `configs/smoke/bench_*.yaml` (convlstm/convgru/galileo train on 250 samples, exchanger on 100 with batch 1 — 14 GB RAM OOM at batch 4; all noted in the configs). |
| 3 Dated Amazon dataset | PARTIAL (no EE creds) | Full GEE pipeline in `gee/`, tested with mocked `ee`; CLI exits 2 "BLOCKED: no Earth Engine credentials". DETER/PRODES download script written, never reached the server. |
| 4 Prefix-truncation training | PARTIAL (SMOKE done, no dated data) | `src/data/prefix.py` (fixed after review), `configs/early/utae_prefix_{deter,burn}.yaml`. SMOKE runs on the fixed code done: `results/smoke/smoke_early_{deter,burn}_metrics_test.csv`. On the dated set the `label[0] OR pred` rule inflates IoU (cumulative labels keep 1→1 pixels): 0.92 with OR vs ~0.00 without after 2 epochs. **Report the without-OR numbers for Phases 4–6.** Checkpoints are in gitignored `results/runs/smoke_early_*/checkpoints/` (regenerate if missing: `python -m src.train --config configs/smoke/early_deter.yaml`, ~9 min on 4 CPU cores). |
| 5 Early-detection eval | PARTIAL (SMOKE done, no dated data) | SMOKE early_eval on both Phase 4 checkpoints: `results/smoke/early_{deter,burn}_*` (summary, latency, recall, false alarms, plots). The 2-epoch SMOKE model detects 0/33 events (pipeline check only). Note: with noisy-OR over ~14 monthly intervals the validation-chosen tau was ~0.9999 — noisy-OR saturates on long windows; compare `interval_combiner: max` on real data. | `src/early_eval.py` (sliding prefix, noisy-OR over 30-day intervals, tau on validation at a false-alarm budget, latency vs DETER/burn/RADD, recall at +0/12/24/48/90 d, splits by stage/size/edge), `scripts/plot_early.py`. |
| 6 Cross-biome | CODE DONE, no runs | `src/cross_biome.py`, `src/label_decomposition.py`, GEE configs for Congo/Borneo pilots, per-region norm / crop / label-source switches in `src/data/`. |

### SMOKE results so far (synthetic data, 500/100/100 samples, 2 CPU epochs; they reproduce nothing)
Small CSV copies of every SMOKE metrics file are committed in `results/smoke/` (benchmark table, ablation, Phase 1
and Phase 4 test metrics). The table below is from the Phase 1 runs.
Pixel IoU with the `label[0] OR prediction` rule, from `results/runs/<run>/metrics_test.json`
(note: `results/runs/` is gitignored, so these files are **not** in the repo; regenerate with the commands below):

| Run | Pixel IoU | Source file |
|---|---|---|
| upstream reference, CE | 0.7228 | results/runs/smoke_reference_utae_ce/metrics_test.json |
| ours, CE | 0.7438 | results/runs/smoke_utae_ce/metrics_test.json |
| upstream reference, focal γ=1 | 0.7774 | results/runs/smoke_reference_utae_focal/metrics_test.json |
| ours, focal γ=1 (retrained on current code) | 0.7697 | results/smoke/smoke_utae_focal_metrics_test.csv |

Regenerate: `python -m src.smoke_data --config configs/smoke/synthetic_bradd.yaml`, then
`python -m src.reference_run --config configs/smoke/reference_utae_ce.yaml` and
`python -m src.train --config configs/smoke/utae_ce.yaml` (same for `*_focal`).

## 4. Remaining work, in order

1. ~~Phase 4 SMOKE~~ done (and the mixed-label-count test exists: `tests/test_train_intervals.py`).
   If `data/synthetic_dated` or the checkpoints are missing locally, regenerate:
   `python -m src.smoke_data --config configs/smoke/synthetic_dated.yaml`, then the two `src.train` commands.
2. ~~Phase 5 SMOKE~~ done (`results/smoke/early_*`). Command used:
   `python -m src.early_eval --config results/runs/smoke_early_deter/config_resolved.yaml --checkpoint <best ckpt>`
   then `python scripts/plot_early.py --early-dir results/runs/smoke_early_deter/early_eval` (~7.5 min each on CPU).
3. ~~Phase 2 SMOKE~~ done (`results/smoke/`). `src.evaluate` now applies `trainer.float32_matmul_precision` so its
   counts match the test pass inside `src.train` exactly.
4. **Phase 6 SMOKE**: `src.cross_biome` on a copy of the synthetic dated set with DETER/burn keys stripped, and
   `src.label_decomposition` on two SMOKE runs (`data.label_source` prodes vs hansen at train, prodes at test).
5. **Final cleanup** (from the orchestrator prompt):
   - Keep: `PRD.md`, `README.md`, `REPORT.md`, `environment.yml`, `configs/`, `src/`, `gee/`, `scripts/`
     (download + plotting only), `third_party/`, `tests/`, `results/` (final CSVs and figures only), `.gitignore`.
   - Delete: scratch files, debug scripts, smoke checkpoints, temporary datasets, `__pycache__`, unused configs,
     and this `HANDOFF.md` (after its content is in `REPORT.md`).
   - Add return type hints + one-line docstrings to all test functions, then remove the `xfail` marker in
     `tests/test_review_code_rules.py::test_test_code_follows_rules`.
   - Remove the now-redundant per-module single-thread fixtures in `tests/test_eval_*.py` (conftest has one).
   - The synthetic `label_hansen` in `src/data/synthetic.py` is dated by RADD; make it Hansen-year-end dated to
     match the real pipeline.
   - Copy the SMOKE CSVs/figures you want to keep into `results/` (small files only) and commit them.
   - Write `README.md` (setup, data download, the exact command to train and evaluate each model) and
     `REPORT.md`: (1) one-line status per phase with reason; (2) results tables, every number tied to its source
     file, SMOKE labelled; (3) the decisions log (section 6 below); (4) numbered owner to-do list (section 5).
   - Run the full suite one last time; commit; push.
6. **Real runs (owner, needs GPU / data / credentials)**: section 5.

## 5. Owner to-do list (goes into REPORT.md)

1. Download BraDD: `python scripts/download_bradd.py --config configs/data/downloads.yaml` (needs ~36 GB free, or
   ~18 GB with zip deletion). Then `python scripts/inspect_bradd.py --config configs/data/bradd.yaml` and read the
   report: does anything geolocate the patches? does `close_stats.pt` ship? split counts vs paper?
2. Phase 1 full runs on a GPU (each ~hours; 200 epochs max, early stopping 30):
   ```
   python -m src.reference_run --config configs/reference_baseline_utae.yaml       data.root=<BraDD dir>
   python -m src.train         --config configs/baseline_utae.yaml                 data.root=<BraDD dir>
   python -m src.reference_run --config configs/reference_baseline_utae_focal.yaml data.root=<BraDD dir>
   python -m src.train         --config configs/baseline_utae_focal.yaml           data.root=<BraDD dir>
   ```
   Acceptance: CE 47.3 ± 2, focal 48.6 ± 2 pixel IoU, ours within 1 point of reference. If the archive ships
   `close_stats.pt`, point both tracks' `data.stats_path` at the same file.
3. DETER / PRODES: `python scripts/download_deter_prodes.py --config configs/data/downloads.yaml`; check the class
   names it prints against the PRD list.
4. Earth Engine: set `EE_SERVICE_ACCOUNT`, `EE_KEY_FILE`, `EE_PROJECT` (never commit the key). Fill every `VERIFY`
   item in `configs/gee/*.yaml`, especially the MapBiomas Fogo Collection 5 asset ID and band names, the GAUL
   states layer, RADD property names, and the GCS bucket. Then
   `python gee/run_export.py --config configs/gee/amazon_dated.yaml --pilot`, review the QA report and NICFI
   spot-check list, then the full export (refused until the pilot passes).
5. Decide: (a) DETER burn scars (`CICATRIZ_DE_QUEIMADA`) count as positives in the Amazon dated set; keep?
   (b) early-eval false-alarm budget (placeholder 0.5 / km² / month) and `component_hit_fraction` (0.5).
   (c) Phase 6 Hansen dating at 31 Dec of the loss year (causal-safe, 1-year resolution); acceptable?
6. Verify from the papers: AnySat's BraDD-S1TS IoU and protocol (a search snippet suggests ~75–90, so likely a
   different metric/protocol; don't compare until checked), and Flores-Anderson et al. 2026 (56%, p = 0.63).
7. Supply the AnySat checkpoint (`checkpoint_path` in `configs/models/anysat.yaml`) if you want Tier C AnySat.

## 6. Decisions log (copied from the cloud orchestrator's gitignored `.orchestrator/DECISIONS.md`)

- Autonomous mode, all permissions granted by the owner; commits authored as Krish Kubadia, no Claude trailer.
- Python 3.11; torch 2.5.1 / lightning 2.4.0 / pandas 2.2.3 / numpy 2.1.3 (pins in `environment.yml`).
- Vendored ecovision-uzh/BraDD-S1TS @ f91d1806 (pictures dropped) and VSainteuf/utae-paps @ 987874e2 (reference
  copy). Tier B/C: DeepSatModels @ 53e65593 (TSViT, Apache-2.0), Exchanger4SITS @ 6a5ecf39 (MIT), AnySat @ 5f6f475e
  (MIT), galileo @ 0f0b5b95 (MIT, nano weights 4.2 MB vendored). Each has `VENDORED.md`.
- Upstream `source/utae/sltae.py` does `from turtle import forward` (unused); crashes without tkinter. Stub in
  `src/models/vendor.py`.
- Upstream `Network._segment_wise` mutates `target_days` in place; for t > 2 later intervals lose the leading
  30-day margin. Ours never mutates.
- Upstream `run_via_parser.py` passes `Normalization_method` but the dataset reads `NormalizationMethod`, so the
  CLI flag is ignored (always ZScore). Our reference launcher passes the key upstream reads.
- Upstream stats bootstrap crashes (`KeyError 'train_set'`, PRD 4.4 issue 1, reproduced). We compute train-split
  stats ourselves; the reference launcher writes `close_stats.pt` only if missing.
- Upstream notebook shows 0.0% 1→1 pixels and 0.87% 1→0, so `label[0] OR pred` can only add false positives
  (lower IoU). Metrics report both variants plus the forced-FP count.
- Upstream ConvLSTM/ConvGRU run the recurrence through padded dates (batch-dependent predictions); our wrappers
  freeze state on padding. UNet3D crashes for T < 4 and drops T mod 4 dates; our wrapper pads T.
- `TemporalSubsample` modes: `uniform` = upstream `is_random=True` (deterministic, despite the name), `random` =
  upstream `is_random=False`, `last_only` = the paper's single-date case upstream can't produce.
- Phase 4: trailing margin 0 and an inclusive end bound (`model.inclusive_end: true`) so the image at t_c is used
  and nothing after it; baseline keeps upstream's strict bounds.
- DETER download widened to 2019-01-01 .. 2024-04-30 to screen negatives over the full window.
- Deter-anchored prefix labels use per-pixel `ref_day` (all DETER polygons in the patch), not just the central
  polygon (review fix).
- Burn anchor only labels deforestation pixels; fires on pasture/negatives never become positives (review fix).
- Burn months (MapBiomas Fogo) dated to the month's last day: causal-safe; latency vs burn has ±1-month resolution.
- Phase 6 positives sampled and dated from Hansen alone (31 Dec of loss year); RADD kept as an attribute only, so
  latency vs RADD isn't circular (PRD 5: RADD never ground truth).
- One state layer (GAUL level 1, VERIFY) gives `state` to positives and negatives, so per-region norm can't leak
  the label.
- Patch centre spacing ≥ 700 m (patch diagonal 679 m + grid snapping) so patches never overlap.
- Phase 5 scores the same ~30-day interval grid prefix training uses at each cutoff, combined per pixel by noisy-OR
  (1 − Π(1 − p_k)); `max` available. A single long interval would be out of distribution.
- Crop tiling: 16 px → 9 disjoint tiles; 32 px → 4 corner crops overlapping by 16 px (metrics summed over 32-px
  tiles double-count overlap strips).
- Tier B/C recipe differences are flagged in each `configs/models/*.yaml` header (they all train here with the
  shared focal α=0.75 γ=1 recipe per PRD 6).

## 7. Key interfaces (from `.orchestrator/INTERFACES.md`)

- Batch dict: `Images` [B,T,2,H,W] float (0-padded at the end), `ImageDays` [B,T] (days since earliest
  image/label date, from 1; 0 = pad), `TargetDays` [B,t], `Targets` [B,t,H,W], `PadMask` [B,T] bool, `Index` [B].
  Dated set adds `EventDay`, `BurnDay`, `RaddDay`, `RefDay`, `EventMask`.
- `build_model(cfg)` → module `forward(images, days, target_days, pad_mask) -> [B,t-1,2,H,W]` (segment mode).
- Scoring: change target = `Targets[:, :-1] != Targets[:, 1:]`; scored prediction = `Targets[:, :-1] | argmax`;
  scored target = `Targets[:, 1:]`; confusion matrix over the whole set (matches upstream exactly, tested).
- Dated sample keys: `ref_day` int32 [48,48] (days since 1970-01-01, -1 none; invariant
  `label[k] == prior | (ref_day valid & ref_day <= label_dates[k])`), `label_hansen` int64 [t,48,48],
  `burn_month` = last day of first burn month, `radd_date` attribute only.
