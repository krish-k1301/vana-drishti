# VanaDrishti build report

Autonomous build of `PRD.md`, Phases 0–6, run in a cloud container on 2026-10-06. Everything in the PRD that can be
built without real data, a GPU or Earth Engine credentials is built and tested. **No result in this report comes
from real data.** Every number below is from a SMOKE run: synthetic BraDD-format data
(`src/data/synthetic.py`), CPU only, 2 epochs. SMOKE runs prove the code runs end to end. They reproduce nothing
and say nothing about the models.

Why nothing ran on real data: the container's network policy blocked zenodo.org (BraDD-S1TS),
terrabrasilis.dpi.inpe.br (DETER, PRODES), huggingface.co (AnySat weights) and arxiv.org/CVF (paper checks). There
was no GPU and there were no Earth Engine credentials.

Test suite at the final commit: `OMP_NUM_THREADS=1 python -m pytest -q tests` (see the commit message for counts).

## 1. Phase status

| Phase | Status | Reason |
|---|---|---|
| 0 Setup and data inspection | PARTIAL | Layout, environment, vendoring, resumable md5-checked download and full inspection script done. BraDD download BLOCKED (zenodo.org denied by proxy), so the inspection report, the geolocation question and the `close_stats.pt` question are unanswered. |
| 1 Baseline reproduction | PARTIAL (no GPU, no data) | Our pipeline and the upstream reference launcher are built and match upstream exactly where tested. Only SMOKE runs; the 47.3 / 48.6 targets are untested. |
| 2 Benchmark suite | PARTIAL (no GPU, no data) | Tier A and B models and Galileo (Tier C) work; AnySat BLOCKED (weights on huggingface.co). SMOKE benchmark and temporal-depth ablation ran. |
| 3 Dated Amazon dataset | PARTIAL (no Earth Engine credentials) | Full export pipeline built and tested with a mocked `ee`; the CLI stops with "BLOCKED: no Earth Engine credentials". DETER/PRODES download BLOCKED (terrabrasilis denied). Several asset IDs are `VERIFY` placeholders. |
| 4 Early-detection training | PARTIAL (no dated data) | Prefix-truncation training for both anchors (DETER, burn) built and run as SMOKE. |
| 5 Early-detection evaluation | PARTIAL (no dated data) | Sliding-prefix inference, latency, recall at offsets, strata and false-alarm threshold selection built and run as SMOKE. |
| 6 Cross-biome transfer | PARTIAL (no Earth Engine data) | Transfer evaluation and label decomposition built and run as SMOKE on a synthetic Hansen-labelled copy. |

## 2. Results (all SMOKE)

MMU for every table: 0.01 ha (1 pixel of 10 m, 8-connected components). "OR" means the upstream scoring rule
`label[0] OR prediction` (PRD 4.2); "no OR" scores the raw prediction.

### 2.1 Phase 1: our pipeline vs the upstream reference (500/100/100 synthetic samples, 2 epochs)

| Run | Pixel IoU (OR) | Pixel IoU (no OR) | Source |
|---|---|---|---|
| Upstream reference, cross-entropy | 0.7228 | not reported by upstream | `results/smoke/smoke_reference_utae_ce_metrics_test.csv` |
| Ours, cross-entropy | 0.7438 | 0.7442 | `results/smoke/smoke_utae_ce_metrics_test.csv` |
| Upstream reference, focal α 0.75 γ 1 | 0.7774 | not reported by upstream | `results/smoke/smoke_reference_utae_focal_metrics_test.csv` |
| Ours, focal α 0.75 γ 1 | 0.7697 | 0.7698 | `results/smoke/smoke_utae_focal_metrics_test.csv` |

The ours-vs-reference gaps (2.1 and 0.8 points) mean nothing on 2-epoch synthetic runs: the two tracks shuffle with
different RNG streams. Exact equivalence was tested separately: our `SegmentNetwork` is bit-identical to upstream
`Network('UTAE','segment')`, and our scoring equals upstream `ForwardFunction` + `SegmentationScores`.

### 2.2 Phase 2: benchmark (synthetic, 100 test samples, focal α 0.75 γ 1)

Source for every row: `results/smoke/benchmark_smoke.csv` (figure: `results/smoke/benchmark_smoke.png`).

| Model | Pixel IoU (OR) | Pixel IoU (no OR) | Patch IoU (OR) | Params | SMOKE deviation |
|---|---|---|---|---|---|
| U-TAE (reference) | 0.7697 | 0.7698 | 1.0000 | 1,069,158 | none |
| U-TAE seq2seq | 0.7490 | 0.7491 | 1.0000 | 1,417,126 | none |
| TSViT | 0.7601 | 0.7618 | 0.9388 | 1,706,500 | none |
| ConvLSTM | 0.7462 | 0.7481 | 0.9783 | 936,642 | trained on 250 samples |
| ConvGRU | 0.6120 | 0.7538 | 0.9783 | 888,302 | trained on 250 samples |
| 3D-UNet | 0.6978 | 0.7164 | 0.9783 | 1,539,090 | none |
| Galileo (nano, pretrained) | 0.3022 | 0.3131 | 0.6304 | 1,041,984 | trained on 250 samples |
| Exchanger+U-Net | 0.0501 | 0.0501 | 0.4691 | 8,075,410 | trained on 100 samples, batch 1 (batch 4 needed >14 GB RAM) |
| AnySat | not run | | | | weights BLOCKED |

Temporal-depth ablation, U-TAE at test time (`results/smoke/ablation_utae_*.csv`):

| Dates kept | Pixel IoU (OR) | Pixel IoU (no OR) |
|---|---|---|
| last date only | 0.7144 | 0.8550 |
| 2 (uniform) | 0.7613 | 0.7615 |
| 5 (uniform) | 0.7658 | 0.7664 |
| 10 (uniform) | 0.7681 | 0.7681 |
| all | 0.7697 | 0.7698 |

### 2.3 Phase 4: prefix-truncation training (300/80/80 synthetic dated samples, 2 epochs)

| Anchor | Pixel IoU (OR) | Pixel IoU (no OR) | Source |
|---|---|---|---|
| DETER date | 0.9156 | 0.0013 | `results/smoke/smoke_early_deter_metrics_test.csv` |
| Burn month | 0.9157 | 0.0000 | `results/smoke/smoke_early_burn_metrics_test.csv` |

On the dated set the labels are cumulative, so pixels already cleared at `label[0]` stay 1, and the OR rule turns
them into free true positives. The OR numbers are inflated; **use the no-OR numbers for Phases 4–6.** After 2 epochs
the models predict almost no change.

### 2.4 Phase 5: early-detection evaluation (synthetic dated test set, 33 events)

tau chosen on validation at 0.5 false alarms / km² / month: 0.99996 (DETER-anchored model), 0.99988 (burn-anchored).
Sources: `results/smoke/early_{deter,burn}_summary.json`, `..._latency.csv`, `..._recall.csv`, `..._false_alarms.csv`,
plots `..._early_probability_vs_days.png`, `..._early_latency_hist.png`.

| Model | Reference | Events with reference | Detected | Missed | Median latency (days) | IQR |
|---|---|---|---|---|---|---|
| DETER-anchored | DETER | 33 | 0 | 33 | n/a | n/a |
| Burn-anchored | DETER | 33 | 10 | 23 | 117.0 | 5.75 |
| Burn-anchored | burn month | 18 | 6 | 12 | 67.5 | 29.0 |
| Burn-anchored | RADD | 33 | 10 | 23 | 103.5 | 12.25 |

Test false-alarm rate: 0.164 (DETER-anchored) and 0.180 (burn-anchored) per km² per month.

### 2.5 Phase 6: cross-biome transfer and label decomposition

Transfer of the DETER-anchored Phase 4 model to a synthetic Hansen-labelled copy of the same scenes (not a
different biome), tau reused from the Amazon run (`results/smoke/cross_biome_summary.json`,
`cross_biome_early_latency.csv`):

| Metric | Value |
|---|---|
| Pixel IoU (OR / no OR) | 0.8980 / 0.0013 |
| Patch IoU (OR / no OR) | 0.9426 / 0.1217 |
| Events detected vs RADD | 0 of 33 |
| Pixel latency vs RADD | 454 of 4582 pixels detected, median 105 days, IQR 13 |

Label decomposition (identical models trained on PRODES vs Hansen labels, both tested on PRODES;
`results/smoke/label_decomposition_smoke.csv`):

| Level | IoU, PRODES-trained | IoU, Hansen-trained | Hansen penalty |
|---|---|---|---|
| Pixel, no OR | 0.0180 | 0.2410 | −0.2230 |
| Patch, no OR | 0.0536 | 0.4204 | −0.3668 |

The negative penalty is a SMOKE artifact: the synthetic Hansen masks are the DETER masks grown by one pixel, so an
under-predicting model gains recall from them. It says nothing about real labels.

## 3. Findings from the code (verified, not SMOKE)

These came from reading and testing the upstream code. They hold regardless of data.

1. **The OR scoring rule.** On BraDD, the upstream notebook output shows 0.0% of pixels stay 1→1 and 0.87% go 1→0.
   So `label[0] OR prediction` can only add false positives and can only lower IoU. On cumulative dated labels it
   inflates IoU instead (section 2.3). Every metric file reports both variants and the forced-FP count.
2. **PRD 4.4 issues 1–6 are all real.** For example, the stats bootstrap raises `KeyError: 'train_set'`, and
   `TemporalDropout(is_random=True)` is the deterministic one. Each has a work-around in `src/` (decisions log).
3. **Upstream CLI ignores `--DataModule_Dataset_Normalization_method`.** The dataset reads `NormalizationMethod`.
4. **Upstream `Network._segment_wise` mutates `target_days` in place.** With more than 2 label dates, every interval
   after the first loses its 30-day leading margin. This doesn't affect BraDD (2 label dates).
5. **Upstream `sltae.py` imports `turtle`.** Importing U-TAE crashes on Python builds without tkinter.
6. **Upstream ConvLSTM/ConvGRU run through padded dates.** A sample's prediction depends on the other samples in its
   batch. **UNet3D** crashes for fewer than 4 dates and drops the last T mod 4 dates.
7. **Upstream uses `torch.load` without `weights_only`.** On torch ≥ 2.6 that breaks loading BraDD samples, which
   hold `datetime.date` lists.
8. **The bottleneck is 6×6 for 48×48 input** (confirmed by forward hooks). U-TAE has 1,069,158 parameters.

## 4. Decisions log

- The owner granted all permissions for this run. Commits are authored as Krish Kubadia with no Claude co-author line.
- Python 3.11; torch 2.5.1, lightning 2.4.0, pandas 2.2.3, numpy 2.1.3 (`environment.yml`).
- Vendored, unmodified, with commit and license in each `VENDORED.md` or in the git log:
  - BraDD-S1TS @ f91d1806, utae-paps @ 987874e2 (reference copy)
  - DeepSatModels @ 53e65593 (TSViT)
  - Exchanger4SITS @ 6a5ecf39
  - AnySat @ 5f6f475e
  - galileo @ 0f0b5b95 (nano weights, 4.2 MB)
- Work-arounds in `src/` for the upstream quirks in section 3:
  - a `turtle` stub;
  - a non-mutating segment network;
  - train-split stats computed by our own code;
  - explicit `TemporalSubsample` modes, including `last_only` for the paper's single-date case;
  - an explicit `PadMask`;
  - recurrent wrappers that freeze state on padded dates;
  - a UNet3D wrapper that pads the time axis.
- Phase 4 uses a trailing margin of 0 and an inclusive end bound, so the image taken at the cutoff t_c is used and
  nothing after it is. The baseline keeps upstream's strict bounds.
- Phase 4 labels:
  - **DETER anchor:** uses a per-pixel `ref_day` holding every dated DETER polygon in the patch, not just the
    central one (review fix).
  - **Burn anchor:** labels only deforestation pixels; fires on pasture or negatives never become positives.
  - **Burn months:** dated to the month's last day, which is causal-safe. Latency against burn date therefore has
    ±1-month resolution.
- Phase 6 positives are sampled and dated from Hansen alone (31 Dec of the loss year). RADD is kept as an attribute
  only, because PRD 5 says RADD is never ground truth and selecting by RADD would make latency-vs-RADD circular.
- Export geometry:
  - **Shared state layer:** one layer (FAO GAUL level 1, `VERIFY`) assigns states to positives and negatives, so
    per-region normalisation can't leak the label.
  - **No overlap:** patch centres are at least 700 m apart (the patch diagonal is 679 m), so patches never overlap.
  - **Split by region block:** 0.5° blocks; a patch that crosses a block edge is dropped.
- DETER download window: 2019-01-01 to 2024-04-30, wide enough to screen negatives over the whole ±window.
- Amazon dated positives include DETER burn scars (`CICATRIZ_DE_QUEIMADA`). This needs the owner's confirmation.
- Phase 5 scores the same 30-day interval grid that prefix training uses, combined per pixel by noisy-OR,
  1 − Π(1 − p_k).
  - Over ~14 monthly intervals noisy-OR saturates: the validation tau came out at ~0.9999.
  - Compare `interval_combiner: max` on real data.
- Thresholds are chosen on validation only, at a false-alarm budget. The budget (0.5 / km² / month) and
  `component_hit_fraction` (0.5) are placeholders for the owner.
- Crop tiling: 16 px gives 9 disjoint tiles; 32 px gives 4 corner crops overlapping by 16 px.
- `src.evaluate` uses the training matmul precision, so its counts equal the test pass inside `src.train`.
- Tier B and C models train with the shared focal recipe (PRD 6). Each model's own upstream recipe is noted at the
  top of its config.
- SMOKE deviations for CPU: ConvLSTM, ConvGRU and Galileo trained on 250 samples; Exchanger on 100 samples with
  batch 1. These are noted in `configs/smoke/bench_*.yaml`.

## 5. What the owner must do next

1. **Download and inspect BraDD.**
   - `python scripts/download_bradd.py --config configs/data/downloads.yaml`
   - `python scripts/inspect_bradd.py --config configs/data/bradd.yaml --root data/BraDD-S1TS --out-dir results/phase0`
   - From the report, answer: does anything geolocate the patches? Does `close_stats.pt` ship? Do the split counts
     match the paper?
   - If `close_stats.pt` ships, point both Phase 1 tracks' `data.stats_path` at it.
2. **Run Phase 1 on a GPU:** the four commands in `README.md`. Check CE 47.3 ± 2, focal 48.6 ± 2, and ours within
   1 point of the reference. If anything fails, debug before going further (PRD 7).
3. **Run Phase 2 on a GPU:** the `README.md` commands, then choose which challengers go to Phases 4–5.
4. **Download DETER and PRODES:** `python scripts/download_deter_prodes.py --config configs/data/downloads.yaml`.
   Check the printed class names and the WFS layer names (unverified).
5. **Set up Earth Engine.**
   - Set `EE_SERVICE_ACCOUNT`, `EE_KEY_FILE` and `EE_PROJECT`. Never commit the key.
   - Fill every `VERIFY` item in `configs/gee/*.yaml`: the MapBiomas Fogo Collection 5 asset and band names, the GAUL
     states layer, the RADD properties and `Date` encoding, Sentinel-1C availability, and the GCS bucket.
   - Run the Amazon pilot, review its QA report and NICFI sample, run the full export, then the Congo and Borneo
     pilots.
6. **Decide:**
   - (a) whether DETER burn scars stay positive;
   - (b) the false-alarm budget and `component_hit_fraction`;
   - (c) noisy-OR vs max interval combiner, after a first real Phase 5 run;
   - (d) whether Hansen year-end dating (1-year resolution) is acceptable for Phase 6.
7. **Verify two claims from the papers:**
   - AnySat's BraDD-S1TS IoU and protocol. A search snippet suggested ~75–90, which looks like a different metric;
     don't compare until checked.
   - The Flores-Anderson et al. 2026 figure (56% accuracy, p = 0.63) before any write-up.
8. **Optional:** supply an AnySat checkpoint (`configs/models/anysat.yaml`, `checkpoint_path`) to run Tier C AnySat.
