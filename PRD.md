# PRD: VanaDrishti, Sentinel-1 deforestation detection with early-stage evaluation

Owner: Krish Kubadia (IPD-III, DJSCE CSE-DS). Guide: Prof. Samiran Maity.
Status: implementation start, October 2026. Revision 2: Section 4 and data notes verified against the BraDD-S1TS code.
Audience: Claude Code, working inside this repository under the rules in Section 1.

---

## 1. How Claude Code must work on this project (read first, follow always)

This project runs in **ask-first mode**. Nothing changes without my explicit approval.

1. **Plan before touching anything.** At the start of every task, write a short plan: which files you will create or modify, what each change does, and what command (if any) you want to run. Then stop and wait.
2. **Show code before writing it.** For every new file or edit, show me the full content or the diff in chat first. Write it to disk only after I reply "approved" (or an equivalent clear yes). "Looks fine, but change X" means apply X and show me again.
3. **Never run anything without approval.** This includes training, evaluation, `pip`/`conda` installs, dataset downloads, Earth Engine exports, notebooks, and scripts. Do not run a file after creating it unless I ask. I run most things myself.
4. **Never touch git history.** No `git commit`, `push`, `reset`, `rebase`, or branch changes unless I ask for that exact action.
5. **Never delete files.** If something should go, tell me and I'll remove it.
6. **One phase at a time.** Phases in Section 7 have gates. Finish a phase, write a short summary of what exists and what's untested, then stop. Do not start the next phase on your own.
7. **Vendored code stays untouched.** Code copied from upstream repos (`third_party/`) is never edited in place. Wrap or subclass it in `src/`.
8. **Flag, don't guess.** If a paper detail, dataset field, or config value is ambiguous, say so and ask. Several items in this PRD are marked `VERIFY`; confirm them from the actual source (code, file metadata, paper) before relying on them.
9. **Push back.** If something in this PRD is wrong or a better option exists, tell me directly and explain why. Don't silently "fix" the design.

---

## 2. Project summary

Train a temporal-attention encoder-decoder on Sentinel-1 C-band GRD time series (VV, VH) to segment deforestation at 10 m. Use it to answer two questions:

- **Q1, stage and latency (new, primary for this semester):** How early after a clearing event does the model detect it, and which physical stage of deforestation is it actually responding to: felling (biomass still on the ground) or burning/removal (exposed soil)?
- **Q2, cross-biome transfer (from the review deck):** How much does an Amazon-trained model degrade on Congo Basin and Borneo, broken down by clearing size, and how much of the error comes from coarse labels versus the model?

The baseline architecture is fixed (Section 4). Other models are benchmarked against it under identical data, splits, loss and metrics (Section 6).

---

## 3. What "early-stage detection" means here (important, physics constraint)

The naive goal, "detect deforestation the moment trees are cut", is likely out of reach for C-band. Our own review deck cites Flores-Anderson et al. (2026): Sentinel-1 could not separate stable forest from forest that was felled with biomass still lying on the ground (56% accuracy, p = 0.63), and only became effective after burning or removal. **`VERIFY`: this statistic has not been checked against the paper's full text yet.** Until verified, treat it as a hypothesis, not a fact.

So this project does not promise felling-stage detection. It **measures** what stage and how early, using three operational definitions:

1. **Causal (online) detection.** At inference time the model sees only observations up to a cutoff date `t_c`. It never sees the future. This is what "early warning" means operationally.
2. **Detection latency.** For each event, latency = (first `t_c` at which the model's prediction crosses threshold τ) − (reference event date). Negative latency means we beat the reference system. Reference dates:
   - **DETER alert date** (INPE optical near-real-time system, Amazon only). Latency vs DETER answers "does SAR beat optical alerts?"
   - **Burn date** from MapBiomas Fogo monthly burned area (30 m, Brazil) or MODIS MCD64A1 (500 m, global, coarse). Latency vs burn date answers "is the model really detecting the burn?"
   - **RADD alert date** (Sentinel-1 statistical baseline). Latency vs RADD answers "does DL beat the operational SAR system on timing?"
3. **Stage-conditioned recall.** DETER labels clear-cuts as either *clear-cut with exposed soil* or *clear-cut with vegetation remaining*. The second class is the closest public proxy we have for "felled, biomass on ground". Recall and latency are reported separately for each class, plus for DETER burn-scar alerts.

**Hypothesis H1** (testable, from the deck's problem definition): detection latency clusters around burn dates, not felling dates; recall on the "vegetation remaining" class is much lower than on "exposed soil".
**Hypothesis H2** (secondary): early cues, if any, live at clearing edges (radar shadow/layover from the new canopy height step, per Bouvet et al. 2018) rather than in clearing interiors. Test by splitting pixels into edge (within 2 px of polygon boundary) and interior.

A negative result on H1/H2 is still a result. The deliverable is the measurement.

---

## 4. Baseline architecture (fixed, do not redesign)

**Model:** U-TAE (U-Net with Lightweight Temporal Attention Encoder), Sainte Fare Garnot & Landrieu, ICCV 2021, as adapted for deforestation by Karaman, Sainte Fare Garnot & Wegner (ISPRS Annals 2023). In the project this is called SN-TUNet when per-region normalisation is switched on (Section 4.3).

### 4.1 How the encoder detects deforestation (from Karaman et al. 2023)

- Input per sample: `x ∈ R^{T×C×H×W}` with C = 2 (VV, VH in dB), H = W = 48, T = 19–63 acquisitions (≈50 on average), same relative orbit within a sequence.
- Dates: per-acquisition day offsets relative to the first acquisition, used as the temporal positional encoding. This is how the model handles irregular Sentinel-1 revisit (6–12 days).
- **Spatial encoder** runs a shared 2D conv stack on every date independently.
- **Temporal attention (L-TAE)** runs at the lowest-resolution feature map. A learned master query per head attends over all dates and produces per-date attention weights. Those weights collapse the time axis, and the same attention masks are upsampled and reused to collapse time on every skip connection. This is genuine learned attention, not fixed statistical pooling.
- **Decoder** upsamples with skips back to 48×48 and outputs a per-pixel deforestation probability.
- Training target: binary mask; event happened somewhere inside the window. The model is never told *when*.

### 4.2 Config and training recipe (verified against the code, October 2026)

The reference implementation is https://github.com/ecovision-uzh/BraDD-S1TS (PyTorch Lightning). It ships its own copy of U-TAE in `source/utae/` (plus ConvLSTM, ConvGRU, 3D-UNet), slightly modified from `VSainteuf/utae-paps` (`out_conv=[32, 2]`, extra `seq2seq` and `aux_conv` options). **Use the BraDD-S1TS copy as the vendored backbone**, not utae-paps directly, so our numbers are comparable.

**The repo defaults do NOT reproduce the paper's best numbers.** The argparse defaults in `run_via_parser.py` differ from the paper in several places. Our configs must set every value explicitly.

| Item | Paper (Karaman et al. 2023) | Repo default (`run_via_parser.py` / constructors) | Use for reproduction |
|---|---|---|---|
| Backbone | U-TAE | `UTAE` | U-TAE |
| `encoder_widths` / `decoder_widths` | not stated | `[64,64,64,128]` / `[32,32,64,128]` | repo default |
| Encoder stages | "four blocks" | 4 stages = 1 input conv + **3** strided downsamples (k=4, s=2, p=1) | 48 → 24 → 12 → **6×6** bottleneck |
| `agg_mode`, `encoder_norm` | not stated | `att_group`, `group` | repo default |
| `n_head` | 8 is best (47.3% pixel IoU) | **16** | **8** |
| `d_model`, `d_k` | not stated | 256, 4 | repo default |
| Positional encoding period `T` | not stated | 1000 days; dates = days since earliest image/label date, starting at 1 | repo default |
| Loss | Table 1 rows: **cross-entropy**. Table 2: focal α=0.75, γ=1 gives +1.3 IoU | `CrossEntropy`; `FocalLoss` class defaults α=0.75, **γ=2.0** | Run 1: CE. Run 2: focal with γ=**1.0** set explicitly |
| Optimiser | AdamW, lr 1e-3, wd 1e-6 | AdamW, lr 1e-3, wd **0.0** | wd = **1e-6** |
| Scheduler | ReduceLROnPlateau on val IoU | ReduceLROnPlateau(mode=max, patience=10) on `ValidationScores/Epoch/IoU` | repo default |
| Early stopping | not stated | patience 30 epochs, min_delta 0.001, max 200 epochs | repo default |
| Batch size | 4 | 4 (comment: 8 needs ~9 GB GPU) | 4 |
| Normalisation | not stated | Z-score per channel, stats from train split, cached as `close_stats.pt` | repo default |
| Forward mode | not stated | `segment`, ±30 "error days" around label dates | repo default |
| Seed | not stated | `pl.seed_everything(42)` | 42 |
| Split | "15,625 train / 5,251 val / 5,112 test" | `close_set` column in `meta.csv` | `meta.csv` (see split note below) |

**Reproduction targets (corrected):**
- U-TAE, 8 heads, cross-entropy → **47.3%** pixel IoU (Table 1).
- U-TAE, 8 heads, focal α=0.75 γ=1 → about **48.6%** pixel IoU (Table 2).
- Acceptance band: ±2 points for each.

**How the label and prediction are actually built** (`forward_function.py`, `ChangeDetection` mode): each sample stores two label masks (`label[0]` at the first label date, `label[1]` at the last). The training target is the *change* between them. The scored prediction is `label[0] OR predicted_change`, compared against `label[1]`. Claude Code must reproduce this exactly in our metric code, then explain whether the OR with `label[0]` inflates IoU (the notebook shows 0.87% of pixels go 1→0, so the effect should be small, but measure it).

Resolved items (were `VERIFY` in the first draft):
- Bottleneck is 6×6, confirmed from the code: 4 encoder stages, 3 downsampling blocks. The paper's "h = H/2⁴" is a typo. The deck's "encoder depth 3, bottleneck 6×6" is correct. Still print the shapes once during Phase 1 as a sanity check.
- The split is fixed by the `close_set` column in `meta.csv`, not by a seed.
- Temporal attention is learned L-TAE (master query per head over dates), not statistical pooling.

### 4.3 Project-specific switches (off by default; baseline first)

- `per_region_norm`: per-region mean/std normalisation of dB values (needed later for cross-biome). Off for baseline reproduction.
- `seasonal_window`: restrict acquisitions to a fixed calendar window. Off for baseline reproduction.

### 4.4 Known issues in the upstream code (found by reading it; confirm each before working around it)

Do not edit `third_party/`. Work around these in `src/` and tell me each time you do.

1. **Normalisation-stats bootstrap looks broken.** In `dataset.py`, when `close_stats.pt` is missing, the code calls `BraDDS1TSDataset(path, split, 'train', ...)`. The constructor signature is `(path, phase, split)`, so this passes `phase='close'` and `split='train'`, then looks up a `train_set` column that doesn't exist. Expected result: a `KeyError` on first run unless `close_stats.pt` already ships inside the Zenodo archive. Check the archive in Phase 0. If the file is missing, compute the stats in our own code from the train split.
2. **Temporal dropout flag is inverted.** In `TemporalDropout`, `is_random=True` picks evenly spaced dates (deterministic) and `is_random=False` picks random dates. The docstring says the opposite. This matters for the temporal-depth ablation in Phase 2. Our implementation should use explicit names (`mode='uniform' | 'random'`).
3. **Split sizes disagree with the paper.** The repo notebook's `meta.csv` counts are train 15,625 / **test 5,251** / **validation 5,112**. The paper says validation 5,251 and test 5,112. Treat `meta.csv` as the truth, report both in the Phase 0 summary, and note it in our write-up.
4. **Focal loss default γ = 2.0**, while the paper's best is γ = 1.0. `run_via_parser.py` has no CLI argument for loss hyperparameters (it's commented out), so the shipped script trains with cross-entropy unless edited.
5. **Positive windows use ±30 days around label dates** (`Network._error_days = 30`). The model therefore sees up to 30 days after the last label date. Fine for the baseline, but it breaks causality for early detection (Phase 4).
6. **Padding uses value 0 after Z-score normalisation.** U-TAE detects padded dates as frames where every value equals `pad_value`. A real normalised frame being exactly all-zero is very unlikely, but add an explicit padding mask in our loader rather than relying on this.

---

## 5. Data sources

| Dataset | Role | Access | Notes |
|---|---|---|---|
| BraDD-S1TS | Baseline training/eval | Zenodo record 8060250: https://zenodo.org/record/8060250 , direct: https://zenodo.org/record/8060250/files/BraDD-S1TS_zenodo.zip?download=1 (17.7 GB, md5 `16f29111728372414c1bebfdbb156f20`) | 25,988 sequences, 48×48, VV+VH dB, PRODES-derived labels, alerts Jul 2020 – Nov 2021 |
| BraDD-S1TS code | Reference implementation (vendor this) | https://github.com/ecovision-uzh/BraDD-S1TS (MIT; also kaankaramanofficial/BraDD-S1TS) | Lightning; `run_via_parser.py` entry point; `source/utae/` holds U-TAE, ConvLSTM, ConvGRU, 3D-UNet. Defaults differ from the paper (Section 4.2) |
| U-TAE original | Reference only | https://github.com/VSainteuf/utae-paps (MIT) | Upstream of the copy above |
| Sentinel-1 GRD | New patches (early-detection set, Congo, Borneo) | GEE `COPERNICUS/S1_GRD` | IW mode, VV+VH, 10 m, dB; single relative orbit per sequence |
| PRODES (Amazon) | High-quality annual labels | TerraBrasilis: https://terrabrasilis.dpi.inpe.br/ (Downloads section, shapefile/GeoTIFF) | PRODES year runs 1 Aug – 31 Jul |
| **DETER (Amazon)** | **Event dates + stage classes for Q1** | TerraBrasilis, DETER-B Amazônia downloads | Near-daily optical alerts, ~3 ha minimum. `VERIFY` the class attribute values in the shapefile; expected names include `DESMATAMENTO_CR`, `DESMATAMENTO_VEG`, `CICATRIZ_DE_QUEIMADA`, `DEGRADACAO`, `CS_DESORDENADO`, `CS_GEOMETRICO`, `MINERACAO` |
| **MapBiomas Fogo, Collection 5** | **Burn month at 30 m (Brazil)** | https://brasil.mapbiomas.org/en/mapbiomas-fogo/ (GEE toolkit + asset IDs listed there) | Monthly burned area 1985–2025 |
| MODIS MCD64A1 v6.1 | Burn day-of-year, global | GEE `MODIS/061/MCD64A1`, band `BurnDate` | 500 m: one pixel ≈ one 48×48 patch, so use only as a patch-level burn flag outside Brazil |
| Hansen GFC v1.13 | Global annual loss labels (Congo, Borneo, label-decomposition run) | GEE `UMD/hansen/global_forest_change_2025_v1_13`; https://storage.googleapis.com/earthenginepartners-hansen/GFC-2025-v1.13/download.html | 30 m, `lossyear` 1–25 = 2001–2025 |
| RADD alerts | Non-DL SAR baseline + alert dates | GEE `projects/radar-wur/raddalert/v1` (regional images with `Alert` and `Date` bands); GFW: https://data.globalforestwatch.org/datasets/gfw::deforestation-alerts-radd/about | Comparison only, never ground truth |
| Planet NICFI / Google Earth history | Visual QA of a sample of events | NICFI program | Spot-checks only |

BraDD-S1TS file format (from the repo's `notebooks/DatasetAnalysis.ipynb` and `source/dataset.py`):
- `meta.csv` columns: `alert_idx`, `center_idx`, `date` (the PRODES alert date), `sampling_type` (`positive` 8,712 / `boundary` 2,294 / `oldDeforest` 7,557 / `forest` 5,472 / `herbaceous` 1,780 / `agriculture` 152 / `shrubs` 21), `state` (9 Brazilian states), `file`, `close_set` (train/validation/test).
- `Samples/<file>.pt` is a dict: `image_dates` (list of dates, length T), `label_dates` (list, length 2), `image` float32 `[T, 2, 48, 48]` (VV, VH in dB), `label` int64 `[2, 48, 48]` (mask at first and last label date).
- Pixel transitions across the dataset: no change 83.3%, forest→deforested 15.9%, deforested→forest 0.87%.
- **No coordinates are visible in `meta.csv` or in the sample files.** `alert_idx` and `center_idx` look like internal indices. Whether a coordinate table exists elsewhere in the Zenodo archive is the main Phase 0 question (Section 7).

Data notes Claude Code must respect:
- Sentinel-1B failed in December 2021, so 2022–2024 Amazon time series are at 12-day revisit. `VERIFY` Sentinel-1C availability in GEE for 2025+ before designing windows that rely on it.
- BraDD positives cover from 1 year + 2 weeks before the PRODES alert to 2 weeks after it, so the clearing happened somewhere inside, at an unknown date. **BraDD alone cannot train or evaluate latency.** That is why Phase 3 builds a dated set.
- PRODES alert dates cluster in July–August because cloud blocks optical labelling in the wet season. Never use PRODES dates as event dates.

---

## 6. Models to benchmark against the U-TAE baseline

All models use the same data loaders, splits, focal loss with α=0.75 and γ=1.0 set explicitly (unless the model's own recipe demands otherwise, flagged in results), metric code, and early-detection protocol. U-TAE is the reference row in every table.

| Tier | Model | Why include it | Code |
|---|---|---|---|
| A (cheap, already in Karaman's table) | ConvLSTM, ConvGRU, 3D-UNet | Reproduces the published comparison (pixel IoU 44.6 / 41.8 / 45.8); sanity check for our pipeline | Already in the vendored repo: `source/utae/convlstm.py`, `convgru.py`, `unet3d.py`. Upstream's `Network` factory only registers `UTAE`, so add the others in our `build_model` |
| A (free ablation) | U-TAE `seq2seq=True` | Upstream adds a full self-attention layer over dates (`sLTAE`) before the L-TAE. Cheap test of whether richer temporal modelling helps | `source/utae/sltae.py`; flag only, same backbone |
| B (stronger SITS architectures) | TSViT (Tarasiou et al., CVPR 2023) | Fully attentional temporal-then-spatial transformer with acquisition-date positional encoding; strong on PASTIS | https://github.com/michaeltrs/DeepSatModels |
| B | Exchanger + U-Net (Cai et al. 2023) | Collect-update-distribute temporal encoder, beats U-TAE on PASTIS; drops into a U-Net | https://github.com/TotalVariation/Exchanger4SITS |
| C (pretrained foundation model) | AnySat (CVPR 2025) | Already evaluated on BraDD-S1TS by its authors; handles S1 time series natively; fine-tune and linear-probe both | https://github.com/gastruc/AnySat |
| C (optional) | Galileo (nasaharvest) | Small pretrained model with S1 VV/VH inputs; useful as a low-label transfer candidate for Congo/Borneo | https://github.com/nasaharvest/galileo |
| Non-DL reference | RADD alerts (RADD+) | Operational SAR system; IoU and latency comparison | GEE asset above |

`VERIFY` for AnySat: pull the exact BraDD-S1TS IoU and protocol from the paper's appendix table before quoting it anywhere. If their split differs from Karaman's, say so.

Expectations to keep honest: TSViT and Exchanger were tuned on optical crop data with 128×128 patches; 48×48 SAR patches may not suit them. A benchmark where U-TAE wins is a valid outcome.

---

## 7. Phases and gates

Each phase ends with: a summary of what was built, what was run (by me), what's untested, and open questions. **Then stop.**

### Phase 0: Setup and data inspection
- Propose repo layout (Section 8), `environment.yml` (the upstream code needs `lightning`, `torch`, `torchvision`, `pandas`; pin versions), and vendoring of the BraDD-S1TS repo into `third_party/bradd_s1ts/`. Keep `utae-paps` only as a reference copy.
- After I download BraDD, write a read-only inspection script. Confirm the format in Section 5 rather than assume it: keys, shapes, dtypes, dB value ranges, T distribution, label-date spacing relative to the `date` column, split counts per `close_set`.
- **Key question: does anything in the archive geolocate the patches?** List every file in the unzipped archive, not just `meta.csv` and `Samples/`. Also check whether `close_stats.pt` ships (Section 4.4, issue 1).
  - If coordinates exist: Phase 3 can attach DETER, MapBiomas Fogo and RADD dates to existing BraDD patches, which saves an Earth Engine export.
  - If not: Phase 3 needs a fresh GEE export. The PRODES alert date in `meta.csv` is still useful as an upper bound on when each event happened.
- Gate: I review the inspection report.

### Phase 1: Baseline reproduction (U-TAE on BraDD-S1TS)
- Two tracks, in this order:
  1. **Reference run.** A thin launcher in `src/` that calls the vendored upstream code with every Section 4.2 value passed explicitly (8 heads, wd 1e-6, CE). No reimplementation. This tells us what the upstream code really scores on our hardware.
  2. **Our pipeline.** Our own dataset class, training loop and metrics in `src/`. These must reproduce the upstream number before we change anything, because Phases 3–6 need features upstream doesn't have (prefix truncation, latency, size bins).
- Dataset class with explicit padding mask, date offsets, `meta.csv` split, Z-score stats from the train split.
- Training on the Section 4.2 recipe; config via YAML; seed 42; logging to CSV (W&B off by default).
- Metric module: pixel and patch IoU/precision/recall from a confusion matrix accumulated over the whole set (upstream `score.py` already does this; match it), plus the `label[0] OR prediction` scoring rule from Section 4.2.
- Unit tests for the metric module and dataset shapes.
- **Acceptance:** CE run within ±2 points of 47.3% pixel IoU, focal γ=1 run within ±2 points of 48.6%, and our pipeline within 1 point of the reference run. If any fails, stop and debug before anything else.
- Gate: I review results.

### Phase 2: Benchmark suite on BraDD-S1TS
- Add Tier A models first, then Tier B, then Tier C, each behind a common `build_model(cfg)` interface.
- Report pixel IoU/P/R, patch IoU/P/R, params, train time per epoch, peak GPU memory.
- Also report the temporal-depth ablation for U-TAE (1, 2, 5, 10, all dates) to confirm the published 3.3% / 36.0% / 47.3% shape. Upstream implements this with `TemporalDropout`, which always keeps the first and last date. Watch the inverted flag in Section 4.4, issue 2. Also note that the paper's single-date case uses only the last date, which the upstream code (always keeping first and last) can't produce; implement that case separately.
- Gate: I choose which challengers go forward to Phases 4–5 (likely U-TAE + best two).

### Phase 3: Dated early-detection dataset (Amazon)
- GEE export pipeline (Python `earthengine-api`), producing BraDD-compatible patches so loaders are reused.
- Positives: patches around DETER alerts in a chosen period (propose 2020–2023 to overlap BraDD; flag the S1B gap). Store per patch: DETER alert date, DETER class, MapBiomas Fogo burn month per pixel, RADD alert date per pixel, PRODES year, polygon area.
- Window: from ~12 months before to ~4 months after the DETER date, so prefix cutoffs span before and after the event.
- Negatives: same three types as BraDD (non-forest vegetation, already-cleared, stable forest), with matched window lengths and date distribution.
- Spatial split by region blocks (not random patches) so train and test patches don't share clearings.
- Same preprocessing as BraDD (GEE default GRD chain, dB, single relative orbit). Any difference here will read as a model effect later.
- Gate: I review export code before any export runs, then review a QA sample (I'll spot-check against NICFI).

### Phase 4: Early-detection training (architecture unchanged)
- **Prefix-truncation training.** For each training sample, draw a cutoff `t_c` and drop all acquisitions after it. Pixel label = 1 if the event reference date ≤ `t_c`, else 0. This teaches the unchanged U-TAE to answer "has clearing happened by now?" from past data only.
- **Reuse upstream's multi-date label design.** The upstream code already supports more than two label dates per sample: the `segment` forward mode predicts change for every interval between consecutive `label_dates`, and `Targets` is `[t, H, W]`. For the dated dataset, store cumulative masks at several dates (for example monthly, built from DETER and burn dates), so one sample yields several "change between date k and k+1" targets. **Set the trailing error margin to 0** (upstream uses ±30 days, Section 4.4, issue 5), otherwise the model sees the future and the latency numbers are invalid. A leading margin is fine.
- Two label anchors, trained separately, compared: (a) DETER date, (b) burn month where available. The difference between them is part of the H1 evidence.
- Optional, propose only, do not implement without approval: an auxiliary onset head that regresses event date from the attention weights. This changes the architecture slightly, so it is a separate ablation, never the baseline.
- Gate: I review results.

### Phase 5: Early-detection evaluation
- For each test event, run inference at every available `t_c` (sliding prefix). Record probability curves per pixel and per patch.
- Report:
  - Latency distribution (median, IQR) vs DETER date, vs burn date, vs RADD date.
  - Recall at +0, +12, +24, +48, +90 days after the DETER date.
  - Everything split by DETER class (exposed soil / vegetation remaining / burn scar), by clearing size bin (<0.5 ha, 0.5–2 ha, >2 ha), and by edge vs interior pixels.
  - False alarms per km² per month on stable-forest negatives, with τ chosen on validation at a fixed false-alarm budget (not chosen on test).
- Plots: probability vs days-since-event, stratified by stage; latency histograms.
- Gate: I review.

### Phase 6: Cross-biome transfer (Congo Basin, Borneo)
- Same GEE pipeline, Hansen v1.13 labels, RADD dates as the timing reference (no DETER outside Brazil).
- Apply Amazon-trained weights unchanged. Report IoU/F1 per size bin and latency vs RADD.
- Label decomposition: train identical Amazon models on PRODES vs Hansen labels, test both against PRODES; the IoU gap is the Hansen label penalty.
- Mitigations after the raw numbers exist: per-region normalisation, smaller patches (32×32, 16×16) for Congo, fine-tuning sample-count curve.
- Gate: I review.

NE India (Mizoram pilot) is out of scope for this PRD.

---

## 8. Repository layout (proposal, confirm in Phase 0)

```
vanadrishti/
├── PRD.md
├── CLAUDE.md                # I maintain this; read it every session
├── environment.yml
├── configs/
│   ├── baseline_utae.yaml
│   ├── models/              # one YAML per challenger
│   └── early/               # prefix-truncation experiments
├── third_party/             # vendored upstream code, never edited
│   ├── utae_paps/
│   └── bradd_s1ts/
├── src/
│   ├── data/                # bradd_dataset.py, dated_dataset.py, transforms.py
│   ├── models/              # build_model.py + thin wrappers
│   ├── losses/              # focal.py
│   ├── metrics/             # segmentation.py, latency.py
│   ├── train.py
│   ├── evaluate.py
│   └── early_eval.py        # sliding-prefix inference + latency
├── gee/                     # Earth Engine export scripts
├── notebooks/               # inspection and plotting only
├── tests/
└── results/                 # CSVs and figures, git-ignored if large
```

Coding conventions: Python 3.11, PyTorch, type hints, docstrings on public functions, no hard-coded paths (everything through config), deterministic seeds, small functions. Notebooks only for inspection and plotting, never for training logic.

---

## 9. Metrics, precisely

- **Pixel IoU** = TP / (TP + FP + FN) for the positive class, accumulated over the whole evaluation set.
- **Patch level**: a patch is positive if it has ≥1 positive pixel; same for predictions (Karaman's definition).
- **F1, precision, recall** of the positive class. Never report overall accuracy as a headline number (deforested pixels are a small minority).
- **Latency** (Section 3) in days; events never detected within the window are reported as a separate "missed" count, not dropped.
- **Size bins** from connected components of the reference mask, area in hectares.
- **Minimum mapping unit** stated explicitly in every results table.

---

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| Baseline won't reproduce 47.3% | Phase 1 reference run on the upstream code with explicit paper settings first; then compare our pipeline against it |
| Upstream defaults silently differ from the paper | Never rely on argparse defaults; every config value set explicitly in YAML (Section 4.2) |
| DETER dates are optical, so they lag the real event in the wet season | Report latency against three references, not one; treat DETER as an upper bound on true event date |
| "Vegetation remaining" class is rare | Report counts; if too few, pool years or report as qualitative |
| MCD64A1 too coarse outside Brazil | Use only as patch-level flag; rely on RADD dates for timing |
| Flores-Anderson statistic unverified | Verify before any write-up; H1 framing holds either way since we measure it |
| GPU limits | U-TAE is ~1M params with batch 4 on 48×48; fits a single consumer GPU or Colab. Foundation models may need gradient checkpointing or linear probing only |
| Earth Engine quotas / export time | Export in tiles, log task IDs, resume-safe scripts |

## 11. Out of scope

Interferometric coherence (needs SLC), optical fusion, operational alert delivery, driver attribution, biomass estimation, selective logging, NE India deployment.

## 12. References

1. Karaman, K., Sainte Fare Garnot, V., Wegner, J. D. Deforestation detection in the Amazon with Sentinel-1 SAR image time series. ISPRS Annals X-1/W1-2023, 835–842, 2023. https://isprs-annals.copernicus.org/articles/X-1-W1-2023/835/2023/
2. Sainte Fare Garnot, V., Landrieu, L. Panoptic segmentation of satellite image time series with convolutional temporal attention networks. ICCV 2021. https://github.com/VSainteuf/utae-paps
3. Tarasiou, M., Chavez, E., Zafeiriou, S. ViTs for SITS: Vision Transformers for Satellite Image Time Series. CVPR 2023.
4. Cai, X., Bi, Y., Nicholl, P., Sterritt, R. Revisiting the encoding of satellite image time series. 2023. arXiv:2305.02086
5. Astruc, G., Gonthier, N., Mallet, C., Landrieu, L. AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities. CVPR 2025. arXiv:2412.14123
6. Reiche, J. et al. Forest disturbance alerts for the Congo Basin using Sentinel-1. Environmental Research Letters 16(2), 024005, 2021.
7. Bouvet, A. et al. Use of the SAR shadowing effect for deforestation detection with Sentinel-1 time series. Remote Sensing 10(8), 1250, 2018.
8. Flores-Anderson, A. et al. (2026), Remote Sensing of Environment 333. `VERIFY` full citation and the 56% / p = 0.63 figure.
9. Hansen, M. C. et al. High-resolution global maps of 21st-century forest cover change. Science 342, 850–853, 2013 (GFC v1.13).
