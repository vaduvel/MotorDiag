# MotorDiag — external dataset ingestion & model results

## CLI map
| Command | What it does |
|---|---|
| `uv run cardiag doctor` | System check (ffmpeg, dirs, Python) |
| `uv run cardiag train` | Embed corpus with CLAP, train heads |
| `uv run cardiag train --prune-noisy 0.15 --weak-label-sources <s>` | Train with label cleaning + source guard |
| `uv run cardiag train --fixtures` | Train on bundled fixture embeddings (offline) |
| `uv run cardiag diagnose <wav>` | Run inference on one audio file |
| `uv run cardiag ingest <dir> --kind fault|normal [--cause <c>]` | Ingest a folder of audio |
| `uv run python scripts/ingest_external_datasets.py` | Ingest DB1 + Car-Engine-Sounds + Ai_Mechanic |
| `uv run python scripts/ingest_kaggle_datasets.py` | Ingest Revix + Car Diagnostics from Kaggle |
| `python -m cardiag.training.eval.scorecard` | By-video CV scorecard |
| `python -m cardiag.training.eval.confound_probe` | LOSO domain-adaptation probe |

## Data layout
```
data/
├── external/
│   ├── corpus.jsonl           # main external corpus (2763 rows)
│   ├── ingest_report.json
│   ├── repos/                  # cloned repositories (gitignored)
│   └── clips/                  # per-source clip WAVs (gitignored)
├── revix/
│   ├── corpus.jsonl           # Revix Engine Knock clips (1161 rows)
│   └── clips/
├── car_diagnostics/
│   ├── corpus.jsonl           # 862 clips (NOT read by load_corpus — see "orphaned clips")
│   └── clips/
├── kaggle/                    # raw downloads (gitignored)
│   ├── engine_sounds_hf_dataset/
│   └── car diagnostics dataset/
├── youtube/                    # scraped clips (275)
├── tiktok/                     # scraped clips (14)
├── training/
│   ├── best_model_clap.joblib  # the shipped model
│   ├── triage_model.joblib
│   ├── train_report.json
│   └── corpus_emb.npz          # CLAP embedding cache
└── reddit/                     # (not scraped)
```

## Corpus composition

| Source | Clip ID prefix | n clips | License | Label quality |
|---|---|---|---|---|
| **DB1** | `db1_` | 39 | CC BY-SA 4.0 | Clean (27 folders → 17 canonical causes) |
| **Car-Engine-Sounds** | `car_engine_` | 301 | CC BY-SA 4.0 | Moderate (title keyword extraction) |
| **Ai_Mechanic** | `ai_mechanic_` | 339 | Apache 2.0 | Moderate (title keyword extraction) |
| **Revix Engine Knock** | `revix_` | 1,161 | Apache 2.0 | Clean (per-folder: car/truck + clean/knocking) |
| **Car Diagnostics** | `cdd_` | 923 | unknown | Clean (per-folder: brakes, belt, PS, ignition, etc.) |
| **YouTube scrape** | youtube path | 275 | scraped | Weak (clickbait titles) |
| **TikTok scrape** | tiktok path | 14 | scraped | Weak |
| **Total in corpus** | | **3,052** | | |

## Baseline (fixtures, 70 clips bundled in repo)
| Head | Balanced Accuracy | σ | Notes |
|---|---|---|---|
| kind (fault/normal) | 0.471 | 0.039 | near chance |
| cause (22 part families) | 0.217 | 0.117 | near majority |
| triage (engine/chassis) | 0.855 | 0.049 | strongest head |

## Final model — 3,052 clips, weak-source guard active

`cardiag train --prune-noisy 0.15 --weak-label-sources youtube,tiktok`

YouTube/TikTok excluded from label heads (kind, cause, region, triage) but kept for knock/knock_region.

| Head | Balanced Accuracy | σ | Classes | n | Majority base | Temperature |
|---|---|---|---|---|---|---|
| kind (fault/normal) | **0.830** | 0.015 | 2 | 2,796 | 0.546 | 1.81 |
| knock (knock/normal_idle) | **0.952** | 0.027 | 2 | 1,059 | 0.934 | 1.42 |
| cause (12 part families) | **0.445** | 0.062 | 12 | 1,282 | 0.690 | 1.0 |
| region (6 zones) | **0.666** | 0.062 | 6 | 1,231 | 0.726 | 1.0 |
| triage (engine/chassis) | **0.896** | 0.028 | 2 | 1,231 | 0.830 | 1.995 |
| knock_region | 0.483 | 0.094 | 3 | 20 | 0.650 | 1.0 |

### A/B: weak-source guard impact

| Head | All 3,052 clips | Guard: exclude YT/TT | Δ |
|---|---|---|---|
| kind | 0.775 | **0.830** | +0.055 ✅ |
| knock | 0.952 | **0.952** | 0 (YT/TT always kept for knock) |
| cause | 0.340 | **0.445** | +0.105 ✅ |
| region | 0.557 | **0.666** | +0.109 ✅ |
| triage | 0.832 | **0.896** | +0.064 ✅ |

YT/TT's weak supervision polluted label heads; their real-world audio still helps knock.

## Scorecard — by-video CV, per-source held-constant

`python -m cardiag.training.eval.scorecard`

| Head | n | balAcc | AUROC | majority |
|---|---|---|---|---|
| knock (knock vs normal-idle) | 1,015 | 0.956±0.04 | 0.998±0.00 | 0.937 |
| kind \| all sources | 3,052 | 0.781±0.03 | 0.864±0.03 | 0.580 |
| **kind \| cdd (source held constant)** | 923 | **0.929±0.02** | **0.978±0.01** | 0.568 |
| kind \| revix (source held constant) | 1,161 | 0.842±0.02 | 0.908±0.02 | 0.503 |
| kind \| car_engine (source held constant) | 301 | 0.833±0.05 | 0.913±0.03 | 0.528 |
| kind \| ai_mechanic (source held constant) | 339 | 0.792±0.05 | 0.873±0.05 | 0.631 |
| kind \| youtube (source held constant) | 275 | 0.456±0.08 | 0.285±0.25 | 0.967 |
| triage (engine vs running-gear) | 1,437 | 0.837±0.07 | 0.903±0.07 | 0.827 |
| cause (top-3 shortlist) | 1,526 | **0.862** | — | 0.594 |

Source confound: predicting source from embedding = balAcc 0.533±0.05 — shortcut exists. Trust source-held-constant rows, not all-sources.

YouTube permutation: p=0.662 (chance — 266 fault / 9 normal, hopeless split).

## LOSO (leave-one-source-out) — confound probe

`python -m cardiag.training.eval.confound_probe`

5 actionable domains: `ai_mechanic`, `car_engine`, `cdd`, `revix`, `youtube`
(db1 and tiktok excluded — one-class or too small for held-out split).

| Transform | all-sources balAcc | LOSO balAcc (mean) | LOSO AUROC (mean) | source predictability |
|---|---|---|---|---|
| baseline (raw embeddings) | 0.780±0.03 | **0.615** | 0.658 | 0.536±0.04 (chance 0.143) |
| A: per-source standardise | 0.802±0.03 | 0.597 | 0.645 | 0.143±0.00 |
| B: source-nullspace projection | 0.781±0.03 | 0.615 | 0.661 | 0.533±0.06 |

LOSO baseline rose from **0.597** (3 sources) to **0.615** (5 sources, Kaggle added).
Neither DA transform improved LOSO past baseline — confirming the original conclusion that domain adaptation with existing data alone can't recover LOSO, but adding independent sources (Kaggle) makes marginal gains.

### Per-held-out-source detail (baseline)

| Held-out source | n_test | balAcc | AUROC | majority |
|---|---|---|---|---|
| cdd | 923 | 0.689 | 0.743 | 0.568 |
| ai_mechanic | 339 | 0.684 | 0.767 | 0.631 |
| car_engine | 301 | 0.628 | 0.673 | 0.528 |
| youtube | 275 | 0.537 | 0.536 | 0.967 |
| revix | 1,161 | 0.536 | 0.569 | 0.503 |

Revix holds back LOSO — homogeneous engine-internal + normal from one car family. YouTube still suffers from 9:266 normal/fault imbalance. cdd (Car Diagnostics) is the strongest held-out performer.

## Historical progression

### External-only (679 clips, 3 datasets before scraping)
| Head | balAcc |
|---|---|
| kind | 0.793 |
| cause | 0.381 |
| knock | 0.858 |

### +YouTube +TikTok (968 clips, noisy labels)
| Head | balAcc |
|---|---|
| kind | 0.727 ↓ (YT/TT polluted labels) |
| knock | 0.914 ↑ (real audio helped) |
| cause | 0.236 ↓ |

### +Kaggle, all 3,052 (no source guard)
| Head | balAcc |
|---|---|
| kind | 0.775 (partial recovery from Kaggle) |
| knock | 0.952 |
| cause | 0.340 |
| triage | 0.832 |

### +Kaggle + weak-source guard (current best model)
| Head | balAcc |
|---|---|
| kind | **0.830** |
| knock | **0.952** |
| cause | **0.445** |
| region | **0.666** |
| triage | **0.896** |

Reference targets: fault/normal AUROC ~0.79 (DCASE ceiling), cause top-3 ~45-65%, region top-3 ~75%.

## Known issues

- **Orphaned car_diagnostics clips (862)**: `data/car_diagnostics/corpus.jsonl` has 862 clips with `car_diagnostics_` prefix, but `load_corpus()` does not read `data/car_diagnostics/`. Only the 923 `cdd_` clips in `data/external/corpus.jsonl` are consumed. The two ingestion runs may have overlapped audio.
- **clip_id prefix inconsistency**: Kaggle Car Diagnostics ingested with `cdd_` prefix (via an earlier attempt) rather than `car_diagnostics_`. Source attribution uses `cdd` in `_EXTERNAL_SOURCES`.
- knock_region still degenerate (20 clips, ~chance)
- YouTube confound-free cut: 266 fault / 9 normal → perm p=0.66, structurally broken split

## Key changes
| File | Change |
|---|---|
| `src/cardiag/pipeline/build.py` | `load_corpus()` reads `external/` + YT + TT corpus.jsonl |
| `src/cardiag/pipeline/build.py` | `_source_of()` detects `external`, `youtube`, `tiktok`, `reddit` |
| `src/cardiag/pipeline/build.py` | `_cause_of()` prefers explicit `cause` over `l2_candidates` |
| `src/cardiag/pipeline/build.py` | `_self_confidence()` single-class fix for StratifiedGroupKFold |
| `src/cardiag/pipeline/build.py` | `train()` accepts `weak_label_sources` — source-level guard |
| `src/cardiag/pipeline/build.py` | `_label_audio()` reads description/transcript/OCR for cause matching |
| `src/cardiag/cli.py` | `--weak-label-sources` & `--prune-noisy` flags |
| `src/cardiag/config.py` | Romanian/European queries for YouTube + TikTok |
| `src/cardiag/ingest/tiktok/discover.py` | Romanian queries added |
| `src/cardiag/training/eval/scorecard.py` | `source_of()` with `_EXTERNAL_SOURCES`; per-source held-constant cuts |
| `src/cardiag/training/eval/confound_probe.py` | `_EXTERNAL_SOURCES` extended for Kaggle domain attribution |
| `src/cardiag/training/eval/external_scorecard.py` | Renamed → within-source + LOSO at source-time (PR #1) |
| `src/cardiag/training/eval/head_tuning.py` | threshold + C tuning headroom probe (PR #4) |
| `scripts/ingest_external_datasets.py` | DB1 + Car-Engine-Sounds + Ai_Mechanic ingestion |
| `scripts/ingest_kaggle_datasets.py` | Revix + Car Diagnostics ingestion through cascade |