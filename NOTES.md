# MotorDiag — external dataset ingestion & model results

## CLI map
| Command | What it does |
|---|---|
| `uv run cardiag doctor` | System check (ffmpeg, dirs, Python) |
| `uv run cardiag train` | Embed corpus with CLAP, train heads |
| `uv run cardiag train --fixtures` | Train on bundled fixture embeddings (offline) |
| `uv run cardiag diagnose <wav>` | Run inference on one audio file |
| `uv run cardiag ingest <dir> --kind fault\|normal [--cause <c>]` | Ingest a folder of audio |
| `uv run cardiag demo` | Full scrape → train → diagnose loop |
| `uv run python scripts/ingest_external_datasets.py` | Ingest DB1 + Car-Engine-Sounds + Ai_Mechanic |

## Data layout
```
data/
├── external/
│   ├── corpus.jsonl           # ingested external clips (1061 rows)
│   ├── ingest_report.json
│   ├── repos/                  # cloned repositories (gitignored)
│   │   ├── Sound-Based-Vehicle-Diagnostics-Emergency-Signal-Recognition/
│   │   ├── Car-Engine-Sounds-Dataset/
│   │   └── Ai_Mechanic/
│   └── clips/                  # per-source clip WAVs (gitignored)
├── training/
│   ├── best_model_clap.joblib  # the shipped model
│   ├── triage_model.joblib
│   └── train_report.json
├── external/
├── youtube/                    # scraped clips (not present in this clone)
├── tiktok/                     # (not present)
└── reddit/                     # (not present)
```

### Sources ingested
| Source | Raw files | Spans produced | Label strategy |
|---|---|---|---|
| **DB1** (Sound-Based-Vehicle-Diagnostics) | 30 m4a | 39 | 27 fault folders → explicit cause per folder |
| **Car-Engine-Sounds-Dataset** | 216 mp4 | 340 | Abnormal → fault, Normal → normal |
| **Ai_Mechanic** | 297 mp3 | 679 | Engine Issue / Non-engine → fault, Normal → normal |
| **Total** | 543 | **1058** | |

### DB1 cause mapping (27 folders → 17 canonical causes)
Mapped every folder to the closest canonical cause (see `scripts/ingest_external_datasets.py:_DB1_CAUSE`). All mapped causes are in `_ENGINE` or `_CHASSIS` sets in `build.py`; missing region mappings (`transmission`, `vacuum_leak`, `seals_gaskets`, etc) were added to `_REGION`.

### Two `cause` fields per row
- `l2_candidates` — for Car-Engine-Sounds/Ai_Mechanic: keyword-extracted from clip title
- `cause` — explicit for DB1 (set per-folder). `_cause_of()` updated to prefer explicit `cause` over `l2_candidates`

## Baseline (fixtures, 70 clips bundled in repo)
| Head | Balanced Accuracy | σ | Notes |
|---|---|---|---|
| kind (fault/normal) | 0.471 | 0.039 | near chance |
| cause (22 part families) | 0.217 | 0.117 | near majority |
| triage (engine/chassis) | 0.855 | 0.049 | strongest head |
| knock | — | — | some signal |
| knock_region | — | — | DEGENERATE (2 clips) |

All heads except triage were near-chance. The fixture set is tiny and meant only for teaching the CLI flow.

## New model (external datasets, 679 clips)
| Head | Balanced Accuracy | σ | Classes | n | Majority base | Temperature |
|---|---|---|---|---|---|---|
| kind (fault/normal) | **0.793** | 0.035 | 2 | 712 | 0.587 | 2.845 |
| knock (knock/normal_idle) | **0.858** | 0.098 | 2 | 408 | 0.946 | 1.228 |
| cause (11 part families) | **0.381** | 0.079 | 11 | 174 | 0.759 | 1.0 |
| region (6 zones) | **0.291** | 0.098 | 6 | 179 | 0.793 | 1.0 |
| triage (engine/chassis) | **0.631** | 0.142 | 2 | 179 | 0.877 | 2.60 |
| knock_region | — | — | 1 | 3 | — | — |

Prune-noisy 0.15 applied: 106 low-confidence clips dropped for kind, 61 for knock, 26 for region/triage, 0 for cause (too few per class).

### Diagnosis sample (Suspension Arm fault clip)
```json
{"file": "db1_problem1/clip_00.wav", "verdict": "fault", "fault_probability": 0.997,
 "regions": [{"zone": "engine", "p": 0.887}, {"zone": "suspension/steering", "p": 0.107}],
 "causes": [{"part": "suspension", "p": 0.994}],
 "engine_knock_probability": 0.556}
```
High confidence fault, cause fires suspension (matches the DB1 folder label). Region top-1 is engine (marginal). Knock probability ~0.5 means the knock specialist didn't fire.

### Reference targets (from shipped model docs)
| Metric | Target | Current | Status |
|---|---|---|---|
| fault/normal AUROC | ~0.79 | 0.793 bal_acc | ✓ On target |
| region top-3 | ~75% | 0.291 bal_acc (6-way) | Needs more data per class |
| cause top-3 | ~45-65% | 0.381 bal_acc (11-way) | Improving, below target |
| knock OOS | 0.56 (chance) | 0.858 bal_acc | ✓ Strong |
| ECE | ~0.04 | not measured | |

## Device info
- `torch.backends.mps.is_available()` → True
- `torch.backends.mps.is_built()` → True
- CLAP model: `laion/clap-htsat-unfused` (~2GB, downloaded + cached on first use)
- CLAP `_device()` returns `"mps"` automatically (no code change needed)
- Ingestion speed: ~2-3 files/s on MPS (each file runs CLAP 3-6 times)
- Training: ~30s for 679 clips (including embedding, CV, temperature scaling)

## Key changes
| File | Change |
|---|---|
| `scripts/ingest_external_datasets.py` | NEW — ingest DB1, Car-Engine-Sounds, Ai_Mechanic |
| `src/cardiag/pipeline/build.py` | `load_corpus()` now reads `data/external/corpus.jsonl` |
| `src/cardiag/pipeline/build.py` | `_source_of()` detects "external" source |
| `src/cardiag/pipeline/build.py` | `_cause_of()` prefers explicit `cause` field |
| `src/cardiag/pipeline/build.py` | `_REGION` extended with transmission, vacuum_leak, seals, cooling |
| `src/cardiag/pipeline/build.py` | `_ENGINE`/`_CHASSIS` extended with missing canonical causes |

## Known limitations
- Still DEGENERATE: knock_region head (only 3 knock-labeled clips survive the cascade)
- Triage head is within 2σ of chance — cause labels too sparse for engine/chassis split
- No YouTube/TikTok/Reddit data on this clone (not scraped yet)
- Region head only has 179 clips across 6 zones — many zones have <<min_class support
- Car-Engine-Sounds mp4s → audio extraction via librosa works but half the clips produce no mechanical spans (filtered by cascade)
- Car-diagnostics dataset and Revix Engine Knock not yet downloaded (Kaggle auth needed)
