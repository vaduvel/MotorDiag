"""Ingest external datasets (DB1, Car-Engine-Sounds, Ai_Mechanic) into the
training corpus. Each source writes to ``data/external/corpus.jsonl`` via the
same cascade that scrape uses (``_label_audio``), so train/serve are consistent.

Usage::

    uv run python scripts/ingest_external_datasets.py

Requires the cloned repos under ``data/external/repos/``.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from cardiag import paths
from cardiag.pipeline.build import _label_audio, _write_corpus, _clap

REPOS = paths.DATA / "external" / "repos"

_DB1_CAUSE = {
    "bad CV joint": "cv_joint",
    "bad transimision": "transmission",
    "bad wheal bearing": "wheel_bearing",
    "clunking over bumps bad stabilizer link noise": "suspension",
    "engine chriping -squealing belt": "belt",
    "engine rattle noise": "engine_internal",
    "Engine Misfire": "fuel_ignition",
    "flooded engin": "fuel_ignition",
    "Fuel Pump Cartridge fault": "fuel_pump",
    "knocking": "rod_knock",
    "lifter ticking": "valvetrain",
    "loose exhaust shield": "exhaust",
    "muffler running loud exhaust leak": "exhaust",
    "pre-ignition": "fuel_ignition",
    "problem 6": "engine_internal",
    "Radiator Fan Failure": "accessories",
    "seized engin": "engine_internal",
    "squeaky belt": "belt",
    "squeaky brake -grinding brake": "brakes",
    "stearing groaning whining low power steering fluid": "power_steering",
    "stearing noise": "power_steering",
    "Strut Mount Failure": "suspension",
    "Suspension Arm fault": "suspension",
    "thrown rod": "rod_knock",
    "turning front end clicking bad CV axle": "cv_axle",
    "Universal Joint Failure or Steering Rack Failure": "cv_axle",
    "vacuum leak": "vacuum_leak",
}


def _list_audio(d: Path, exts=("mp3", "wav", "m4a", "mp4", "ogg", "flac")):
    return sorted([Path(p) for ext in exts
                   for p in glob.glob(str(d / f"**/*.{ext}"), recursive=True)])


def ingest_source(label: str, files: list[tuple[Path, str, str | None]],
                  clap) -> int:
    out_base = paths.DATA / "external"
    recs = []
    t0 = time.time()
    for i, (f, kind, cause) in enumerate(files):
        vid = f"{label}_{f.stem}"[:60]
        spans = 0
        try:
            new = _label_audio(f, vid, f.stem, kind, out_base, clap, cause=cause)
            spans = len(new)
            recs.extend(new)
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] ERROR {f.name}: {e}")
        if (i + 1) % 20 == 0 or i == len(files) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{label} {i+1}/{len(files)}] {spans} spans, "
                  f"{len(recs)} total, {rate:.1f} files/s", flush=True)
    n = _write_corpus(recs, out_base)
    print(f"  {label}: {n} clips -> {out_base/'corpus.jsonl'}")
    return n


def ingest_db1(clap):
    db1 = REPOS / "Sound-Based-Vehicle-Diagnostics-Emergency-Signal-Recognition" / "Datasets" / "DB1"
    if not db1.exists():
        return 0
    files = []
    for folder in sorted(db1.iterdir()):
        if not folder.is_dir():
            continue
        cause = _DB1_CAUSE.get(folder.name)
        if cause is None:
            print(f"  [DB1] no cause mapping for '{folder.name}', skipping")
            continue
        for f in _list_audio(folder):
            files.append((f, "fault", cause))
    return ingest_source("db1", files, clap)


def ingest_car_engine(clap):
    base = REPOS / "Car-Engine-Sounds-Dataset"
    if not base.exists():
        return 0
    files = []
    for label, kind in [("Abnormal Car Sounds", "fault"),
                         ("Normal Car Sounds", "normal")]:
        folder = base / label
        if not folder.exists():
            continue
        for f in _list_audio(folder):
            files.append((f, kind, None))
    return ingest_source("car_engine", files, clap)


def ingest_ai_mechanic(clap):
    base = REPOS / "Ai_Mechanic" / "ML dataset" / "Audio"
    if not base.exists():
        return 0
    files = []
    for label, kind in [("Engine Issue", "fault"),
                         ("Non engine issue", "fault"),
                         ("Normal engine", "normal")]:
        folder = base / label
        if not folder.exists():
            continue
        for f in _list_audio(folder):
            files.append((f, kind, None))
    return ingest_source("ai_mechanic", files, clap)


def main():
    print("Loading CLAP on MPS...", flush=True)
    clap = _clap()

    print("\n--- DB1 (Sound-Based-Vehicle-Diagnostics) ---")
    t0 = time.time()
    n1 = ingest_db1(clap)
    print(f"  elapsed: {time.time()-t0:.0f}s")

    print("\n--- Car-Engine-Sounds-Dataset ---")
    t0 = time.time()
    n2 = ingest_car_engine(clap)
    print(f"  elapsed: {time.time()-t0:.0f}s")

    print("\n--- Ai_Mechanic ---")
    t0 = time.time()
    n3 = ingest_ai_mechanic(clap)
    print(f"  elapsed: {time.time()-t0:.0f}s")

    total = n1 + n2 + n3
    print(f"\nTotal new clips: {total}")
    print(f"Corpus: {paths.DATA / 'external' / 'corpus.jsonl'}")

    report = {"db1": n1, "car_engine": n2, "ai_mechanic": n3, "total": total}
    (paths.DATA / "external" / "ingest_report.json").write_text(
        json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
