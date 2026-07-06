"""Ingest Kaggle datasets: Revix Engine Knock, Car Diagnostics, Vehicle Sounds.

Each folder is ingested through the same cascade as scrape data:
  cardiag ingest <dir> --kind fault|normal [--cause <c>] [--source <s>]
"""
import subprocess
import sys
from pathlib import Path

KAGGLE = Path("data/kaggle")


def ingest(dir_path, kind, cause=None, source="kaggle"):
    cmd = ["uv", "run", "cardiag", "ingest", str(dir_path),
           "--kind", kind, "--source", source]
    if cause:
        cmd += ["--cause", cause]
    print(f"  {' '.join(cmd)}")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"    ERROR: {r.stderr.strip()}")
    else:
        print(f"    {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else 'ok'}")


def main():
    # ---- Revix Engine Knock (Apache 2.0) ----
    revix = KAGGLE / "engine_sounds_hf_dataset" / "audio_files"
    if revix.exists():
        print("\n=== Revix Engine Knock (Apache 2.0) ===")
        ingest(revix / "car_clean", "normal", source="revix")
        ingest(revix / "car_knocking", "fault", cause="engine_internal", source="revix")
        ingest(revix / "truck_clean", "normal", source="revix")
        ingest(revix / "truck_knocking", "fault", cause="engine_internal", source="revix")
    else:
        print("  Revix not found, skipping")

    # ---- Car Diagnostics Dataset (unknown license, research-only) ----
    cdd = KAGGLE / "car diagnostics dataset"
    if cdd.exists():
        print("\n=== Car Diagnostics Dataset (research-only) ===")
        # Braking state
        ingest(cdd / "braking state" / "normal_brakes", "normal", source="car_diagnostics")
        ingest(cdd / "braking state" / "worn_out_brakes", "fault", cause="brakes", source="car_diagnostics")
        # Idle state
        ingest(cdd / "idle state" / "normal_engine_idle", "normal", source="car_diagnostics")
        ingest(cdd / "idle state" / "serpentine_belt", "fault", cause="belt", source="car_diagnostics")
        ingest(cdd / "idle state" / "power_steering", "fault", cause="power_steering", source="car_diagnostics")
        ingest(cdd / "idle state" / "low_oil", "fault", cause="engine_internal", source="car_diagnostics")
        ingest(cdd / "idle state" / "combined", "fault", cause="engine_internal", source="car_diagnostics")
        # Startup state
        ingest(cdd / "startup state" / "normal_engine_startup", "normal", source="car_diagnostics")
        ingest(cdd / "startup state" / "dead_battery", "fault", cause="electrical", source="car_diagnostics")
        ingest(cdd / "startup state" / "bad_ignition", "fault", cause="engine_internal", source="car_diagnostics")
    else:
        print("  Car Diagnostics not found, skipping")

    # ---- Vehicle Sounds Dataset (unknown license, research-only) ----
    # Skipped: 9448 files of generic traffic sounds, low diagnostic value
    # and would take ~1h through the cascade for minimal gain.


if __name__ == "__main__":
    main()
