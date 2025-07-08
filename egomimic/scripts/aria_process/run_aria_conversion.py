#!/usr/bin/env python3
"""
run_aria_conversion.py
─────────────────────────────────────────────────────────────────────────
 • Scans every task folder listed in /mnt/raw/task_map.csv
 • Spawns one Ray task *per VRS bundle* that hasn’t been converted yet
 • Each task builds its **own** temp workspace of symlinks, then runs
       lerobot_job(...)                 (see aria_helper.py)
 • Results are written to /mnt/processed/<task>/<stem>_processed/…
 • A single global progress file is kept:
       /mnt/processed/vrs_conversion_status.csv
   We update it on the local disk first, then copy the plain *file data*
   back to S3 to avoid utime/permission issues on the fuse mount.
"""

import argparse, csv, json, os, shutil, uuid
from pathlib import Path
from filelock import FileLock
import ray

# ───────────── fixed paths ─────────────────────────────────────────
RAW_ROOT        = Path("/mnt/raw")
PROCESSED_ROOT  = Path("/mnt/processed")
TASK_MAP_CSV    = RAW_ROOT / "task_map.csv"                # cols: task,arm
GLOBAL_STATUS   = PROCESSED_ROOT / "vrs_conversion_status.csv"

LOCAL_STATUS    = Path("/tmp/vrs_conversion_status.csv")   # local shadow
LOCK_PATH       = Path("/tmp/vrs_status.lock")
lock            = FileLock(str(LOCK_PATH))

# helper that actually calls aria_to_lerobot.py
from aria_helper import lerobot_job                        # noqa: E402

# ═════════════════ helpers ════════════════════════════════════════
def load_task_map() -> dict[str, str]:
    with TASK_MAP_CSV.open() as f:
        return {r["task"].strip(): r["arm"].strip().lower()
                for r in csv.DictReader(f)}

def already_done() -> set[str]:
    if not GLOBAL_STATUS.exists():
        return set()
    with GLOBAL_STATUS.open() as f:
        return {f"{r['task']}/{r['vrs']}" for r in csv.DictReader(f)}

def vrs_bundles(task_dir: Path):
    """Yield (vrs_file, json_file, mps_dir) triples that pass integrity checks."""
    for vrs in task_dir.glob("*.vrs"):
        stem  = vrs.stem
        jsonf = task_dir / f"{stem}.vrs.json"
        mps   = task_dir / f"mps_{stem}_vrs"
        if not (jsonf.exists() and mps.is_dir() and
                (mps/"hand_tracking/wrist_and_palm_poses.csv").exists() and
                (mps/"slam/closed_loop_trajectory.csv").exists()):
            continue
        yield vrs, jsonf, mps

def append_status(row: dict):
    """Append a row to LOCAL_STATUS, then copy file (data-only) to S3."""
    with lock:
        new_file = not LOCAL_STATUS.exists()
        with LOCAL_STATUS.open("a", newline="") as f:
            w = csv.DictWriter(
                f, fieldnames=["task", "vrs", "total_frames", "output_path"])
            if new_file:
                w.writeheader()
            w.writerow(row)

        tmp_target = GLOBAL_STATUS.with_suffix(".tmp")
        with LOCAL_STATUS.open("rb") as src, tmp_target.open("wb") as dst:
            shutil.copyfileobj(src, dst)

        tmp_target.replace(GLOBAL_STATUS)

# ════════════════ Ray remote task ═════════════════════════════════
@ray.remote(num_cpus=8, memory=16 * 1024**3)
def convert_one(vrs: str, jsonf: str, mps_dir: str,
                out_dir: str, arm: str) -> tuple[str, int]:
    """
    Worker-side conversion.  Creates its own temp dir on *this* VM.
    Returns (dataset_path, total_frames).  On failure → total_frames = -1.
    """
    stem    = Path(vrs).stem
    tmp_dir = Path.home() / "temp_mps_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # symlinks ─ fast and zero-copy
    for src in (vrs, jsonf, mps_dir):
        os.symlink(src, tmp_dir / Path(src).name,
                   target_is_directory=Path(src).is_dir())

    ds_path = Path(out_dir) / f"{stem}_processed"
    try:
        print(f"[INFO] Converting → {ds_path}", flush=True)

        lerobot_job(raw_path=str(tmp_dir),
                    output_dir=str(out_dir),
                    dataset_name=f"{stem}_processed",
                    arm=arm,
                    description="")

        frames = -1
        info_p = ds_path / "meta/info.json"
        if info_p.exists():
            frames = int(json.loads(info_p.read_text()).get("total_frames", -1))

        print(f"[INFO] Done   → {ds_path} ({frames} frames)", flush=True)
        return str(ds_path), frames

    except Exception as exc:
        print(f"[ERROR] {ds_path} failed: {exc}", flush=True)
        return str(ds_path), -1

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

# ═══════════════ main driver ══════════════════════════════════════
def launch(dry_run: bool = False):
    done = already_done()
    pending: dict[ray.ObjectRef, tuple[str, str, str]] = {}   # ref → (task, vrs_stem, ds_path)

    for task, arm in load_task_map().items():
        for vrs, jsonf, mps in vrs_bundles(RAW_ROOT / task):
            key = f"{task}/{Path(vrs).stem}"
            if key in done:
                continue

            out_dir = PROCESSED_ROOT / task
            ds_path = out_dir / f"{Path(vrs).stem}_processed"

            if dry_run:
                print(f"[DRY-RUN] {task} → {ds_path}  (arm={arm})")
            else:
                ref = convert_one.remote(str(vrs), str(jsonf), str(mps),
                                         str(out_dir), arm)
                pending[ref] = (task, Path(vrs).stem, str(ds_path))

    if dry_run or not pending:
        print("Dry-run complete." if dry_run else "Nothing to do.")
        return

    print(f"Submitted {len(pending)} jobs to Ray…")

    while pending:
        finished, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = finished[0]
        ds_path, frames = ray.get(ref)
        task, vrs_stem, _ = pending.pop(ref)

        append_status({
            "task": task,
            "vrs":  vrs_stem,
            "total_frames": frames,
            "output_path": ds_path
        })
        print(f"[LOG] {task}/{vrs_stem} → {frames} frames")

# ═══════════════ CLI entry ════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="List conversions without running them")
    args = parser.parse_args()

    ray.init(address="auto")
    launch(dry_run=args.dry_run)
