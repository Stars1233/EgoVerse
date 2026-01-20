#!/usr/bin/env python3
"""
run_aria_conversion_sql.py – SQL-backed driver (no CSV/S3)

• Walks /mnt/raw for bundles: {name}.vrs, {name}.json , mps_{name}_vrs/
• name == episode_hash (TEXT in DB) – row must already exist in app.episodes; otherwise skipped
• Uses Ray to run conversions with absolute symlinks in per-job tmp dirs
• On success, updates app.episodes.processed_path and num_frames
• Passes SQL task_description into the converter as `description`
• Determines arm automatically from SQL (left, right, or bimanual)
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple
from tqdm import tqdm

import ray
from ray.exceptions import OutOfMemoryError, RayTaskError, WorkerCrashedError

from egomimic.utils.aws.aws_data_utils import s3_sync_to_local, upload_dir_to_s3
import boto3

import traceback

import csv

# --- Conversion wrapper ------------------------------------------------------
from aria_helper import lerobot_job

# --- SQL helpers --------------------------------------------------------------
from egomimic.utils.aws.aws_sql import (
    TableRow,
    create_default_engine,
    episode_hash_to_table_row,
    update_episode,
)

# --- Paths -------------------------------------------------------------------
RAW_ROOT = Path("/mnt/raw")
PROCESSED_ROOT = Path("/home/ubuntu/processed")
PROCESSED_LOCAL_ROOT = Path(
    os.environ.get("PROCESSED_LOCAL_ROOT", "/home/ubuntu/processed")
).resolve()
PROCESSED_REMOTE_PREFIX = os.environ.get(
    "PROCESSED_REMOTE_PREFIX", "s3://rldb/processed_v2/test_aria"
).rstrip(
    "/"
)  # TODO switch back to aria instead of test_aria
BUCKET = os.environ.get("BUCKET", "rldb")
DATA_KEY = os.environ.get(
    "DATA_KEY", "raw_v2/test_aria"
)  # TODO switch back to aria instead of test_aria

LOG_ROOT = Path(
    os.environ.get(
        "ARIA_CONVERSION_LOG_ROOT",
        str(PROCESSED_LOCAL_ROOT / "aria_conversion_logs"),
    )
).resolve()


# --- Utilities ---------------------------------------------------------------
def ensure_path_ready(p: str | Path, retries: int = 30) -> bool:
    p = Path(p)
    for _ in range(retries):
        try:
            if p.exists():
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _map_processed_local_to_remote(p: str | Path) -> str:
    """Map any path under PROCESSED_LOCAL_ROOT → PROCESSED_REMOTE_PREFIX/relative."""
    if not p:
        return ""
    p = Path(p).resolve()
    try:
        rel = p.relative_to(PROCESSED_LOCAL_ROOT)  # raises if not under root
    except Exception:
        return str(p)
    return (
        f"{PROCESSED_REMOTE_PREFIX}/{rel.as_posix()}"
        if PROCESSED_REMOTE_PREFIX
        else str(p)
    )


def _parse_s3_uri(uri: str, *, default_bucket: str | None = None) -> tuple[str, str]:
    """
    Parse s3 URI or key prefix.
      - "s3://bucket/prefix" -> ("bucket", "prefix")
      - "prefix" -> (default_bucket, "prefix")
    """
    uri = (uri or "").strip()
    if uri.startswith("s3://"):
        rest = uri[len("s3://") :]
        bucket, _, key_prefix = rest.partition("/")
        return bucket, key_prefix.strip("/")
    if default_bucket is None:
        raise ValueError(f"Expected s3://... but got '{uri}' and no default_bucket provided")
    return default_bucket, uri.strip("/")


def iter_vrs_bundles(root: Path) -> Iterator[Tuple[Path, Path, Path]]:
    """
    Yield (vrs_file, json_file, mps_dir) for every valid bundle in `root`.

    Accept bundles containing:
      • {name}.vrs
      • {name}.json
      • mps_{name}_vrs/
    All must be in the SAME directory.
    """
    for vrs in sorted(root.glob("*.vrs")):
        name = vrs.stem
        jsonf = root / f"{name}.json"
        mpsdir = root / f"mps_{name}_vrs"

        # Require all four to exist
        if (
            mpsdir.is_dir()
            and (mpsdir / "hand_tracking").is_dir()
            and (mpsdir / "slam").is_dir()
        ):
            yield vrs, jsonf, mpsdir


def infer_arm_from_row(row: TableRow) -> str:
    """
    Infer arm from SQL row.robot_name (e.g., 'aria_left', 'aria_right', 'aria_bimanual').
    Falls back to 'bimanual'.
    """
    emb = (row.robot_name or "").lower()
    if "left" in emb:
        return "left"
    if "right" in emb:
        return "right"
    if "bimanual" in emb:
        return "bimanual"
    return "bimanual"


def _load_episode_hash(episode_hash: Path) -> str | None:
    return datetime.fromtimestamp(float(episode_hash) / 1000.0, timezone.utc).strftime(
        "%Y-%m-%d-%H-%M-%S-%f"
    )


def _is_oom_exception(e: Exception) -> bool:
    if isinstance(e, OutOfMemoryError):
        return True
    if isinstance(e, (RayTaskError, WorkerCrashedError)):
        s = str(e).lower()
        return (
            ("outofmemory" in s)
            or ("out of memory" in s)
            or ("oom" in s)
            or ("killed" in s)
        )
    s = str(e).lower()
    return ("outofmemory" in s) or ("out of memory" in s) or ("oom" in s)


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for s in self._streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self) -> None:
        for s in self._streams:
            s.flush()

    def isatty(self) -> bool:
        return False


# --- Ray task ----------------------------------------------------------------
def convert_one_bundle_impl(
    vrs: str,
    jsonf: str,
    mps_dir: str,
    out_dir: str,
    s3_processed_dir: str,
    dataset_name: str,
    arm: str,
    description: str,
) -> tuple[str, str, int]:
    """
    Perform symlink-based conversion for a single episode.
    Returns (ds_path, mp4_path, total_frames).
      • ds_path: dataset folder path
      • mp4_path: per-episode MP4 ('' if not created)
      • total_frames: -1 if unknown/failure
    """
    s3 = boto3.client("s3")
    stem = Path(vrs).stem
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = LOG_ROOT / f"{stem}-{uuid.uuid4().hex[:8]}.log"

    tmp_dir = Path.home() / "temp_mps_processing" / f"{stem}-{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("a", encoding="utf-8") as log_fh:
        tee_out = _Tee(sys.stdout, log_fh)
        tee_err = _Tee(sys.stderr, log_fh)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[LOG] {stem}: {log_path}", flush=True)
            targets = [
                Path(vrs).resolve(strict=True),
                Path(jsonf).resolve(strict=True),
                Path(mps_dir).resolve(strict=True),
            ]

            for t in targets:
                if not ensure_path_ready(t):
                    print(f"[ERR] missing {t}", flush=True)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return "", "", -1
                link = tmp_dir / t.name
                rel = t.relative_to(RAW_ROOT)
                t_key = f"{DATA_KEY.rstrip('/')}/{rel.as_posix()}"

                try:
                    if t.is_dir():
                        s3_sync_to_local(BUCKET, t_key, str(link))
                    else:
                        s3.download_file(BUCKET, t_key, str(link))
                except Exception as e:
                    print(f"[ERR] aws copy failed for {t}: {e}", flush=True)
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                    return "", "", -1
                print()
            # TODO remove this
            # List the contents of the temporary directory to show the download was successful
            print(f"[INFO] Listing {tmp_dir} contents after S3 download:", flush=True)
            for item in tmp_dir.iterdir():
                print(f"  {item}", flush=True)

            ds_parent = Path(out_dir)
            ds_parent.mkdir(parents=True, exist_ok=True)
            ds_path = ds_parent / dataset_name

            try:
                print(f"[INFO] Converting: {stem} → {ds_path} (arm={arm})", flush=True)
                lerobot_job(
                    raw_path=str(tmp_dir),
                    output_dir=str(ds_parent),
                    dataset_name=dataset_name,
                    arm=arm,
                    description=description or "",
                )

                frames = -1
                info = ds_path / "meta/info.json"
                if info.exists():
                    try:
                        meta = json.loads(info.read_text())
                        frames = int(meta.get("total_frames", -1))
                    except Exception:
                        frames = -1

                candidate = ds_parent / f"{stem}_video.mp4"
                if candidate.exists():
                    mp4_str = str(candidate)
                else:
                    matches = list(ds_path.glob(f"*{stem}*_video.mp4"))
                    mp4_str = str(matches[0]) if matches else ""
                
                try:
                    out_bucket, out_prefix = _parse_s3_uri(s3_processed_dir, default_bucket=BUCKET)
                    ds_rel = ds_path.resolve().relative_to(PROCESSED_LOCAL_ROOT).as_posix()
                    ds_s3_prefix = f"{out_prefix.rstrip('/')}/{ds_rel}".strip("/")
                    upload_dir_to_s3(str(ds_path), out_bucket, prefix=ds_s3_prefix)
                    if mp4_str:
                        mp4_path = Path(mp4_str)
                        if mp4_path.exists():
                            mp4_rel = mp4_path.resolve().relative_to(PROCESSED_LOCAL_ROOT).as_posix()
                            mp4_s3_key = f"{out_prefix.rstrip('/')}/{mp4_rel}".strip("/")
                            try:
                                s3.upload_file(str(mp4_path), out_bucket, mp4_s3_key)
                            except Exception as e:
                                print(f"[ERR] Failed to upload mp4 {mp4_path} to S3: {e}", flush=True)
                except Exception as e:
                    print(f"[ERR] Failed to upload {ds_path} to S3: {e}", flush=True)
                #TODO delete local directory
                    
                return str(ds_path), mp4_str, frames

            except Exception as e:
                err_msg = f"[FAIL] {stem}: {e}\n{traceback.format_exc()}"
                print(err_msg, flush=True)
                return str(ds_path), "", -1
            finally:
                # shutil.rmtree(tmp_dir, ignore_errors=True)
                #TODO delete local directory
                pass


@ray.remote(num_cpus=8, resources={"aria_small": 1})
def convert_one_bundle_small(*args, **kwargs):
    return convert_one_bundle_impl(*args, **kwargs)


@ray.remote(num_cpus=32, resources={"aria_big": 1})
def convert_one_bundle_big(*args, **kwargs):
    return convert_one_bundle_impl(*args, **kwargs)


# --- Driver ------------------------------------------------------------------
def launch(
    dry: bool = False,
    skip_if_done: bool = False,
    episode_hashes: list[str] | None = None,
):
    engine = create_default_engine()
    pending: Dict[ray.ObjectRef, Dict[str, Any]] = {}

    benchmark_rows = []

    for vrs, jsonf, mps in iter_vrs_bundles(RAW_ROOT):
        name = vrs.stem

        # IMPORTANT: episode_hash is TEXT in DB; do not cast to int
        episode_key = _load_episode_hash(name)

        if not episode_key:
            print(f"[SKIP] {name}: could not parse episode_hash from stem", flush=True)
            continue

        if episode_hashes is not None and episode_key not in episode_hashes:
            print(
                f"[SKIP] {name}: episode_key '{episode_key}' not in provided episode_hashes list",
                flush=True,
            )
            continue

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(f"[SKIP] {name}: no matching row in SQL (app.episodes)", flush=True)
            continue

        processed_path = (row.processed_path or "").strip()
        if skip_if_done and len(processed_path) > 0:
            print(
                f"[SKIP] {name}: already has processed_path='{processed_path}'",
                flush=True,
            )
            continue

        if row.processing_error != "":
            print(
                f"[INFO] skipping episode hash: {row.episode_hash} due to processing error",
                flush=True,
            )
            continue

        if row.is_deleted:
            print(f"[SKIP] {name}: episode marked as deleted in SQL", flush=True)
            continue

        print(f"[INFO] processing {name}: episode_key={episode_key}", flush=True)

        arm = infer_arm_from_row(row)
        dataset_name = episode_key
        out_dir = PROCESSED_ROOT
        s3out_dir = PROCESSED_REMOTE_PREFIX
        description = row.task_description or ""

        if dry:
            ds_path = (PROCESSED_ROOT / dataset_name).resolve()
            stem = vrs.stem
            mp4_candidate = PROCESSED_ROOT / f"{stem}_video.mp4"

            mapped_ds = _map_processed_local_to_remote(ds_path)
            mapped_mp4 = _map_processed_local_to_remote(mp4_candidate)

            print(
                f"[DRY] {name}: arm={arm} | out_dir={out_dir}/{dataset_name}\n"
                f"      desc='{description[:60]}'\n"
                f"      would write to SQL:\n"
                f"        processed_path={mapped_ds}\n"
                f"        mp4_path={mapped_mp4}",
                flush=True,
            )
            continue

        args = (
            str(vrs),
            str(jsonf),
            str(mps),
            str(out_dir),
            str(s3out_dir),
            dataset_name,
            arm,
            description,
        )

        start_time = time.time()
        ref = convert_one_bundle_small.remote(*args)
        pending[ref] = {
            "episode_key": episode_key,
            "dataset_name": dataset_name,
            "start_time": start_time,
            "size": "small",
            "args": args,
        }

    if dry or not pending:
        return

    # Collect and update SQL (with OOM retry on BIG)
    while pending:
        done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
        ref = done_refs[0]
        info = pending.pop(ref)

        episode_key = info["episode_key"]
        start_time = info["start_time"]
        duration_sec = time.time() - start_time

        row = episode_hash_to_table_row(engine, episode_key)
        if row is None:
            print(
                f"[WARN] Episode {episode_key}: row disappeared before update?",
                flush=True,
            )
            continue

        try:
            ds_path, mp4_path, frames = ray.get(
                ref
            )  # can throw (OOM, index error, etc.)

            row.num_frames = int(frames) if frames is not None else -1
            if row.num_frames > 0:
                row.processed_path = _map_processed_local_to_remote(ds_path)
                row.mp4_path = _map_processed_local_to_remote(mp4_path)
                row.processing_error = ""
            else:
                row.processed_path = ""
                row.mp4_path = ""
                row.processing_error = "Zero Frames"

            update_episode(engine, row)
            print(
                f"[OK] Updated SQL for {episode_key}: "
                f"processed_path={row.processed_path}, num_frames={row.num_frames}, "
                f"duration_sec={duration_sec:.2f}",
                flush=True,
            )

            if row.num_frames > 0 and row.processed_path:
                benchmark_rows.append(
                    {
                        "episode_key": episode_key,
                        "processed_path": row.processed_path,
                        "mp4_path": row.mp4_path,
                        "num_frames": row.num_frames,
                        "duration_sec": duration_sec,
                    }
                )

        except Exception as e:
            # If OOM on small, retry once on big
            if _is_oom_exception(e) and info.get("size") == "small":
                print(
                    f"[OOM] Episode {episode_key} failed on SMALL. Retrying on BIG...",
                    flush=True,
                )
                args = info["args"]
                ref2 = convert_one_bundle_big.remote(*args)
                pending[ref2] = {
                    **info,
                    "start_time": time.time(),
                    "size": "big",
                }
                continue

            print(
                f"[FAIL] Episode {episode_key} task failed ({info.get('size','?')}): "
                f"{type(e).__name__}: {e}",
                flush=True,
            )

            # mark failed in SQL (so skip-if-done won't think it's done)
            row.num_frames = -1
            row.processed_path = ""
            row.mp4_path = ""
            # row.processing_error = f"{type(e).__name__}: {e}" #TODO undo this
            try:
                update_episode(engine, row)
                print(
                    f"[FAIL] Marked SQL failed for {episode_key} (cleared processed_path)",
                    flush=True,
                )
            except Exception as ee:
                print(
                    f"[ERR] SQL update failed for failed episode {episode_key}: {ee}",
                    flush=True,
                )

    if benchmark_rows:
        timing_file = Path("./aria_conversion_timings.csv")
        file_exists = timing_file.exists()
        fieldnames = [
            "episode_key",
            "processed_path",
            "mp4_path",
            "num_frames",
            "duration_sec",
        ]
        try:
            with timing_file.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for bench_row in benchmark_rows:
                    writer.writerow(bench_row)
            print(
                f"[BENCH] wrote {len(benchmark_rows)} entries → {timing_file.resolve()}",
                flush=True,
            )
        except Exception as e:
            print(f"[ERR] Failed to write benchmark CSV {timing_file}: {e}", flush=True)


# --- CLI ---------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--skip-if-done",
        action="store_true",
        help="Skip episodes that already have a processed_path in SQL",
    )
    p.add_argument(
        "--ray-address", default="auto", help="Ray cluster address (default: auto)"
    )
    p.add_argument(
        "--episode-hash",
        action="append",
        dest="episode_hashes",
        help="Episode hash to process. Can be specified multiple times to process multiple episodes.",
    )
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    if args.debug:
        runtime_env = {"working_dir": "/home/ubuntu/EgoVerse", "excludes": [
      "**/.git/**",
      "external/openpi/third_party/aloha/**",
      "**/*.stl",
      "**/*.pack",
      "**/__pycache__/**",
      "egomimic/robot/**",
      "external/openpi/**"
    ],}
    else:
        runtime_env = None

    ray.init(address=args.ray_address, runtime_env=runtime_env)
    launch(
        dry=args.dry_run,
        skip_if_done=args.skip_if_done,
        episode_hashes=args.episode_hashes,
    )


if __name__ == "__main__":
    main()
