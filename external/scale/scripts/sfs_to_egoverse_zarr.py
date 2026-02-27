#!/usr/bin/env python3
"""
Scale SFS -> EgoVerse Zarr converter.

Output keys per episode:
    left.obs_ee_pose                 (T, 7)         xyz + quat(w, x, y, z)
    right.obs_ee_pose                (T, 7)         xyz + quat(w, x, y, z)
    left.obs_keypoints               (T, 63)        21 keypoints * 3 (xyz)
    right.obs_keypoints              (T, 63)        21 keypoints * 3 (xyz)
    left.obs_wrist_pose              (T, 7)         xyz + quat(w, x, y, z)
    right.obs_wrist_pose             (T, 7)         xyz + quat(w, x, y, z)
    obs_head_pose                    (T, 7)         xyz + quat(w, x, y, z)
    images.front_1                   (T, H, W, 3)   JPEG-compressed by ZarrWriter

Usage:
  python sfs_to_egoverse_zarr.py --task-ids TASK1 TASK2 --output-dir ./zarr_out
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import simplejpeg
from decord import VideoReader, cpu as decord_cpu
from scipy.spatial.transform import Rotation as R

from egomimic.rldb.zarr.zarr_writer import ZarrWriter

from scale_api import (
    download_from_simple_response_dict,
    get_simple_response_dict_egocentric,
)
from sfs_data import (
    INVALID_VALUE,
    FrameData,
    SFSDataExtractor,
    batch_pose6_to_pose7,
    compute_palm_6dof,
    compute_wrist_6dof,
)

ACTION_WINDOW = 30
SUB_EPISODE_LENGTH = 300
IMAGE_SIZE = (640, 480)  # (W, H) for cv2.resize


# ---------------------------------------------------------------------------
# Video / image helpers
# ---------------------------------------------------------------------------


def _get_video_frame_count(video_path: str) -> int:
    """Get frame count without decoding the video."""
    vr = VideoReader(video_path, ctx=decord_cpu())
    return len(vr)


def _decode_selected_frames(
    video_path: str,
    indices: list[int],
    chunk_size: int = 500,
    resize: tuple[int, int] | None = IMAGE_SIZE,
) -> dict[int, np.ndarray]:
    """Batch-decode only the requested frame indices via decord.

    Decodes in chunks and eagerly resizes to *resize* (W, H) to keep
    memory usage bounded.  Returns a dict mapping frame index to RGB uint8.
    """
    if not indices:
        return {}
    indices_sorted = sorted(set(indices))
    vr = VideoReader(video_path, ctx=decord_cpu())
    max_idx = len(vr) - 1
    valid = [i for i in indices_sorted if i <= max_idx]
    if not valid:
        return {}
    result: dict[int, np.ndarray] = {}
    for start in range(0, len(valid), chunk_size):
        chunk_indices = valid[start : start + chunk_size]
        batch = vr.get_batch(chunk_indices).asnumpy()
        for i, t in enumerate(chunk_indices):
            frame = batch[i]
            if resize is not None:
                frame = cv2.resize(frame, resize, interpolation=cv2.INTER_LINEAR)
            result[t] = frame
        del batch
    return result


def _resize_and_encode(frame: np.ndarray) -> tuple[tuple[int, ...], bytes]:
    """Resize frame to IMAGE_SIZE and JPEG-encode it. GIL-releasing."""
    resized = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    jpeg = simplejpeg.encode_jpeg(
        resized, quality=ZarrWriter.JPEG_QUALITY, colorspace="RGB"
    )
    return resized.shape, jpeg


# ---------------------------------------------------------------------------
# Language annotations
# ---------------------------------------------------------------------------


def _build_language_annotations(sub_frames: list[FrameData]) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    current_text: str | None = None
    start_idx: int | None = None
    for idx, frame in enumerate(sub_frames):
        label = frame.subgoal.get("text", "").strip() if frame.subgoal else ""
        text = label if label else None
        if text != current_text:
            if current_text is not None and start_idx is not None:
                rows.append((current_text, start_idx, idx - 1))
            current_text = text
            start_idx = idx if text is not None else None
    if current_text is not None and start_idx is not None:
        rows.append((current_text, start_idx, len(sub_frames) - 1))
    return rows


def _task_description(frames: list[FrameData], demo_meta: dict[str, Any]) -> str:
    candidate = str(demo_meta.get("Demonstration", "")).strip()
    if candidate:
        return candidate
    skip = {"Inactive Time", "Collector Issue", "inactive time", "collector issue"}
    for frame in frames:
        for ann in frame.text_annotations:
            text = str(ann.get("text", "")).strip()
            label = str(ann.get("label", "")).strip()
            if text and text not in skip and label not in skip:
                return text
    return "Unknown task"


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------


def convert_task_to_zarr(
    task_id: str,
    output_dir: str,
    download_dir: str,
    robot_type: str = "scale_bimanual",
    fps: int = 30,
    img_workers: int | None = None,
) -> dict[str, Any]:
    """Convert one Scale task to one or more Zarr episodes.

    Returns a dict with keys: episodes, folder, task_desc, total_frames, output_dir.
    """
    t_start = time.perf_counter()
    if img_workers is None:
        img_workers = min(os.cpu_count() or 4, 8)

    print(f"[{task_id}] Fetching task metadata...")
    task_download_path = os.path.join(download_dir, task_id)
    os.makedirs(task_download_path, exist_ok=True)

    response = get_simple_response_dict_egocentric(task_id)
    if response is None:
        raise ValueError(f"Task {task_id} not found or Scale API failed")

    print(f"[{task_id}] Downloading files...")
    local_paths = download_from_simple_response_dict(task_download_path, response)
    sfs_path = local_paths.get("sfs")
    annotations_path = local_paths.get("annotations")
    video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
    if not all([sfs_path, annotations_path, video_path]):
        raise ValueError(f"Missing SFS/annotation/video files for task {task_id}")

    def _nonempty(p: str | None) -> bool:
        return bool(p) and os.path.exists(p) and os.path.getsize(p) > 0

    if not (_nonempty(sfs_path) and _nonempty(annotations_path)):
        raise ValueError(f"Downloaded SFS/annotation files are empty for task {task_id}")

    print(f"[{task_id}] Loading SFS metadata...")
    try:
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    except ValueError:
        print(f"[{task_id}] Load failed — re-downloading SFS + annotations...")
        for p in (sfs_path, annotations_path):
            if p and os.path.exists(p):
                os.remove(p)
        local_paths = download_from_simple_response_dict(task_download_path, response)
        sfs_path = local_paths.get("sfs")
        annotations_path = local_paths.get("annotations")
        video_path = local_paths.get("left_rectified") or local_paths.get("left_rgb")
        extractor = SFSDataExtractor(sfs_path, annotations_path, video_path)
    frames = extractor.extract_all_frames_metadata()
    n_frames = len(frames)
    if n_frames <= ACTION_WINDOW:
        raise ValueError(f"Task {task_id} has too few frames ({n_frames})")

    task_desc = _task_description(frames, extractor.demonstration_metadata)
    valid_frame_count = n_frames - ACTION_WINDOW

    # ------------------------------------------------------------------
    # Precompute all per-frame data into dense arrays (no video needed)
    # ------------------------------------------------------------------
    left_world_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_world_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_wrist_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    right_wrist_6 = np.full((n_frames, 6), INVALID_VALUE, dtype=np.float32)
    left_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    right_kps = np.full((n_frames, 63), INVALID_VALUE, dtype=np.float32)
    head_pose_6 = np.zeros((n_frames, 6), dtype=np.float32)

    for i, frame in enumerate(frames):
        if frame.hand_keypoints.left is not None:
            left_world_6[i] = compute_palm_6dof(frame.hand_keypoints.left)
            left_wrist_6[i] = compute_wrist_6dof(frame.hand_keypoints.left)
            left_kps[i] = frame.hand_keypoints.left.flatten().astype(np.float32)
        if frame.hand_keypoints.right is not None:
            right_world_6[i] = compute_palm_6dof(frame.hand_keypoints.right, flip_x=True)
            right_wrist_6[i] = compute_wrist_6dof(frame.hand_keypoints.right, flip_x=True)
            right_kps[i] = frame.hand_keypoints.right.flatten().astype(np.float32)
        head_pose_6[i, :3] = frame.camera_pose.position.astype(np.float32)
        head_pose_6[i, 3:] = R.from_matrix(frame.camera_pose.rotation_matrix).as_euler(
            "ZYX", degrees=False
        ).astype(np.float32)

    left_world = batch_pose6_to_pose7(left_world_6)
    right_world = batch_pose6_to_pose7(right_world_6)
    left_wrist = batch_pose6_to_pose7(left_wrist_6)
    right_wrist = batch_pose6_to_pose7(right_wrist_6)
    head_pose_world = batch_pose6_to_pose7(head_pose_6)

    # ------------------------------------------------------------------
    # Filter valid frame indices
    # ------------------------------------------------------------------
    valid_indices: list[int] = []
    for t in range(valid_frame_count):
        if (
            frames[t].collector_issue is not None
            and frames[t].collector_issue.get("issue_type") == "Inactive Time"
        ):
            continue
        window = slice(t, t + ACTION_WINDOW)
        n_invalid = (
            np.sum(np.any(left_world[window] >= INVALID_VALUE - 1, axis=1))
            + np.sum(np.any(right_world[window] >= INVALID_VALUE - 1, axis=1))
        )
        if n_invalid > ACTION_WINDOW:
            continue
        valid_indices.append(t)

    if not valid_indices:
        raise ValueError(f"Task {task_id} has no valid frames after filtering")

    print(f"[{task_id}] {len(valid_indices)} valid frames out of {valid_frame_count}")

    # ------------------------------------------------------------------
    # Probe video frame count
    # ------------------------------------------------------------------
    video_frame_count = _get_video_frame_count(video_path)
    print(f"[{task_id}] Video: {video_frame_count} frames  SFS: {n_frames} frames")
    if video_frame_count != n_frames:
        print(f"[{task_id}] WARNING: video/SFS frame count mismatch")

    # ------------------------------------------------------------------
    # Plan sub-episodes (index lists only, no video decode yet)
    # ------------------------------------------------------------------
    sub_episode_plans: list[list[int]] = []
    for ep_start in range(0, len(valid_indices), SUB_EPISODE_LENGTH):
        sub = valid_indices[ep_start : ep_start + SUB_EPISODE_LENGTH]
        if len(sub) < 10:
            continue
        preliminary_kept = [t for t in sub if t < video_frame_count]
        if len(preliminary_kept) < 10:
            continue
        sub_episode_plans.append(sub)

    # ------------------------------------------------------------------
    # Write sub-episodes (streaming: decode per sub-episode to bound memory)
    # ------------------------------------------------------------------
    folder = datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S-%f")
    task_output_dir = Path(output_dir) / folder
    task_output_dir.mkdir(parents=True, exist_ok=True)

    written = 0

    for sub in sub_episode_plans:
        sub_indices = sorted(t for t in sub if t < video_frame_count)
        decoded = _decode_selected_frames(video_path, sub_indices)
        kept = [t for t in sub if t in decoded]
        none_count = len(sub) - len(kept)
        print(f"[ep{written}] sub={len(sub)}  kept={len(kept)}  dropped(no image)={none_count}  frames=[{sub[0]}..{sub[-1]}]")
        if len(kept) < 10:
            del decoded
            continue

        T = len(kept)
        kept_arr = np.array(kept)
        left_curr_7 = np.where(
            left_world[kept_arr] >= INVALID_VALUE - 1, 0.0, left_world[kept_arr]
        ).astype(np.float32)
        right_curr_7 = np.where(
            right_world[kept_arr] >= INVALID_VALUE - 1, 0.0, right_world[kept_arr]
        ).astype(np.float32)
        left_wrist_curr_7 = np.where(
            left_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, left_wrist[kept_arr]
        ).astype(np.float32)
        right_wrist_curr_7 = np.where(
            right_wrist[kept_arr] >= INVALID_VALUE - 1, 0.0, right_wrist[kept_arr]
        ).astype(np.float32)

        actions_head = head_pose_world[kept_arr]
        left_keypoints = np.where(
            left_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, left_kps[kept_arr]
        ).astype(np.float32)
        right_keypoints = np.where(
            right_kps[kept_arr] >= INVALID_VALUE - 1, 0.0, right_kps[kept_arr]
        ).astype(np.float32)

        ordered_frames = [decoded[t] for t in kept]
        del decoded
        n_workers = min(img_workers, T)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            encode_results = list(pool.map(_resize_and_encode, ordered_frames))
        del ordered_frames
        image_shape = list(encode_results[0][0])
        pre_encoded = np.array([r[1] for r in encode_results], dtype=object)
        del encode_results

        print(f"[ep{written}] T={T}  image_shape={image_shape}  kept={len(kept_arr)}")

        numeric_data = {
            "left.obs_ee_pose": left_curr_7,
            "right.obs_ee_pose": right_curr_7,
            "left.obs_keypoints": left_keypoints,
            "right.obs_keypoints": right_keypoints,
            "left.obs_wrist_pose": left_wrist_curr_7,
            "right.obs_wrist_pose": right_wrist_curr_7,
            "obs_head_pose": actions_head,
        }

        used_frames = [frames[t] for t in kept]
        lang_ann = _build_language_annotations(used_frames)

        episode_path = task_output_dir / f"{task_id}_episode_{written:06d}.zarr"
        ZarrWriter.create_and_write(
            episode_path=episode_path,
            numeric_data=numeric_data,
            pre_encoded_image_data={
                "images.front_1": (pre_encoded, image_shape),
            },
            embodiment=robot_type,
            fps=fps,
            task=task_desc,
            annotations=lang_ann if lang_ann else None,
            enable_sharding=True,
        )
        written += 1
        print(f"[{task_id}] Wrote episode {written} ({T} frames) -> {episode_path.name}")

    if os.path.exists(task_download_path):
        shutil.rmtree(task_download_path, ignore_errors=True)

    elapsed = time.perf_counter() - t_start
    print(f"[{task_id}] Done: {written} episode(s) in {elapsed:.1f}s -> {task_output_dir}")
    return {
        "episodes": written,
        "folder": folder,
        "task_desc": task_desc,
        "total_frames": n_frames,
        "output_dir": str(task_output_dir),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert Scale SFS tasks to EgoVerse Zarr episodes"
    )
    parser.add_argument("--task-ids", nargs="+", required=True, help="Scale task IDs")
    parser.add_argument("--output-dir", default="egoverse_zarr_dataset", help="Output root")
    parser.add_argument("--download-dir", default="scale_data", help="Temp download cache")
    parser.add_argument("--robot-type", default="scale_bimanual", help="Embodiment tag")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel task workers (default: 1 = sequential)",
    )
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.download_dir).mkdir(parents=True, exist_ok=True)

    img_workers = max(1, (os.cpu_count() or 4) // max(args.workers, 1))

    total_episodes = 0
    failed: list[str] = []

    if args.workers > 1:
        print(f"Running with {args.workers} parallel workers "
              f"({img_workers} image threads per worker)")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    convert_task_to_zarr,
                    task_id=tid,
                    output_dir=args.output_dir,
                    download_dir=args.download_dir,
                    robot_type=args.robot_type,
                    fps=args.fps,
                    img_workers=img_workers,
                ): tid
                for tid in args.task_ids
            }
            for future in as_completed(futures):
                tid = futures[future]
                try:
                    total_episodes += future.result()["episodes"]
                except Exception as exc:
                    print(f"[{tid}] ERROR: {exc}")
                    traceback.print_exc()
                    failed.append(tid)
    else:
        for idx, task_id in enumerate(args.task_ids, start=1):
            print(f"\n[{idx}/{len(args.task_ids)}] {task_id}")
            try:
                result = convert_task_to_zarr(
                    task_id=task_id,
                    output_dir=args.output_dir,
                    download_dir=args.download_dir,
                    robot_type=args.robot_type,
                    fps=args.fps,
                    img_workers=img_workers,
                )
                total_episodes += result["episodes"]
            except Exception as exc:
                print(f"[{task_id}] ERROR: {exc}")
                traceback.print_exc()
                failed.append(task_id)

    print(f"\n{'=' * 60}")
    print(f"Conversion complete: {len(args.task_ids)} tasks, "
          f"{len(args.task_ids) - len(failed)} ok, {len(failed)} failed")
    print(f"Episodes written: {total_episodes}")
    if failed:
        print(f"Failed: {failed}")
    print(f"Output: {Path(args.output_dir).resolve()}")
    print("=" * 60)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
