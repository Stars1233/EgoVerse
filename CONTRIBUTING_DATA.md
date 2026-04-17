# EgoVerse Data Contribution Guide

*For new labs and companies contributing egocentric human demonstration data to the EgoVerse consortium.*

---

## Table of Contents

1. [Overview](#1-overview)
2. [Prerequisites](#2-prerequisites)
3. [Episode Hash Convention](#3-episode-hash-convention)
4. [Database Registry](#4-database-registry)
5. [Zarr v3 Episode Format](#5-zarr-v3-episode-format)
6. [Coordinate Frame Conventions](#6-coordinate-frame-conventions)
7. [Pose Representation](#7-pose-representation)
8. [Image Format](#8-image-format)
9. [Language Annotations](#9-language-annotations)
10. [Embodiment Identifiers](#10-embodiment-identifiers)
11. [Writing an Episode (Code)](#11-writing-an-episode-code)
12. [Registering Episodes in the Database](#12-registering-episodes-in-the-database)
13. [Uploading to S3](#13-uploading-to-s3)
14. [Validation and Verification](#14-validation-and-verification)
15. [Task Taxonomy](#15-task-taxonomy)
16. [Pre-Submission Checklist](#16-pre-submission-checklist)
17. [Getting Access and Contact](#17-getting-access-and-contact)

---

## 1. Overview

EgoVerse is a multi-lab egocentric human demonstration dataset for robot co-training. The primary storage and training format is **EgoVerse's own Zarr v3 schema** — a custom, S3-native per-episode format described in full in this guide. It is used as the official H2R (Human-to-Robot) metadata standard.

Every contributed episode must satisfy three contracts:

| Contract | What it enforces |
|---|---|
| **File format** | Zarr v3 store with specific key names, dtypes, and shapes |
| **Coordinate frame** | All poses expressed in a consistent reference frame (SLAM world frame at write time; head frame at train time) |
| **Database record** | One row per episode registered in the PostgreSQL episode registry before upload |

The pipeline at a glance:

```
Your raw data
    └─► Convert to Zarr v3 (this guide)
    └─► Register row in app.episodes DB
    └─► Upload to s3://rldb/processed_v3/<embodiment>/<episode_hash>.zarr/
    └─► Available for download via sync_s3.py
```

---

## 2. Prerequisites

### 2.1 Hardware

EgoVerse is hardware-agnostic. Any egocentric camera with a SLAM system that provides 6-DOF pose tracking is supported. The minimum requirements are:

| Item | Requirement |
|---|---|
| Egocentric camera | Any camera worn or mounted on the head/torso providing a first-person RGB stream at ≥ 30 fps. Examples: Project Aria glasses, OAK-D, ZED Mini, RealSense T265, GoPro + external SLAM. |
| SLAM / pose tracking | A system that outputs 6-DOF device pose in a consistent metric world frame at ≥ 30 fps, synchronized with the RGB stream. Examples: Aria MPS, ZED SDK positional tracking, ORB-SLAM3, OpenVINS, RealSense tracking firmware. |
| Hand tracking | Per-frame 3D hand landmark estimates (21 keypoints per hand) synchronized to the RGB stream, expressed in the same SLAM world frame. Examples: Aria MPS hand tracking, MediaPipe + depth unprojection, OAK-D depthai hand tracker, Ultraleap. If your setup does not produce hand keypoints, omit `*.obs_keypoints` and `*.obs_wrist_pose` and use only `*.obs_ee_pose` (e.g. derived from a robot's FK or a wrist-worn IMU). |
| Wrist cameras | Optional. Include as `images.left_wrist` / `images.right_wrist` if present. |
| Robot | Any bimanual arm or single-arm platform. See §10 for embodiment identifiers. |

**Minimum viable setup (no robot):** egocentric camera + SLAM + hand tracking → contributes `images.front_1`, `obs_head_pose`, `left/right.obs_ee_pose`, `left/right.obs_wrist_pose`, `left/right.obs_keypoints`.

**If your SLAM system does not run at 30 fps**, ensure you upsample or interpolate pose tracks to match the RGB frame rate before writing. The training pipeline assumes all arrays are frame-aligned.

### 2.2 Software

```bash
# Clone and install EgoVerse
git clone git@github.com:GaTech-RL2/EgoVerse.git
cd EgoVerse
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

### 2.3 Credentials

You need two things: AWS credentials for the episode registry (PostgreSQL via Secrets Manager) and Cloudflare R2 credentials for the data bucket.

**Step 1 — AWS keys (one-time, ask the consortium lead for these):**
```bash
aws configure
# AccessKeyId: <provided by consortium>
# SecretAccessKey: <provided by consortium>
# Default region: us-east-2
# Output format: (leave blank)
```

**Step 2 — Fetch R2 and DB credentials:**
```bash
bash egomimic/utils/aws/setup_secret.sh
# Writes ~/.egoverse_env with R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY,
# AWS_ENDPOINT_URL_S3, SECRETS_ARN, etc.
```

Verify your setup:
```python
from egomimic.utils.aws.aws_data_utils import load_env
from egomimic.utils.aws.aws_sql import create_default_engine

load_env()
engine = create_default_engine()   # should print: Tables in schema 'app': ['episodes']
```

---

## 3. Episode Hash Convention

Every episode is identified by a **UTC timestamp** rendered as:

```
YYYY-MM-DD-HH-MM-SS-ffffff
```

where `ffffff` is microseconds zero-padded to 6 digits.

Examples:
```
2025-10-14-04-15-30-000000
2026-01-12-03-47-29-664000
```

**Rules:**
- The episode hash is the **primary key** in the database. It must be globally unique.
- Use the UTC wall-clock time at the **start of the recording** as the hash.
- If your hardware does not produce a UTC timestamp natively, convert from device clock using a synchronized offset.
- The `.zarr` directory on S3 is named exactly `<episode_hash>.zarr`.

**Python helpers:**
```python
from egomimic.utils.aws.aws_sql import episode_hash_to_timestamp_ms, timestamp_ms_to_episode_hash

# Convert a UTC epoch millisecond integer to an episode hash string
hash_str = timestamp_ms_to_episode_hash(1736651249664)
# → "2026-01-12-03-47-29-664000"

# Convert back
ts_ms = episode_hash_to_timestamp_ms("2026-01-12-03-47-29-664000")
# → 1736651249664
```

---

## 4. Database Registry

Every episode must be registered in the PostgreSQL `app.episodes` table **before** its Zarr store is uploaded. The registry is the authoritative index used by all download and training tooling.

### 4.1 Schema

```python
@dataclass
class TableRow:
    # ── Required at insert time ──────────────────────────────────────────────
    episode_hash: str       # PRIMARY KEY. UTC timestamp string (see §3)
    operator:     str       # Person who collected the episode (e.g. "jane_doe")
    lab:          str       # Your lab/org identifier (e.g. "stanford", "scale_ai")
    task:         str       # Task name from the taxonomy (see §15)
    embodiment:   str       # Embodiment string (see §10)
    robot_name:   str       # Hardware variant (e.g. "aria_bimanual", "eva_bimanual")

    # ── Optional / updated by processing pipeline ───────────────────────────
    num_frames:             int   = -1    # Set after conversion
    task_description:       str   = ""   # Free-text description of the specific trial
    scene:                  str   = ""   # Scene identifier (e.g. "kitchen_A")
    objects:                str   = ""   # Comma-separated list of objects in the trial
    processed_path:         str   = ""   # S3 path to LeRobot processed files (legacy)
    zarr_processed_path:    str   = ""   # S3 path: s3://rldb/processed_v3/<emb>/<hash>.zarr
    zarr_mp4_path:          str   = ""   # S3 path to preview MP4 (optional)
    processing_error:       str   = ""   # Non-empty if processing failed
    zarr_processing_error:  str   = ""   # Non-empty if Zarr conversion failed
    mp4_path:               str   = ""   # S3 path to raw MP4 (optional)
    is_deleted:             bool  = False
    is_eval:                bool  = False  # True → held-out evaluation episode
    eval_score:             float = -1
    eval_success:           bool  = True
```

**Field constraints:**
- `episode_hash`: must match the `.zarr` directory name exactly.
- `lab`: use a short, stable, lowercase string. Once set, do not change it (used in filters).
- `task`: must be one of the values in §15 (or pre-approved with the consortium).
- `embodiment`: must be one of the strings in §10.
- `robot_name`: finer-grained variant; use the format `<platform>_<config>` (e.g. `aria_bimanual`, `aria_right_arm`).

### 4.2 Inserting a Row

```python
from egomimic.utils.aws.aws_sql import TableRow, add_episode, create_default_engine
from egomimic.utils.aws.aws_data_utils import load_env

load_env()
engine = create_default_engine()

row = TableRow(
    episode_hash   = "2026-03-15-14-22-10-000000",
    operator       = "jane_doe",
    lab            = "stanford",
    task           = "fold_clothes",
    embodiment     = "aria_bimanual",
    robot_name     = "aria_bimanual",
    task_description = "folding a 2T baby shirt on a blue table",
    scene          = "kitchen_A",
    objects        = "baby_shirt_2T",
    num_frames     = 2712,
)

add_episode(engine, row)
```

`add_episode` raises `RuntimeError` on a duplicate `episode_hash`. Check for collisions before inserting.

### 4.3 Updating a Row After Upload

```python
from egomimic.utils.aws.aws_sql import update_episode

row.zarr_processed_path = "s3://rldb/processed_v3/aria/2026-03-15-14-22-10-000000.zarr"
row.num_frames = 2712
update_episode(engine, row)
```

---

## 5. Zarr v3 Episode Format

Each episode is a **Zarr v3 group** (a directory ending in `.zarr`) containing arrays and top-level attributes.

### 5.1 Directory Structure

```
<episode_hash>.zarr/
├── zarr.json                       ← top-level group metadata + episode attrs
├── annotations/                    ← language annotations (may be empty)
│   ├── zarr.json
│   └── c/                          ← chunk data
├── images.front_1/                 ← egocentric RGB frames (required)
│   ├── zarr.json
│   └── c/
├── images.left_wrist/              ← left wrist camera RGB frames (optional)
│   ├── zarr.json
│   └── c/
├── images.right_wrist/             ← right wrist camera RGB frames (optional)
│   ├── zarr.json
│   └── c/
├── left.obs_ee_pose/               ← left end-effector pose (required for bimanual)
├── right.obs_ee_pose/              ← right end-effector pose (required for bimanual)
├── left.obs_wrist_pose/            ← left wrist pose (required if hand tracking available)
├── right.obs_wrist_pose/           ← right wrist pose (required if hand tracking available)
├── left.obs_keypoints/             ← left hand keypoints (required if hand tracking available)
├── right.obs_keypoints/            ← right hand keypoints (required if hand tracking available)
├── obs_head_pose/                  ← egocentric device pose (required)
├── obs_eye_gaze/                   ← eye gaze direction (if available)
└── obs_rgb_timestamps_ns/          ← per-frame capture timestamps
```

### 5.2 Required Arrays

All arrays are indexed along axis 0 by frame index. Every array must have **exactly `total_frames` entries** along axis 0 (matching the value in `zarr.attrs["total_frames"]`).

#### Images

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `images.front_1` | `(T,)` of variable-length bytes | `VariableLengthBytes` | JPEG-encoded RGB frames; see §8 |
| `images.left_wrist` | `(T,)` of variable-length bytes | `VariableLengthBytes` | Optional. Include if wrist camera present. |
| `images.right_wrist` | `(T,)` of variable-length bytes | `VariableLengthBytes` | Optional. Include if wrist camera present. |

#### Egocentric Device Pose (all contributors)

| Key | Shape | Dtype | Frame | Notes |
|---|---|---|---|---|
| `obs_head_pose` | `(T, 7)` | `float64` | SLAM world frame | 6-DOF pose of the egocentric camera/device as XYZWXYZ; see §7. This is the pivot used at training time to re-express all other poses into head-relative coordinates. **Required for all contributors.** |

#### Hand and Wrist Poses (if hand tracking is available)

Provide these if your setup produces 3D hand estimates. Omit the entire key (do not write zeros) if not available.

| Key | Shape | Dtype | Frame | Notes |
|---|---|---|---|---|
| `left.obs_ee_pose` | `(T, 7)` | `float64` | SLAM world frame | Left hand end-effector (fingertip centroid or palm center) pose as XYZWXYZ |
| `right.obs_ee_pose` | `(T, 7)` | `float64` | SLAM world frame | Right hand end-effector pose as XYZWXYZ |
| `left.obs_wrist_pose` | `(T, 7)` | `float64` | SLAM world frame | Left wrist origin pose as XYZWXYZ |
| `right.obs_wrist_pose` | `(T, 7)` | `float64` | SLAM world frame | Right wrist origin pose as XYZWXYZ |
| `left.obs_keypoints` | `(T, 63)` | `float64` | SLAM world frame | 21 hand landmarks × 3 (x, y, z); flattened row-major (see ordering below) |
| `right.obs_keypoints` | `(T, 63)` | `float64` | SLAM world frame | 21 hand landmarks × 3 (x, y, z); flattened row-major |

**If your system only provides wrist pose (not full keypoints)**, include `*.obs_wrist_pose` and `*.obs_ee_pose` and omit `*.obs_keypoints`.

**If your system provides only a single aggregate hand pose** (e.g. palm center from a depth sensor), populate `*.obs_ee_pose` only.

Keypoint ordering (21 landmarks):
```
Index  0-4:   fingertips (thumb, index, middle, ring, pinky)
Index  5:     wrist
Index  6-7:   thumb (CMC, MCP)
Index  8-10:  index (MCP, PIP, DIP)
Index 11-13:  middle (MCP, PIP, DIP)
Index 14-16:  ring (MCP, PIP, DIP)
Index 17-19:  pinky (MCP, PIP, DIP)
```

#### Robot Arm Poses (if operating alongside a robot)

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `left.obs_ee_pose` | `(T, 7)` | `float64` | Left arm EEF pose as XYZWXYZ in robot base frame |
| `right.obs_ee_pose` | `(T, 7)` | `float64` | Right arm EEF pose as XYZWXYZ in robot base frame |
| `left.obs_gripper` | `(T, 1)` | `float64` | Left gripper aperture in [0, 1] (0 = fully closed) |
| `right.obs_gripper` | `(T, 1)` | `float64` | Right gripper aperture in [0, 1] |
| `left.cmd_ee_pose` | `(T, 7)` | `float64` | Commanded left EEF pose (if available) |
| `right.cmd_ee_pose` | `(T, 7)` | `float64` | Commanded right EEF pose (if available) |
| `left.cmd_gripper` | `(T, 1)` | `float64` | Commanded left gripper (if available) |
| `right.cmd_gripper` | `(T, 1)` | `float64` | Commanded right gripper (if available) |

#### Timestamps and Misc

| Key | Shape | Dtype | Notes |
|---|---|---|---|
| `obs_rgb_timestamps_ns` | `(T,)` | `int64` | UTC nanoseconds for each RGB frame |
| `obs_eye_gaze` | `(T, 3)` | `float64` | Unit gaze direction vector in SLAM world frame (x, y, z) |

### 5.3 Top-Level Attributes (`zarr.attrs`)

The root group's `.attrs` dictionary is the **episode metadata**. It is written as JSON and is the primary indexing surface.

```python
{
    "embodiment":        str,   # e.g. "aria_bimanual"  (must match DB row)
    "total_frames":      int,   # number of valid frames (not padded)
    "fps":               int,   # capture frame rate (typically 30)
    "task_name":         str,   # e.g. "fold_clothes"  (must match DB row)
    "task_description":  str,   # free-text description of the trial
    "features": {
        "<key>": {
            "dtype":  str,        # numpy dtype string, or "jpeg" for images, "json" for annotations
            "shape":  list[int],  # per-frame shape (no time dimension)
            "names":  list[str],  # dimension labels (e.g. ["dim_0"] or ["height", "width", "channel"])
            # images only:
            # "dtype": "jpeg", "shape": [H, W, 3], "names": ["height", "width", "channel"]
            # annotations only:
            # "dtype": "json", "shape": [N], "names": ["json"], "format": "annotation_v1"
        },
        ...
    }
}
```

**Rules:**
- `total_frames` must equal `len(store["images.front_1"])` and every other non-padded array.
- `fps` must be the actual capture rate of `images.front_1`. Do not set to a target rate if the actual rate differs.
- `features` must have one entry per array key present in the store.
- `embodiment` and `task_name` must exactly match the values in the DB row for this episode.

### 5.4 Storage / Chunking

The `ZarrWriter` class handles chunking automatically. If writing manually:
- **Numeric arrays**: chunk shape `(chunk_timesteps, *frame_shape)` with `chunk_timesteps=100`, sharded to full array shape.
- **Image arrays**: chunk shape `(1,)` (one JPEG blob per chunk), sharded to full array shape.
- **Annotation arrays**: chunk shape `(N,)`, sharded to `(N,)`.
- **Zarr format version**: always **v3** (`zarr_format=3`).

---

## 6. Coordinate Frame Conventions

### 6.1 SLAM World Frame (storage frame)

All poses are stored in the **SLAM world frame** produced by your pose-tracking system (e.g. Aria MPS, ZED SDK, ORB-SLAM3). This is an arbitrary fixed Euclidean frame that is consistent within a single recording session but **not** consistent across sessions or between different hardware setups.

- Origin: defined by the SLAM system at recording start; treat as opaque.
- Axes: right-handed, metric (meters).
- **This is what you write into the Zarr arrays.** Do not pre-transform poses to any other frame before writing.

The SLAM world frame origin and orientation will differ between labs and hardware. That is expected and fine — the training-time head-frame normalization (§6.2) cancels out any global offset or rotation.

### 6.2 Head Frame (training frame)

At training time, the pipeline automatically re-expresses all poses **relative to the current egocentric device pose** (`obs_head_pose`) using `ActionChunkCoordinateFrameTransform`. You do **not** need to do this conversion yourself; it is applied on-the-fly by the data loader.

The head frame is:
- Origin: the egocentric camera/device center at the current timestep.
- +Z: forward (into the scene from the camera).
- +X: right.
- +Y: up.

### 6.3 Wrist Frame (optional training frame)

For keypoint-based models, keypoints can optionally be further expressed relative to the wrist frame via `PoseCoordinateFrameTransform`. Again, this is a training-time transform; store everything in the SLAM world frame.

### 6.4 Frame Summary

| Array | Written in | Re-expressed at train time |
|---|---|---|
| `left.obs_ee_pose` | SLAM world | Head frame |
| `right.obs_ee_pose` | SLAM world | Head frame |
| `left.obs_wrist_pose` | SLAM world | Head frame |
| `right.obs_wrist_pose` | SLAM world | Head frame |
| `left.obs_keypoints` | SLAM world | Head frame, then optionally wrist frame |
| `right.obs_keypoints` | SLAM world | Head frame, then optionally wrist frame |
| `obs_head_pose` | SLAM world | Used as the re-expression pivot; deleted from batch after transform |
| `obs_eye_gaze` | SLAM world | Not re-expressed (stored as unit direction) |
| Robot `*.obs_ee_pose` | Robot base frame | Robot base frame (no re-expression) |

---

## 7. Pose Representation

### 7.1 Format: XYZWXYZ

All 6-DOF poses are stored as a **7-element float64 vector**:

```
[tx, ty, tz, qw, qx, qy, qz]
 └─ position ─┘  └─ quaternion ─┘
```

| Index | Symbol | Meaning |
|---|---|---|
| 0 | tx | Translation X (meters) |
| 1 | ty | Translation Y (meters) |
| 2 | tz | Translation Z (meters) |
| 3 | qw | Quaternion scalar (real part) |
| 4 | qx | Quaternion i-component |
| 5 | qy | Quaternion j-component |
| 6 | qz | Quaternion k-component |

**Important:** The quaternion uses **scalar-first** order (`w, x, y, z`). This matches the convention used by `projectaria_tools.core.sophus.SE3`. Note that `scipy.spatial.transform.Rotation` uses scalar-last (`x, y, z, w`) by default — use the helpers below to convert. The quaternion must be **unit-norm**: `sqrt(qw² + qx² + qy² + qz²) == 1.0`.

**Conversion helpers:**
```python
from egomimic.utils.pose_utils import xyzw_to_wxyz, wxyz_to_xyzw

# scipy uses scalar-last [qx, qy, qz, qw]; EgoVerse uses scalar-first [qw, qx, qy, qz]
scipy_quat = rotation.as_quat()           # [qx, qy, qz, qw]
egoverse_quat = xyzw_to_wxyz(scipy_quat)  # → [qw, qx, qy, qz]
```

**Converting from SE3 matrix:**
```python
from egomimic.utils.pose_utils import _matrix_to_xyzwxyz
import numpy as np

# mat: (B, 4, 4) SE3 homogeneous matrices
poses = _matrix_to_xyzwxyz(mat[np.newaxis])  # → (1, 7)
```

### 7.2 Alternative: XYZYPR (Euler)

Some robot embodiments store commanded actions as `[tx, ty, tz, yaw, pitch, roll]` (6-element). This is the `xyzypr` format using **intrinsic ZYX Euler angles in radians**. Obs poses are always XYZWXYZ; this format is only used for action chunks after training-time transformation.

### 7.3 Confidence Filtering

Many hand-tracking systems produce a per-frame confidence or quality score alongside each pose estimate. **Filter out frames below your system's minimum reliable confidence threshold before writing.** Low-confidence poses (near-zero or degenerate quaternions) cause SVD convergence errors in the coordinate frame transforms at training time.

If your system does not provide a confidence score, apply a basic sanity check: reject any frame where the quaternion norm deviates from 1.0 by more than 1e-3, or where the translation jumps by more than a physically plausible amount between consecutive frames (e.g. > 0.5 m in a single timestep at 30 fps).

---

## 8. Image Format

### 8.1 Encoding

| Property | Value |
|---|---|
| Compression | JPEG |
| Quality | **85** (fixed; do not change) |
| Colorspace | **RGB** (not BGR — be careful if you use OpenCV which defaults to BGR) |
| Per-frame shape | `(H, W, 3)` uint8 |
| Zarr dtype | `VariableLengthBytes` (Zarr v3 `vlen-bytes`) |
| Codec stack | `sharding_indexed` → `vlen-bytes` → `zstd(level=0)` |

There is no fixed resolution requirement. Record the actual frame dimensions in `features["images.front_1"]["shape"]` (e.g. `[480, 640, 3]`). All frames in a given episode must have the same resolution. Common resolutions: 480×640 (Aria), 720×1280, 1080×1920. If your camera produces non-standard aspect ratios, do not crop or pad — write as-is and document the shape in `features`.

### 8.2 Writing Images

```python
import simplejpeg
import numpy as np

# rgb_frame: (H, W, 3) uint8 numpy array in RGB order
jpeg_bytes = simplejpeg.encode_jpeg(rgb_frame, quality=85, colorspace="RGB")
```

**If you have OpenCV frames (BGR):**
```python
import cv2
rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
jpeg_bytes = simplejpeg.encode_jpeg(rgb_frame, quality=85, colorspace="RGB")
```

### 8.3 Reading Images

```python
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode
import simplejpeg

ep = ZarrEpisode("/path/to/<episode_hash>")
data = ep.read({"images.front_1": (frame_idx, None)})  # single frame
rgb = simplejpeg.decode_jpeg(bytes(data["images.front_1"]), colorspace="RGB")
# rgb: (H, W, 3) uint8
```

---

## 9. Language Annotations

Language annotations are **optional but strongly encouraged**. They are stored as a span-based structure: each annotation covers a contiguous range of frames.

### 9.1 Format (`annotation_v1`)

The `annotations` array in the Zarr store contains `N` entries, where `N` is the total number of annotation spans in the episode (not the number of frames). Each entry is a UTF-8-encoded JSON string:

```json
{"text": "pick up the shirt", "start_idx": 0, "end_idx": 145}
```

| Field | Type | Description |
|---|---|---|
| `text` | `str` | Natural-language description of what is happening during `[start_idx, end_idx)` |
| `start_idx` | `int` | First frame index (inclusive) |
| `end_idx` | `int` | Last frame index (exclusive). Must satisfy `0 <= start_idx < end_idx <= total_frames`. |

**Rules:**
- Spans may overlap.
- Spans do not need to cover the entire episode.
- `text` must be in English.
- Use the imperative or present-continuous form: "pick up the shirt", "folding the left sleeve", etc.
- Do **not** encode task-level descriptions here (those go in `task_description`). Use annotations for sub-step descriptions.
- An empty `annotations` array (shape `(0,)`) is valid when no annotation is available.

### 9.2 Annotation Granularity

Use at minimum one annotation per task phase. For `fold_clothes`, for example:

| Phase | Example annotation text |
|---|---|
| Grasp | "grasping the shirt by the collar" |
| Unfold | "unfolding and laying the shirt flat" |
| Fold left sleeve | "folding the left sleeve towards the center" |
| Fold right sleeve | "folding the right sleeve towards the center" |
| Fold body | "folding the bottom half up to complete the fold" |

### 9.3 Writing Annotations

Via `ZarrWriter`:
```python
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

annotations = [
    ("grasping the shirt by the collar",        0,   145),
    ("unfolding and laying the shirt flat",    145,   420),
    ("folding the left sleeve towards center", 420,   680),
    ("folding the right sleeve",               680,   910),
    ("folding the bottom half up",             910,  1200),
]

writer = ZarrWriter(
    episode_path="path/to/<episode_hash>.zarr",
    embodiment="aria_bimanual",
    fps=30,
    task_name="fold_clothes",
    task_description="folding a 2T baby shirt",
    annotations=annotations,
)
```

To append annotations to an **existing** Zarr store:
```python
writer = ZarrWriter(episode_path="path/to/<episode_hash>.zarr")
writer.append_annotations(
    annotation_key="annotations",
    annotations=annotations,
    mode="w",   # "w" = overwrite existing, "a" = append
)
```

### 9.4 Scale AI Annotation Format

If you are delivering data through Scale AI, annotations are generated via the Scale annotation API. The `ScaleAnnotationDatasetFilter` class can be used to filter episodes to only those with completed Scale annotations. Set `SCALE_API_KEY` in your environment.

---

## 10. Embodiment Identifiers

The `embodiment` field in the DB row and in `zarr.attrs` must be one of the following strings. The `robot_name` field is the same string (fine-grained variant names are allowed in `robot_name` but `embodiment` must match this list exactly).

| `embodiment` string | Integer ID | Description |
|---|---|---|
| `aria_bimanual` | 5 | Project Aria glasses + two-arm human demonstration |
| `aria_right_arm` | 3 | Project Aria glasses + right arm only |
| `aria_left_arm` | 4 | Project Aria glasses + left arm only |
| `eva_bimanual` | 8 | Eva camera + bimanual robot |
| `eva_right_arm` | 6 | Eva camera + right arm robot |
| `eva_left_arm` | 7 | Eva camera + left arm robot |
| `mecka_bimanual` | 9 | Mecka AI hardware + bimanual |
| `mecka_right_arm` | 10 | Mecka AI hardware + right arm |
| `mecka_left_arm` | 11 | Mecka AI hardware + left arm |
| `scale_bimanual` | 12 | Scale AI EgoDex + bimanual |
| `scale_right_arm` | 13 | Scale AI EgoDex + right arm |
| `scale_left_arm` | 14 | Scale AI EgoDex + left arm |

**If your hardware is not in this list**, contact the consortium leads to register a new embodiment identifier before submitting data.

---

## 11. Writing an Episode (Code)

### 11.1 Using `ZarrWriter` (recommended)

```python
import numpy as np
from pathlib import Path
from egomimic.rldb.zarr.zarr_writer import ZarrWriter

# ── Your data ────────────────────────────────────────────────────────────────
T = 2712                         # total frames
fps = 30
episode_hash = "2026-03-15-14-22-10-000000"
out_path = Path(f"/local/processed/{episode_hash}.zarr")

# Numeric arrays: all shape (T, ...) float64
left_ee   = np.zeros((T, 7), dtype=np.float64)   # XYZWXYZ
right_ee  = np.zeros((T, 7), dtype=np.float64)
left_wrist  = np.zeros((T, 7), dtype=np.float64)
right_wrist = np.zeros((T, 7), dtype=np.float64)
head_pose = np.zeros((T, 7), dtype=np.float64)
left_kp   = np.zeros((T, 63), dtype=np.float64)  # 21 landmarks × 3
right_kp  = np.zeros((T, 63), dtype=np.float64)
eye_gaze  = np.zeros((T, 3), dtype=np.float64)   # unit direction
ts_ns     = np.zeros((T,), dtype=np.int64)        # UTC nanoseconds

# Image array: (T, H, W, 3) uint8 RGB
images_front = np.zeros((T, 480, 640, 3), dtype=np.uint8)

# Language annotations: list of (text, start_idx, end_idx)
annotations = [
    ("grasping the shirt", 0, 200),
    ("folding the shirt",  200, T),
]

# ── Write ─────────────────────────────────────────────────────────────────────
writer = ZarrWriter(
    episode_path    = out_path,
    embodiment      = "aria_bimanual",
    fps             = fps,
    task_name       = "fold_clothes",
    task_description = "folding a 2T baby shirt on kitchen table",
    annotations     = annotations,
    chunk_timesteps = 100,
)

writer.write(
    numeric_data = {
        "left.obs_ee_pose":    left_ee,
        "right.obs_ee_pose":   right_ee,
        "left.obs_wrist_pose": left_wrist,
        "right.obs_wrist_pose": right_wrist,
        "obs_head_pose":       head_pose,
        "left.obs_keypoints":  left_kp,
        "right.obs_keypoints": right_kp,
        "obs_eye_gaze":        eye_gaze,
        "obs_rgb_timestamps_ns": ts_ns,
    },
    image_data = {
        "images.front_1": images_front,   # ZarrWriter handles JPEG encoding
    },
)
```

### 11.2 Incremental Writing (when you cannot load all frames into RAM)

```python
writer = ZarrWriter(
    episode_path = out_path,
    embodiment   = "aria_bimanual",
    fps          = 30,
    task_name    = "fold_clothes",
    annotations  = annotations,
)

with writer.write_incremental() as inc:
    for frame_idx in range(T):
        inc.add_frame(
            numeric = {
                "left.obs_ee_pose":    left_ee[frame_idx],     # shape (7,)
                "right.obs_ee_pose":   right_ee[frame_idx],
                "left.obs_wrist_pose": left_wrist[frame_idx],
                "right.obs_wrist_pose": right_wrist[frame_idx],
                "obs_head_pose":       head_pose[frame_idx],
                "left.obs_keypoints":  left_kp[frame_idx],
                "right.obs_keypoints": right_kp[frame_idx],
                "obs_eye_gaze":        eye_gaze[frame_idx],
                "obs_rgb_timestamps_ns": ts_ns[frame_idx:frame_idx+1],
            },
            images = {
                "images.front_1": images_front[frame_idx],  # shape (H, W, 3) uint8
            },
        )
# Annotations and metadata are written automatically on __exit__
```

### 11.3 Verifying the Written Episode

```python
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode

ep = ZarrEpisode(out_path)
print(ep)               # ZarrEpisode(path=..., frames=2712)
print(ep.metadata)      # dict with embodiment, total_frames, fps, features, ...

# Spot-check a frame
data = ep.read({
    "images.front_1":   (0, None),
    "left.obs_ee_pose": (0, None),
})
import simplejpeg
frame = simplejpeg.decode_jpeg(bytes(data["images.front_1"]), colorspace="RGB")
print(frame.shape)      # (480, 640, 3)
print(data["left.obs_ee_pose"])  # [tx, ty, tz, qw, qx, qy, qz]
```

---

## 12. Registering Episodes in the Database

After writing the Zarr store locally, register the episode in the DB and update its `zarr_processed_path`:

```python
from egomimic.utils.aws.aws_sql import TableRow, add_episode, update_episode, create_default_engine
from egomimic.utils.aws.aws_data_utils import load_env

load_env()
engine = create_default_engine()

episode_hash = "2026-03-15-14-22-10-000000"
s3_zarr_path = f"s3://rldb/processed_v3/aria/{episode_hash}.zarr"

# Step 1: Insert the row
row = TableRow(
    episode_hash     = episode_hash,
    operator         = "jane_doe",
    lab              = "stanford",
    task             = "fold_clothes",
    embodiment       = "aria_bimanual",
    robot_name       = "aria_bimanual",
    task_description = "folding a 2T baby shirt",
    num_frames       = 2712,
)
add_episode(engine, row)

# Step 2: Upload to S3 (see §13), then update zarr_processed_path
row.zarr_processed_path = s3_zarr_path
update_episode(engine, row)
```

---

## 13. Uploading to S3

### 13.1 S3 Path Convention

```
s3://rldb/processed_v3/<embodiment_prefix>/<episode_hash>.zarr/
```

| Embodiment | `<embodiment_prefix>` |
|---|---|
| `aria_*` | `aria` |
| `eva_*` | `eva` |
| `mecka_*` | `mecka` |
| `scale_*` | `scale` |

Examples:
```
s3://rldb/processed_v3/aria/2026-03-15-14-22-10-000000.zarr/
s3://rldb/processed_v3/eva/2025-11-04-09-30-00-000000.zarr/
```

### 13.2 Upload with `s5cmd`

`s5cmd` is the recommended upload tool (installed as part of the Python environment).

```bash
# Upload a local .zarr directory
s5cmd --endpoint-url $AWS_ENDPOINT_URL_S3 \
      sync "/local/processed/2026-03-15-14-22-10-000000.zarr/*" \
           "s3://rldb/processed_v3/aria/2026-03-15-14-22-10-000000.zarr/"
```

Or using the Python utility:
```python
from egomimic.utils.aws.aws_data_utils import upload_dir_to_s3, load_env

load_env()
upload_dir_to_s3(
    local_dir = "/local/processed/2026-03-15-14-22-10-000000.zarr",
    bucket    = "rldb",
    prefix    = "processed_v3/aria/2026-03-15-14-22-10-000000.zarr",
)
```

### 13.3 Bulk Upload with Ray

For batch uploads of many episodes, use Ray to parallelize:

```python
import ray
from egomimic.utils.aws.aws_data_utils import upload_dir_to_s3, load_env

ray.init()

@ray.remote
def upload_one(local_zarr_path: str, s3_prefix: str):
    load_env()
    upload_dir_to_s3(local_zarr_path, bucket="rldb", prefix=s3_prefix)

tasks = [
    upload_one.remote(
        f"/local/processed/{h}.zarr",
        f"processed_v3/aria/{h}.zarr"
    )
    for h in episode_hashes
]
ray.get(tasks)
```

---

## 14. Validation and Verification

### 14.1 Automated Checks

Run these checks on every episode before uploading:

```python
import zarr, numpy as np
from pathlib import Path
from egomimic.rldb.zarr.zarr_dataset_multi import ZarrEpisode
import simplejpeg

def validate_episode(zarr_path: str) -> list[str]:
    """Returns a list of error strings. Empty list = pass."""
    errors = []
    ep = ZarrEpisode(zarr_path)
    meta = ep.metadata
    T = meta["total_frames"]
    store = zarr.open(zarr_path, mode="r")

    # ── Metadata ────────────────────────────────────────────────────────────
    for field in ("embodiment", "total_frames", "fps", "task_name", "features"):
        if field not in meta:
            errors.append(f"Missing metadata field: {field}")

    if meta.get("fps", 0) not in (30, 60):
        errors.append(f"Unexpected fps={meta['fps']}. Expected 30 or 60.")

    # ── Frame counts ────────────────────────────────────────────────────────
    for key in store.keys():
        arr_len = store[key].shape[0]
        if arr_len < T:
            errors.append(f"{key}: array length {arr_len} < total_frames {T}")

    # ── Required keys ───────────────────────────────────────────────────────
    required = ["images.front_1", "left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose"]
    for key in required:
        if key not in store:
            errors.append(f"Missing required key: {key}")

    # ── Pose shapes and norms ───────────────────────────────────────────────
    for key in ("left.obs_ee_pose", "right.obs_ee_pose", "obs_head_pose"):
        if key in store:
            arr = store[key][:]
            if arr.shape != (T, 7) and arr.shape[0] >= T:
                arr = arr[:T]
            if arr.shape[-1] != 7:
                errors.append(f"{key}: expected shape (T, 7), got {arr.shape}")
                continue
            quat = arr[:, 3:7]
            norms = np.linalg.norm(quat, axis=1)
            if not np.allclose(norms, 1.0, atol=1e-4):
                bad = np.where(np.abs(norms - 1.0) > 1e-4)[0]
                errors.append(f"{key}: {len(bad)} frames with non-unit quaternions (e.g. frame {bad[0]}, norm={norms[bad[0]]:.6f})")

    # ── Keypoint shapes ─────────────────────────────────────────────────────
    for key in ("left.obs_keypoints", "right.obs_keypoints"):
        if key in store:
            arr = store[key][:]
            if arr.shape[-1] != 63:
                errors.append(f"{key}: expected last dim 63 (21×3), got {arr.shape[-1]}")

    # ── Image decodability (spot-check first frame) ──────────────────────────
    if "images.front_1" in store:
        data = ep.read({"images.front_1": (0, None)})
        try:
            frame = simplejpeg.decode_jpeg(bytes(data["images.front_1"]), colorspace="RGB")
            if frame.ndim != 3 or frame.shape[2] != 3:
                errors.append(f"images.front_1: decoded frame has unexpected shape {frame.shape}")
        except Exception as e:
            errors.append(f"images.front_1: failed to decode frame 0: {e}")

    return errors

# Usage
errors = validate_episode("/local/processed/2026-03-15-14-22-10-000000.zarr")
if errors:
    for e in errors:
        print("ERROR:", e)
else:
    print("All checks passed.")
```

### 14.2 End-to-End Load Test

Verify the episode loads correctly through the full training pipeline before uploading:

```python
from pathlib import Path
from egomimic.rldb.zarr.zarr_dataset_multi import LocalEpisodeResolver, MultiDataset
from egomimic.rldb.filters import DatasetFilter
from egomimic.rldb.embodiment.human import Aria
import torch

key_map = Aria.get_keymap(keymap_mode="cartesian")
transform_list = Aria.get_transform_list(mode="cartesian")

resolver = LocalEpisodeResolver(
    folder_path    = Path("/local/processed"),
    key_map        = key_map,
    transform_list = transform_list,
)

filters = DatasetFilter(filter_lambdas=[
    "lambda row: row['episode_hash'] == '2026-03-15-14-22-10-000000'"
])

ds = MultiDataset._from_resolver(resolver, filters=filters, mode="total")
loader = torch.utils.data.DataLoader(ds, batch_size=4, num_workers=0)

batch = next(iter(loader))
print("Keys:", list(batch.keys()))
print("actions_cartesian:", batch["actions_cartesian"].shape)  # (4, 100, 12)
print("observations.images.front_img_1:", batch["observations.images.front_img_1"].shape)  # (4, 3, 480, 640)
```

Expected output for a valid Aria bimanual episode in cartesian mode:
- `actions_cartesian`: `(B, 100, 12)` — 100-step action chunk, 6 DOF × 2 arms
- `observations.state.ee_pose`: `(B, 12)` — current EEF poses, 6 DOF × 2 arms
- `observations.images.front_img_1`: `(B, 3, H, W)` — normalized RGB in `[0, 1]`

---

## 15. Task Taxonomy

The `task` field in the DB row and `task_name` in `zarr.attrs` must be one of the following canonical strings. Contact the consortium leads to propose new tasks.

| `task_name` | Description | Arms | Objects |
|---|---|---|---|
| `fold_clothes` | Fold a garment flat | Bimanual | 2T baby shirt (default); adult shirts allowed |
| `object_in_container` | Pick object and place into a container | Bimanual | 10 objects × 10 containers × 8 scenes |
| `scoop_granular` | Scoop granular material from one container to another | Bimanual | Defined by scene setup |
| `bag_grocery` | Place grocery items into a paper or reusable bag | Bimanual | Defined by scene setup |
| `put_cup_on_saucer` | Place a cup precisely onto a saucer | Bimanual | Standard coffee cup + saucer |
| `sort_utensils` | Organize utensils into a tray or drawer | Bimanual | Standard kitchen utensil set |

**Task consistency requirements:**
- **Object sizes must be consistent across labs.** For `fold_clothes`, use **2T (toddler) shirts** unless explicitly approved otherwise. Size consistency is critical for measuring cross-lab transfer.
- **Scene setup should be documented** in the DB `scene` and `objects` fields so downstream filtering is possible.
- **Evaluation episodes** (`is_eval=True`) should come from a held-out set not seen during collection. Use 50 rollouts per task position for statistical confidence (Clopper-Pearson 95% CI).

---

## 16. Pre-Submission Checklist

Complete every item before considering an episode ready for upload.

**Episode hash**
- [ ] Episode hash is a valid UTC timestamp string (`YYYY-MM-DD-HH-MM-SS-ffffff`).
- [ ] Episode hash is unique — not already in the DB (`episode_hash_to_table_row(engine, hash)` returns `None`).

**Zarr format**
- [ ] Zarr v3 format (`zarr_format=3` confirmed in `zarr.json`).
- [ ] `total_frames` in `zarr.attrs` equals the actual number of valid frames.
- [ ] All arrays have at least `total_frames` entries along axis 0.
- [ ] `images.front_1` is present and all frames decode successfully.
- [ ] `obs_head_pose` is present (required for all contributors).
- [ ] `left.obs_ee_pose` and `right.obs_ee_pose` are present if hand tracking is available.
- [ ] All `obs_ee_pose` arrays have shape `(T, 7)` and unit-norm quaternions.
- [ ] All `obs_keypoints` arrays have shape `(T, 63)`.
- [ ] `features` dict in `zarr.attrs` has one entry per array key.
- [ ] `embodiment` and `task_name` in `zarr.attrs` match the DB row values.

**Coordinate frames**
- [ ] All poses are in the SLAM world frame (not head frame, not camera frame).
- [ ] Quaternion is stored in XYZWXYZ order: `[tx, ty, tz, qw, qx, qy, qz]`.
- [ ] Translation units are **meters**.

**Images**
- [ ] Images are in **RGB** order (not BGR).
- [ ] JPEG quality is **85**.
- [ ] Image shape matches `features["images.front_1"]["shape"]`.

**Annotations**
- [ ] `annotations` key is present (may be empty array if no annotations available).
- [ ] All `(start_idx, end_idx)` spans satisfy `0 <= start_idx < end_idx <= total_frames`.
- [ ] Annotation text is in English, imperative or present-continuous form.

**Database**
- [ ] DB row inserted before upload.
- [ ] `zarr_processed_path` updated to the correct S3 path after upload.
- [ ] `num_frames` in DB row matches `total_frames` in `zarr.attrs`.
- [ ] `embodiment` in DB row exactly matches the embodiment enum string (§10).

**Upload**
- [ ] Episode is accessible at `s3://rldb/processed_v3/<prefix>/<episode_hash>.zarr/`.
- [ ] `sync_s3.py` with an appropriate filter can download and open the episode.

---

## 17. Getting Access and Contact

### Access Request

To get credentials for the EgoVerse data bucket and episode registry:

1. Email the consortium leads with your lab name, GitHub handle, and a brief description of the data you intend to contribute.
2. You will receive AWS credentials (for Secrets Manager access) and instructions to run `setup_secret.sh`.

### Consortium Leads

| Person | Affiliation | Role |
|---|---|---|
| **Danfei Xu** | Georgia Tech / NVIDIA GEAR | PI, consortium lead |
| **Simar Kareer** | Georgia Tech | Infrastructure, website, data pipeline |
| **Ryan Punamiya** | Georgia Tech / NVIDIA GEAR | Technical lead, format and schema |

### Resources

| Resource | URL |
|---|---|
| Website | https://egoverse.ai |
| Data browser | https://partners.mecka.ai/egoverse |
| arXiv paper | https://arxiv.org/abs/2604.07607 |
| GitHub | https://github.com/GaTech-RL2/EgoVerse |
| License | CC BY-SA 4.0 |
| Onboarding Slack channel (GT workspace) | `#egoverse-onboarding` |

### Reporting Issues

If you encounter processing errors, S3 permission issues, or schema questions, post in `#egoverse-onboarding` with:
- Your episode hash(es)
- The error message or symptom
- The output of `validate_episode()` for the affected episode
