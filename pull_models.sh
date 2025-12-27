#!/usr/bin/env bash
set -euo pipefail

# ====== config (edit these) ======
REMOTE_USER_HOST="rpunamiya6@sky1.cc.gatech.edu"
REMOTE_PATH="/coc/flash7/rpunamiya6/Projects/EgoVerse/logs/everse_object_in_container/cotrain_cartesian_fixed_data_lowlr_2025-12-22_10-48-14"
LOCAL_PATH="./egomimic/robot/models/"
# =================================

mkdir -p "$LOCAL_PATH"

# Prefer system rsync to avoid OpenSSL/conda mismatch
RSYNC_BIN="/usr/bin/rsync"
if [[ ! -x "$RSYNC_BIN" ]]; then
  RSYNC_BIN="$(command -v rsync)"
fi

# Run rsync without Conda/mamba library injection
env -u LD_LIBRARY_PATH -u CONDA_PREFIX -u MAMBA_ROOT_PREFIX \
  "$RSYNC_BIN" -avh --progress --partial --inplace \
  --exclude='/0/videos/***' \
  --exclude='/0/wandb/***' \
  "${REMOTE_USER_HOST}:${REMOTE_PATH%/}/" \
  "${LOCAL_PATH%/}/"
