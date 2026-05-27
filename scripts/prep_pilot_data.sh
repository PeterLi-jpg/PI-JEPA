#!/usr/bin/env bash
# =============================================================================
# Acquire all data for the dual-GPU pilot on Brev in parallel.
#
# Runs (in background, on different CPU cores):
#   1. Synthetic Darcy 3D — generate locally (numpy)             ~3-5 min
#   2. ADR Pe/Da sweep    — generate locally + convert to .pt     ~10-20 min
#   3. CCSNet             — gdown from Google Drive               ~5-30 min
#   4. FNO4CO2            — clone repo + bash download.sh         ~5-15 min
#
# Waits for all to finish, prints per-task status. Designed to run ONCE
# on a fresh Brev instance, before launch_dual_gpu_pilot.sh.
#
# Usage on Brev:
#   bash scripts/prep_pilot_data.sh
#
# Each task's stdout/stderr → /tmp/prep_<task>.log. The main script
# also tees a combined summary to /tmp/prep_pilot_data.log.
# =============================================================================

set -uo pipefail

PY="${PYTHON:-.venv/bin/python}"
PIP="${PIP:-.venv/bin/pip}"
LOG_DIR="${LOG_DIR:-/tmp}"
mkdir -p "$LOG_DIR"

# Public IDs / URLs from the dataset README files / Brandon's download_data.py.
CCSNET_DRIVE_FOLDER="1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7"
FNO4CO2_REPO="https://github.com/gegewen/fno4co2.git"

echo "============================================================"
echo "  prep_pilot_data: launching 4 parallel acquisitions"
echo "  logs: $LOG_DIR/prep_*.log"
echo "============================================================"

# ------------------------------------------------------------------
# Task 1: synthetic Darcy 3D
# ------------------------------------------------------------------
task_darcy() {
    local log="$LOG_DIR/prep_darcy.log"
    {
        echo "[darcy] start $(date)"
        if [ -s data/darcy_3d/darcy3d_train.pt ] && [ -s data/darcy_3d/darcy3d_test.pt ]; then
            echo "[darcy] already present — skip"
            return 0
        fi
        "$PY" scripts/generate_darcy_data_3d.py \
            --n-train 1024 --n-test 256 --resolution 64 \
            --out-dir data/darcy_3d --normalize
        echo "[darcy] DONE $(date)"
    } > "$log" 2>&1
}

# ------------------------------------------------------------------
# Task 2: ADR Pe/Da sweep
# ------------------------------------------------------------------
task_adr() {
    local log="$LOG_DIR/prep_adr.log"
    {
        echo "[adr] start $(date)"
        if [ -s data/pdebench_adr/adr_train.pt ] && [ -s data/pdebench_adr/adr_test.pt ]; then
            echo "[adr] already present — skip"
            return 0
        fi
        if [ ! -s data/pdebench_adr/pe_da_sweep.h5 ]; then
            "$PY" scripts/generate_adr_pe_da_sweep.py \
                --n-pe 8 --n-da 8 --n-per-cell 16 \
                --pe-range 0.1 30 --da-range 0.1 30 \
                --grid 64 --t-final 0.5 --n-t 16 \
                --output data/pdebench_adr/pe_da_sweep.h5
        fi
        "$PY" scripts/convert_adr_to_pt.py \
            --input data/pdebench_adr/pe_da_sweep.h5 \
            --out-dir data/pdebench_adr \
            --train-frac 0.8 --normalize
        echo "[adr] DONE $(date)"
    } > "$log" 2>&1
}

# ------------------------------------------------------------------
# Task 3: CCSNet via gdown
# ------------------------------------------------------------------
task_ccsnet() {
    local log="$LOG_DIR/prep_ccsnet.log"
    {
        echo "[ccsnet] start $(date)"
        if [ -s "data/ccsnet/CCSNet_v1.0/test_x.hdf5" ] && \
           [ -s "data/ccsnet/CCSNet_v1.0/test_y_SG.hdf5" ]; then
            echo "[ccsnet] already present — skip"
            return 0
        fi
        # Ensure gdown installed
        if ! "$PY" -c 'import gdown' 2>/dev/null; then
            echo "[ccsnet] installing gdown..."
            "$PIP" install --quiet gdown
        fi
        mkdir -p data/ccsnet
        "$PY" -m gdown --folder \
            "https://drive.google.com/drive/folders/${CCSNET_DRIVE_FOLDER}" \
            -O data/ccsnet
        echo "[ccsnet] DONE $(date)"
    } > "$log" 2>&1
}

# ------------------------------------------------------------------
# Task 4: FNO4CO2 via git clone + bash download.sh
# ------------------------------------------------------------------
task_fno4co2() {
    local log="$LOG_DIR/prep_fno4co2.log"
    {
        echo "[fno4co2] start $(date)"
        if [ -s data/fno4co2/dataset/dP_test_a.pt ] && \
           [ -s data/fno4co2/dataset/dP_test_u.pt ]; then
            echo "[fno4co2] already present — skip"
            return 0
        fi
        mkdir -p data/fno4co2
        if [ ! -d data/fno4co2/repo ]; then
            git clone --quiet "$FNO4CO2_REPO" data/fno4co2/repo
        fi
        cd data/fno4co2/repo
        # The repo ships a download.sh that pulls dataset from Dropbox.
        if [ -x download.sh ]; then
            bash download.sh
        elif [ -x scripts/download.sh ]; then
            bash scripts/download.sh
        else
            echo "[fno4co2] no download.sh found in repo — manual fetch needed"
            ls -la
            exit 1
        fi
        cd ../../..
        # Move dataset to expected location if needed
        if [ ! -d data/fno4co2/dataset ] && [ -d data/fno4co2/repo/dataset ]; then
            ln -sfn ../repo/dataset data/fno4co2/dataset
        fi
        echo "[fno4co2] DONE $(date)"
    } > "$log" 2>&1
}

# Launch all 4 in parallel
task_darcy   & PID_DARCY=$!
task_adr     & PID_ADR=$!
task_ccsnet  & PID_CCSNET=$!
task_fno4co2 & PID_FNO4CO2=$!
echo "  launched: darcy(PID $PID_DARCY) adr(PID $PID_ADR) ccsnet(PID $PID_CCSNET) fno4co2(PID $PID_FNO4CO2)"

# Wait and collect statuses (don't use set -e so a failed one doesn't kill the rest)
wait "$PID_DARCY";   RC_DARCY=$?
wait "$PID_ADR";     RC_ADR=$?
wait "$PID_CCSNET";  RC_CCSNET=$?
wait "$PID_FNO4CO2"; RC_FNO4CO2=$?

echo ""
echo "============================================================"
echo "  prep_pilot_data summary"
status() { [ "$1" = "0" ] && echo "OK" || echo "FAIL (rc=$1)"; }
printf "  darcy:    %-12s — %s\n"  "$(status $RC_DARCY)"   "$LOG_DIR/prep_darcy.log"
printf "  adr:      %-12s — %s\n"  "$(status $RC_ADR)"     "$LOG_DIR/prep_adr.log"
printf "  ccsnet:   %-12s — %s\n"  "$(status $RC_CCSNET)"  "$LOG_DIR/prep_ccsnet.log"
printf "  fno4co2:  %-12s — %s\n"  "$(status $RC_FNO4CO2)" "$LOG_DIR/prep_fno4co2.log"
echo "============================================================"

# Exit 0 if at least the "free" datasets (darcy, adr) succeeded. CCSNet
# and FNO4CO2 may need manual intervention if Google Drive / Dropbox
# rate-limit; the pilot can still run on the subset that's ready.
[ "$RC_DARCY" = "0" ] && [ "$RC_ADR" = "0" ]
