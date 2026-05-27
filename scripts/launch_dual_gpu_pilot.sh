#!/usr/bin/env bash
# =============================================================================
# Dual-GPU pilot dispatcher.
#
# On a 2× L40S instance, splits a list of datasets evenly across the two
# GPUs and runs them in parallel. Optionally also starts OPM Flow
# new-well sims on the leftover CPU cores so all three workloads
# (GPU 0, GPU 1, CPU) run concurrently.
#
# Each dataset's GPU stream is an independent run_focused_paper.sh
# invocation pinned to its GPU via CUDA_VISIBLE_DEVICES. Outputs live
# under <root>/<dataset>/.  skip_if_done in each stream means a crash
# in one GPU stream doesn't lose progress in the other.
#
# Usage:
#   bash scripts/launch_dual_gpu_pilot.sh OUTPUT_ROOT
#
# Env vars (pilot scope defaults — override for full grid):
#   GPU0_DATASETS="darcy_3d_synthetic ccsnet"   # for GPU 0
#   GPU1_DATASETS="adr_pe_da fno4co2"           # for GPU 1
#   NEWWELL=1                                    # also start OPM CPU sims
#   NEWWELL_N_WELLS=500
#   NEWWELL_N_WORKERS=12                         # leave 4 vCPUs for the GPU streams
#   N_SEEDS=3 EPOCHS_PRETRAIN=200 EPOCHS_FINETUNE=50 EPOCHS_BASELINE=50
#   N_LABELED="50 100 250"
#   BASELINES="fno3d pi_deeponet3d ufno3d"
#   ABLATION_VARIANTS="full no_physics no_chain no_vicreg"
# =============================================================================

set -euo pipefail

OUTPUT_ROOT="${1:-outputs_pilot/dual_gpu}"
PY="${PYTHON:-.venv/bin/python}"

GPU0_DATASETS="${GPU0_DATASETS:-darcy_3d_synthetic ccsnet}"
GPU1_DATASETS="${GPU1_DATASETS:-adr_pe_da fno4co2}"
NEWWELL="${NEWWELL:-0}"
NEWWELL_N_WELLS="${NEWWELL_N_WELLS:-500}"
NEWWELL_N_WORKERS="${NEWWELL_N_WORKERS:-12}"

# Pilot defaults (override for full grid).
export N_SEEDS="${N_SEEDS:-3}"
export EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-200}"
export EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-50}"
export EPOCHS_BASELINE="${EPOCHS_BASELINE:-50}"
export N_LABELED="${N_LABELED:-50 100 250}"
export BASELINES="${BASELINES:-fno3d pi_deeponet3d ufno3d}"
export ABLATION_VARIANTS="${ABLATION_VARIANTS:-full no_physics no_chain no_vicreg}"
export RESIZE_CUBE="${RESIZE_CUBE:-64}"
export PYTHON="$PY"

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/_logs"

echo "=============================================================="
echo "  DUAL-GPU PARALLEL PILOT"
echo "  output:            $OUTPUT_ROOT"
echo "  GPU 0 datasets:    $GPU0_DATASETS"
echo "  GPU 1 datasets:    $GPU1_DATASETS"
echo "  CPU new-well:      $([ "$NEWWELL" = "1" ] && echo "ENABLED ($NEWWELL_N_WELLS sims, $NEWWELL_N_WORKERS workers)" || echo "disabled")"
echo "  N_SEEDS=$N_SEEDS  EPOCHS_PRETRAIN=$EPOCHS_PRETRAIN  N_LABELED=\"$N_LABELED\""
echo "  BASELINES=\"$BASELINES\""
echo "=============================================================="

# Helper: run a sequence of datasets on a pinned GPU, logging to a file.
run_gpu_stream() {
    local gpu_id="$1"
    local datasets="$2"
    local stream_log="$OUTPUT_ROOT/_logs/gpu${gpu_id}.log"
    {
        for ds in $datasets; do
            echo ""
            echo "=== [GPU $gpu_id] starting $ds @ $(date) ==="
            if CUDA_VISIBLE_DEVICES="$gpu_id" \
                bash scripts/run_focused_paper.sh "$OUTPUT_ROOT/$ds" "$ds"; then
                echo "=== [GPU $gpu_id] $ds OK @ $(date) ==="
            else
                echo "=== [GPU $gpu_id] $ds FAILED (exit $?) @ $(date) ==="
            fi
        done
    } > "$stream_log" 2>&1
}

# ---- Background: New-Well OPM Flow sims on CPU ----
NEWWELL_PID=""
if [ "$NEWWELL" = "1" ]; then
    if ! command -v flow >/dev/null 2>&1; then
        echo "  [bg] flow not on PATH — installing OPM via setup_opm_flow.sh..."
        bash scripts/setup_opm_flow.sh > "$OUTPUT_ROOT/_logs/opm_install.log" 2>&1 \
          || { echo "  [bg] OPM install failed (see log); skipping new-well"; NEWWELL=0; }
    fi
fi
if [ "$NEWWELL" = "1" ]; then
    if [ ! -f data/spe10/spe10_arrays.npz ]; then
        echo "  [bg] SPE10 arrays missing — fetching + parsing..."
        mkdir -p data/spe10 && \
          curl -sSL -o data/spe10/por_perm_case2a.zip \
          "https://www.spe.org/web/csp/datasets/por_perm_case2a.zip" && \
          (cd data/spe10 && unzip -o por_perm_case2a.zip) && \
          "$PY" scripts/load_spe10.py --spe10-dir data/spe10 \
            --out data/spe10/spe10_arrays.npz \
          > "$OUTPUT_ROOT/_logs/spe10_setup.log" 2>&1
    fi
    echo "  [bg] starting OPM new-well sims..."
    nohup "$PY" scripts/generate_newwell_dataset.py \
        --spe10-arrays data/spe10/spe10_arrays.npz \
        --n-wells "$NEWWELL_N_WELLS" \
        --out-dir "$OUTPUT_ROOT/newwell_data" \
        --n-workers "$NEWWELL_N_WORKERS" \
        > "$OUTPUT_ROOT/_logs/newwell.log" 2>&1 &
    NEWWELL_PID=$!
    echo "  [bg] new-well PID=$NEWWELL_PID, log=$OUTPUT_ROOT/_logs/newwell.log"
fi

# ---- Foreground: launch both GPU streams in parallel ----
echo ""
echo "  [gpu0] launching ($GPU0_DATASETS)..."
run_gpu_stream 0 "$GPU0_DATASETS" &
PID0=$!
echo "  [gpu1] launching ($GPU1_DATASETS)..."
run_gpu_stream 1 "$GPU1_DATASETS" &
PID1=$!
echo "  GPU streams launched: GPU0 PID=$PID0, GPU1 PID=$PID1"
echo "  tail logs with: tail -f $OUTPUT_ROOT/_logs/gpu0.log $OUTPUT_ROOT/_logs/gpu1.log"

# Wait for both GPU streams. Exit code reflects whether either failed.
GPU0_RC=0; GPU1_RC=0
wait "$PID0" || GPU0_RC=$?
echo "  [gpu0] stream exited rc=$GPU0_RC"
wait "$PID1" || GPU1_RC=$?
echo "  [gpu1] stream exited rc=$GPU1_RC"

# Wait for new-well CPU job (much longer than GPU usually)
if [ -n "$NEWWELL_PID" ]; then
    echo ""
    echo "  [bg] waiting for new-well to finish (PID $NEWWELL_PID)..."
    wait "$NEWWELL_PID" && echo "  [bg] new-well OK" || echo "  [bg] new-well exited non-zero (see log)"
fi

# Sweet-spot + export
echo ""
echo "  [aggr] sweet-spot tables:"
for ds in $GPU0_DATASETS $GPU1_DATASETS; do
    if [ -d "$OUTPUT_ROOT/$ds/pijepa_finetune" ]; then
        "$PY" scripts/sweet_spot_table.py \
            --output-root "$OUTPUT_ROOT/$ds" \
            --out "$OUTPUT_ROOT/$ds/figures/sweet_spot" \
          > "$OUTPUT_ROOT/_logs/sweet_spot_${ds}.log" 2>&1 || \
          echo "    [warn] sweet_spot failed for $ds (see log)"
        echo "    $ds → $OUTPUT_ROOT/$ds/figures/sweet_spot.md"
    fi
done

echo "  [export] bundling..."
bash scripts/export_results.sh "$OUTPUT_ROOT" "$OUTPUT_ROOT/_exports" \
  > "$OUTPUT_ROOT/_logs/export.log" 2>&1 || true

echo ""
echo "=============================================================="
echo "  DUAL-GPU PILOT COMPLETE"
echo "    GPU 0 rc:  $GPU0_RC   ($GPU0_DATASETS)"
echo "    GPU 1 rc:  $GPU1_RC   ($GPU1_DATASETS)"
echo "    Logs:      $OUTPUT_ROOT/_logs/"
echo "    Exports:   $OUTPUT_ROOT/_exports/"
echo "=============================================================="
