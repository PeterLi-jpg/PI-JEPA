#!/usr/bin/env bash
# =============================================================================
# Parallel pilot launcher for the resubmission grid.
#
# Layout:
#   - GPU process: runs scripts/run_focused_paper.sh sequentially on each
#     dataset in DATASETS, occupying CUDA + 1-2 vCPUs.
#   - CPU process (optional): runs scripts/generate_newwell_dataset.py
#     in parallel on the OTHER CPU cores. OPM Flow is CPU-only so it
#     doesn't fight the GPU training.
#   - When both finish, runs scripts/sweet_spot_table.py per-dataset
#     and emits one cross-dataset summary.
#
# Why this script exists: previously the user had to manually open two
# tmux panes and run the GPU + CPU work separately. This collapses that
# into one invocation that handles both, plus logs both streams so a
# crash in either is recoverable.
#
# Usage:
#   bash scripts/run_pilot_parallel.sh OUTPUT_ROOT [DATASETS] [NEWWELL]
#
# Env vars:
#   DATASETS="darcy_3d_synthetic ccsnet adr_pe_da fno4co2"   # default
#   NEWWELL=1                  # also start the New-Well OPM sims in parallel
#                              # (requires flow on PATH; will skip with a
#                              #  warning if OPM isn't installed)
#   NEWWELL_N_WELLS=500        # number of new-well sims
#   NEWWELL_N_WORKERS=6        # CPU workers for OPM Flow
#   N_SEEDS, EPOCHS_*, BASELINES, ABLATION_VARIANTS, RESIZE_CUBE etc.
#     are passed straight through to run_focused_paper.sh — see that
#     script's docstring.
# =============================================================================

set -euo pipefail

OUTPUT_ROOT="${1:-outputs_pilot}"
DATASETS="${DATASETS:-darcy_3d_synthetic adr_pe_da}"
NEWWELL="${NEWWELL:-0}"
NEWWELL_N_WELLS="${NEWWELL_N_WELLS:-500}"
NEWWELL_N_WORKERS="${NEWWELL_N_WORKERS:-6}"
PY="${PYTHON:-.venv/bin/python}"

mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/_logs"

echo "=============================================================="
echo "  PARALLEL PILOT LAUNCHER"
echo "  output root:        $OUTPUT_ROOT"
echo "  GPU datasets:       $DATASETS"
echo "  New-Well CPU job:   $([ "$NEWWELL" = "1" ] && echo "ENABLED ($NEWWELL_N_WELLS wells, $NEWWELL_N_WORKERS workers)" || echo "disabled")"
echo "=============================================================="

# ----- (A) CPU new-well sims in the background -----
NEWWELL_PID=""
if [ "$NEWWELL" = "1" ]; then
    if ! command -v flow >/dev/null 2>&1; then
        echo "  [warn] flow not on PATH — skipping New-Well sims."
        echo "         Run bash scripts/setup_opm_flow.sh on this Brev instance first."
    else
        echo "  [bg] starting OPM Flow new-well sims..."
        nohup "$PY" scripts/generate_newwell_dataset.py \
            --spe10-arrays data/spe10/spe10_arrays.npz \
            --n-wells "$NEWWELL_N_WELLS" \
            --out-dir "$OUTPUT_ROOT/newwell_data" \
            --n-workers "$NEWWELL_N_WORKERS" \
            > "$OUTPUT_ROOT/_logs/newwell.log" 2>&1 &
        NEWWELL_PID=$!
        echo "  [bg] new-well PID=$NEWWELL_PID, log=$OUTPUT_ROOT/_logs/newwell.log"
    fi
fi

# ----- (B) Sequential GPU pilots, one per dataset -----
GPU_FAILED=()
for ds in $DATASETS; do
    log="$OUTPUT_ROOT/_logs/gpu_${ds}.log"
    echo ""
    echo "  [gpu] running pilot on $ds (log: $log)"
    if PYTHON="$PY" bash scripts/run_focused_paper.sh "$OUTPUT_ROOT/$ds" "$ds" \
        2>&1 | tee "$log"; then
        echo "  [gpu] $ds: DONE"
    else
        rc=$?
        echo "  [gpu] $ds: FAILED (exit $rc) — see $log"
        GPU_FAILED+=("$ds")
    fi
done

# ----- (C) Wait for the CPU new-well job to finish -----
if [ -n "$NEWWELL_PID" ]; then
    echo ""
    echo "  [bg] waiting for new-well sims (PID $NEWWELL_PID)..."
    wait "$NEWWELL_PID" || echo "  [bg] new-well finished with non-zero exit (see log)"
    echo "  [bg] new-well done. data: $OUTPUT_ROOT/newwell_data"
fi

# ----- (D) Cross-dataset sweet-spot summaries -----
echo ""
echo "  [aggr] sweet-spot tables per dataset:"
for ds in $DATASETS; do
    if [ -d "$OUTPUT_ROOT/$ds/pijepa_finetune" ]; then
        "$PY" scripts/sweet_spot_table.py \
            --output-root "$OUTPUT_ROOT/$ds" \
            --out "$OUTPUT_ROOT/$ds/figures/sweet_spot" \
            > "$OUTPUT_ROOT/_logs/sweet_spot_${ds}.log" 2>&1 \
          || echo "    [warn] sweet_spot failed for $ds"
        echo "    $ds: $OUTPUT_ROOT/$ds/figures/sweet_spot.md"
    fi
done

# ----- (E) Bundle everything for export -----
echo ""
echo "  [export] bundling results..."
bash scripts/export_results.sh "$OUTPUT_ROOT" "$OUTPUT_ROOT/_exports" \
  > "$OUTPUT_ROOT/_logs/export.log" 2>&1 || true
echo "  [export] $OUTPUT_ROOT/_exports/*.tar.gz"

# ----- (F) Final summary -----
echo ""
echo "=============================================================="
if [ ${#GPU_FAILED[@]} -eq 0 ]; then
    echo "  PILOT COMPLETE — all GPU datasets succeeded"
else
    echo "  PILOT COMPLETE WITH FAILURES on: ${GPU_FAILED[*]}"
    echo "  See $OUTPUT_ROOT/_logs/ for details."
fi
echo "  Outputs: $OUTPUT_ROOT"
echo "  Logs:    $OUTPUT_ROOT/_logs"
echo "  Exports: $OUTPUT_ROOT/_exports"
echo "=============================================================="
