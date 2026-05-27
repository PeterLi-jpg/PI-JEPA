#!/usr/bin/env bash
# =============================================================================
# PI-JEPA focused paper experiment driver — Option B scope.
#
# Defends ONE thesis:
#   "PI-JEPA, with a true operator-split chain, pretrained on FREE unlabeled
#    parameter fields, beats supervised baselines and from-scratch PI-JEPA
#    when labels are scarce."
#
# Three methods × five N_labeled points × five seeds = 75 fine-tune runs.
# Plus 5 pretrain runs (cached). Plus 3-variant ablation × 5 seeds at one N_l.
# Total: ~110 runs, ~190 A100-hours, ~$285 at $1.50/hr.
#
# Usage:
#   ./scripts/run_focused_paper.sh <output_root> [dataset]
#
# Dataset options (with what's currently downloadable):
#   - darcy_3d_synthetic     (default; works without any external download)
#   - ccsnet                 (uses the CCSNet test_x for pretrain + test_y_SG for finetune)
#
# Environment:
#   N_SEEDS=5      number of seeds (paper standard ≥5)
#   N_LABELED="10 25 50 100 250"   sample-efficiency curve points
#   EPOCHS_PRETRAIN=500
#   EPOCHS_FINETUNE=100
#   EPOCHS_BASELINE=100
#   BASELINES="fno3d ufno3d pino3d deeponet3d pi_deeponet3d"
#   RESIZE_CUBE=64                  # all 5D volumes resized to NxNxN cube
#   FREEZE_EPOCHS_PERCENT=50        # frozen-encoder ablation: pct of epochs to freeze
#
# Resume controls (two levels):
#   RESUME_FROM=N                   start at phase N, skip phases 1..N-1 entirely
#                                     phases: 1 pretrain, 2 finetune, 3 from-scratch,
#                                     3b frozen-encoder, 4 baselines, 5 ablation, 6 figures
#   ONLY_PHASE=N                    run ONLY phase N (mutually exclusive with RESUME_FROM)
#   FORCE_RERUN=1                   don't skip even completed cells (re-do everything)
#
# Cell-level resume is automatic: each (seed, N_l, baseline) cell that already
# wrote a valid result JSON is skipped unless FORCE_RERUN=1.
#
# Examples:
#   RESUME_FROM=4 bash scripts/run_focused_paper.sh outputs/v1     # skip phases 1-3
#   ONLY_PHASE=3b bash scripts/run_focused_paper.sh outputs/v1     # just the freeze ablation
#   FORCE_RERUN=1 bash scripts/run_focused_paper.sh outputs/v1     # redo everything
# =============================================================================

set -euo pipefail

OUTPUT_ROOT="${1:-outputs_focused/v1}"
DATASET="${2:-darcy_3d_synthetic}"
PY="${PYTHON:-.venv/bin/python}"
N_SEEDS="${N_SEEDS:-5}"
SEED_START="${SEED_START:-42}"
N_LABELED="${N_LABELED:-10 25 50 100 250}"
EPOCHS_PRETRAIN="${EPOCHS_PRETRAIN:-500}"
EPOCHS_FINETUNE="${EPOCHS_FINETUNE:-100}"
EPOCHS_BASELINE="${EPOCHS_BASELINE:-100}"
# Reviewer-requested baselines. fno3d_large is the size-matched ~150M FNO
# (qZsm M4); opt in by adding it. Defaults intentionally omit it so the
# grid stays at the cheaper scope unless the user explicitly enables.
BASELINES="${BASELINES:-fno3d ufno3d pino3d deeponet3d pi_deeponet3d}"
# Cubic resize side (Brandon's fourier_encoder_3d is cubic-only). Set 0
# to skip resize (only for synthetic Darcy3D which is already 32^3).
RESIZE_CUBE="${RESIZE_CUBE:-64}"
# Frozen-encoder ablation: freeze for this fraction of total finetune epochs.
# Reviewer YkpY Open Q1.
FREEZE_EPOCHS_PERCENT="${FREEZE_EPOCHS_PERCENT:-50}"
FREEZE_EPOCHS=$(( EPOCHS_FINETUNE * FREEZE_EPOCHS_PERCENT / 100 ))

export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p "$OUTPUT_ROOT"

echo "=============================================================="
echo "PI-JEPA focused paper run"
echo "  output : $OUTPUT_ROOT"
echo "  dataset: $DATASET"
echo "  seeds  : $N_SEEDS (start $SEED_START)"
echo "  N_l    : $N_LABELED"
echo "  pretrain epochs : $EPOCHS_PRETRAIN"
echo "  finetune epochs : $EPOCHS_FINETUNE"
echo "  baseline epochs : $EPOCHS_BASELINE"
echo "  baselines       : $BASELINES"
echo "  resize cube     : ${RESIZE_CUBE}^3"
echo "  freeze epochs   : $FREEZE_EPOCHS  ($FREEZE_EPOCHS_PERCENT%)"
echo "=============================================================="

# Common --resize-cube arg for finetune; empty if RESIZE_CUBE=0
RESIZE_FLAG=""
if [ "$RESIZE_CUBE" -gt 0 ]; then
    RESIZE_FLAG="--resize-cube $RESIZE_CUBE"
fi

# Skip-if-done helper: returns 0 (skip) if RESULT_PATH exists, non-empty,
# and is a valid JSON containing a relative_l2_mean (i.e. a completed
# run). Otherwise returns 1 (run it). Set FORCE_RERUN=1 to bypass.
skip_if_done() {
    local result_path="$1"
    if [ "${FORCE_RERUN:-0}" = "1" ]; then
        return 1
    fi
    [ ! -s "$result_path" ] && return 1
    "$PY" -c "
import json, sys
try:
    d = json.load(open('$result_path'))
    sys.exit(0 if 'relative_l2_mean' in d else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Phase-level resume. Two env vars give you full control:
#   RESUME_FROM=N    Skip phases 1..(N-1) entirely. Default 1 = no skip.
#   ONLY_PHASE=N     Run ONLY phase N, skip everything else. Mutually
#                     exclusive with RESUME_FROM.
# Phases:
#   1   pretrain
#   2   PI-JEPA finetune (sample-efficiency curve)
#   3   PI-JEPA from scratch
#   3b  Frozen-encoder (counts as 3.5 for ordering)
#   4   Supervised baselines
#   5   Ablation grid
#   6   Figures
RESUME_FROM="${RESUME_FROM:-1}"
ONLY_PHASE="${ONLY_PHASE:-}"

# =============================================================================
# Preflight dataset check. Run at startup BEFORE any phase. For the chosen
# DATASET, verifies all required input files exist. If missing, EITHER
# auto-generates them (cheap synthetic data) OR exits with a clear,
# actionable error pointing at the manual fetch the user needs to do.
#
# Set PREFLIGHT_AUTOGEN=0 to disable auto-generation and just fail.
# =============================================================================

PREFLIGHT_AUTOGEN="${PREFLIGHT_AUTOGEN:-1}"

_preflight_fail() {
    echo ""
    echo "================================================================"
    echo "  PREFLIGHT FAILED: dataset=$DATASET"
    echo "================================================================"
    echo "  Missing: $1"
    echo ""
    echo "  Action needed:"
    echo "$2" | sed 's/^/    /'
    echo "================================================================"
    exit 2
}

_check_file() {
    [ -s "$1" ] || return 1
    return 0
}

preflight_dataset() {
    echo ""
    echo "--- Preflight: verifying $DATASET inputs ---"
    case "$DATASET" in
        darcy_3d_synthetic)
            if ! _check_file "$TRAIN_PT" || ! _check_file "$TEST_PT"; then
                # Production-size synthetic Darcy: 1024 train / 256 test
                # at 64^3 (matches other datasets' sample counts and the
                # cube the encoder operates on). Generator runs ~3-5 min
                # on CPU for this size.
                local SYN_NTRAIN="${SYN_NTRAIN:-1024}"
                local SYN_NTEST="${SYN_NTEST:-256}"
                local SYN_RES="${SYN_RES:-64}"
                if [ "$PREFLIGHT_AUTOGEN" = "1" ]; then
                    echo "  $TRAIN_PT or $TEST_PT missing — auto-generating (~3-5 min for ${SYN_NTRAIN}+${SYN_NTEST} samples at ${SYN_RES}^3)..."
                    "$PY" scripts/generate_darcy_data_3d.py \
                        --n-train "$SYN_NTRAIN" --n-test "$SYN_NTEST" --resolution "$SYN_RES" \
                        --out-dir data/darcy_3d --normalize \
                      || _preflight_fail "synthetic darcy 3D generator failed" \
                         "Run manually: $PY scripts/generate_darcy_data_3d.py --n-train $SYN_NTRAIN --n-test $SYN_NTEST --resolution $SYN_RES --out-dir data/darcy_3d --normalize"
                else
                    _preflight_fail "$TRAIN_PT / $TEST_PT" \
                      "Either set PREFLIGHT_AUTOGEN=1 (default) or run: $PY scripts/generate_darcy_data_3d.py --n-train $SYN_NTRAIN --n-test $SYN_NTEST --resolution $SYN_RES --out-dir data/darcy_3d --normalize"
                fi
            fi
            echo "  OK: $TRAIN_PT + $TEST_PT"
            ;;
        ccsnet)
            for path in "$TRAIN_X" "$TRAIN_Y" "$TEST_X" "$TEST_Y"; do
                if ! _check_file "$path"; then
                    _preflight_fail "$path" \
                      "CCSNet must be downloaded from its Google Drive folder.
The required HDF5 files live under data/ccsnet/CCSNet_v1.0/.
- test_x.hdf5            (input fields)
- test_y_SG.hdf5         (saturation target — used for the default pilot)
- test_y_{BPR,BXMF,BYMF,BDENW,BDENG,P_init}.hdf5 (other variants, optional)
On Brev, try: $PY scripts/download_data.py --ccsnet
Or use Brandon's gdown wrapper: bash scripts/download_supervisor.sh ccsnet
If gdown fails, click the README link in https://github.com/gegewen/ccsnet"
                fi
            done
            echo "  OK: CCSNet input + target files present"
            ;;
        fno4co2)
            if ! _check_file "$TRAIN_X" || ! _check_file "$TRAIN_Y"; then
                _preflight_fail "$TRAIN_X / $TRAIN_Y" \
                  "FNO4CO2 must be downloaded from its Dropbox link.
On Brev:
  git clone https://github.com/gegewen/fno4co2 data/fno4co2/repo
  cd data/fno4co2/repo && bash download.sh
The init script downloads dP_test_a.pt + dP_test_u.pt to data/fno4co2/dataset/"
            fi
            echo "  OK: FNO4CO2 dP_test_a.pt + dP_test_u.pt present"
            ;;
        adr_pe_da)
            local train_pt="data/pdebench_adr/adr_train.pt"
            local test_pt="data/pdebench_adr/adr_test.pt"
            local h5="data/pdebench_adr/pe_da_sweep.h5"
            if ! _check_file "$train_pt" || ! _check_file "$test_pt"; then
                if _check_file "$h5"; then
                    echo "  $h5 present but .pt not yet converted — converting..."
                    "$PY" scripts/convert_adr_to_pt.py \
                        --input "$h5" --out-dir data/pdebench_adr \
                        --train-frac 0.8 --normalize \
                      || _preflight_fail "ADR .pt converter failed" \
                         "Run: $PY scripts/convert_adr_to_pt.py --input $h5 --out-dir data/pdebench_adr --train-frac 0.8 --normalize"
                elif [ "$PREFLIGHT_AUTOGEN" = "1" ]; then
                    echo "  $h5 missing — generating sweep (~20 min)..."
                    "$PY" scripts/generate_adr_pe_da_sweep.py \
                        --n-pe 8 --n-da 8 --n-per-cell 16 \
                        --pe-range 0.1 30 --da-range 0.1 30 \
                        --grid 64 --t-final 0.5 --n-t 16 \
                        --output "$h5" \
                      || _preflight_fail "ADR Pe/Da generator failed" \
                         "Manual: $PY scripts/generate_adr_pe_da_sweep.py --output $h5"
                    "$PY" scripts/convert_adr_to_pt.py \
                        --input "$h5" --out-dir data/pdebench_adr \
                        --train-frac 0.8 --normalize \
                      || _preflight_fail "ADR convert failed after generate" ""
                else
                    _preflight_fail "$train_pt / $test_pt + $h5" \
                      "Either run the generator + converter (see scripts/generate_adr_pe_da_sweep.py),
or set PREFLIGHT_AUTOGEN=1 to do it automatically (~20 min)."
                fi
            fi
            echo "  OK: data/pdebench_adr/adr_train.pt + adr_test.pt"
            ;;
        fourier_mionet)
            _preflight_fail "data/fourier_mionet/data/" \
              "Fourier-MIONet requires a manual OneDrive download (no public direct URL).
1. Click: https://yaleedu-my.sharepoint.com/:f:/g/personal/lu_lu_yale_edu/EvWUGDhKje1MsNAtatoxCHsB6qYDyNTpWxDhz_Kf_N7i-Q
2. Sign in with a Yale account or as a guest if the share permits.
3. Download the data folder as a ZIP.
4. Extract to data/fourier_mionet/data/."
            ;;
        newwell)
            if ! command -v flow >/dev/null 2>&1; then
                _preflight_fail "OPM Flow simulator (no 'flow' on PATH)" \
                  "OPM Flow is needed to generate the New-Well dataset.
On Brev: bash scripts/setup_opm_flow.sh
On Mac:  no clean install — use Brev or Docker"
            fi
            if ! _check_file "data/spe10/spe10_arrays.npz"; then
                _preflight_fail "data/spe10/spe10_arrays.npz" \
                  "SPE10 not parsed. Run:
1. mkdir -p data/spe10 && curl -sSL -o data/spe10/por_perm_case2a.zip 'https://www.spe.org/web/csp/datasets/por_perm_case2a.zip'
2. cd data/spe10 && unzip por_perm_case2a.zip
3. $PY scripts/load_spe10.py --spe10-dir data/spe10 --out data/spe10/spe10_arrays.npz"
            fi
            if [ ! -d "data/newwell_spe10" ] || [ -z "$(ls data/newwell_spe10 2>/dev/null)" ]; then
                if [ "$PREFLIGHT_AUTOGEN" = "1" ]; then
                    echo "  data/newwell_spe10 missing — generating (~30 CPU-hours, sit back)..."
                    "$PY" scripts/generate_newwell_dataset.py \
                        --spe10-arrays data/spe10/spe10_arrays.npz \
                        --n-wells 500 --out-dir data/newwell_spe10 --n-workers 4 \
                      || _preflight_fail "OPM Flow new-well generator failed" \
                         "Inspect data/newwell_spe10/run_results.json for per-sim errors."
                else
                    _preflight_fail "data/newwell_spe10/" \
                      "Run: $PY scripts/generate_newwell_dataset.py --spe10-arrays data/spe10/spe10_arrays.npz --n-wells 500 --out-dir data/newwell_spe10"
                fi
            fi
            echo "  OK: SPE10 arrays + new-well sims present"
            ;;
        *)
            _preflight_fail "unknown dataset name" \
              "Supported: darcy_3d_synthetic, ccsnet, fno4co2, adr_pe_da, fourier_mionet, newwell.
Add a new case to preflight_dataset() in scripts/run_focused_paper.sh."
            ;;
    esac
    echo "--- Preflight OK ---"
}
# Encode 3b as 3.5 internally so the numeric comparison works.
phase_enabled() {
    # Args: phase_num (e.g. "1", "3.5")
    local p="$1"
    if [ -n "$ONLY_PHASE" ]; then
        # Normalize "3b" -> "3.5"
        local op="${ONLY_PHASE//3b/3.5}"
        [ "$p" = "$op" ] && return 0 || return 1
    fi
    # RESUME_FROM: convert "3b" -> "3.5" too
    local r="${RESUME_FROM//3b/3.5}"
    # awk does float comparison since bash can't
    awk -v p="$p" -v r="$r" 'BEGIN { exit (p+0 >= r+0) ? 0 : 1 }'
}
echo "Resume: RESUME_FROM=$RESUME_FROM  ONLY_PHASE=${ONLY_PHASE:-<unset>}"

# ---- Resolve dataset paths ----
case "$DATASET" in
    darcy_3d_synthetic)
        PRETRAIN_CONFIG="configs/darcy_3d.yaml"
        TRAIN_PT="data/darcy_3d/darcy3d_train.pt"
        TEST_PT="data/darcy_3d/darcy3d_test.pt"
        FT_DATASET="darcy_3d_pt"
        ;;
    ccsnet)
        PRETRAIN_CONFIG="configs/ccsnet_3d_smoke.yaml"
        # For the focused paper we use CCSNet's test split (38GB on disk) as
        # both pretrain inputs and finetune (input, target) pairs. The train
        # split would be better but isn't downloaded yet.
        TRAIN_X="data/ccsnet/CCSNet_v1.0/test_x.hdf5"
        TRAIN_Y="data/ccsnet/CCSNet_v1.0/test_y_SG.hdf5"
        TEST_X="data/ccsnet/CCSNet_v1.0/test_x.hdf5"
        TEST_Y="data/ccsnet/CCSNet_v1.0/test_y_SG.hdf5"
        FT_DATASET="ccsnet"
        ;;
    fno4co2)
        PRETRAIN_CONFIG="configs/fno4co2_lite_3d_smoke.yaml"
        TRAIN_X="data/fno4co2/dataset/dP_test_a.pt"
        TRAIN_Y="data/fno4co2/dataset/dP_test_u.pt"
        TEST_X="data/fno4co2/dataset/dP_test_a.pt"
        TEST_Y="data/fno4co2/dataset/dP_test_u.pt"
        FT_DATASET="fno4co2"
        ;;
    adr_pe_da)
        PRETRAIN_CONFIG="configs/darcy_3d.yaml"  # same 3D shape (1, 16, 64, 64)
        TRAIN_PT="data/pdebench_adr/adr_train.pt"
        TEST_PT="data/pdebench_adr/adr_test.pt"
        FT_DATASET="darcy_3d_pt"  # uses the same loader path as synthetic Darcy
        ;;
    fourier_mionet|newwell)
        # Both have dataset-specific quirks; preflight prints actionable
        # download / generation instructions and exits if missing.
        PRETRAIN_CONFIG="configs/darcy_3d.yaml"
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        echo "Supported: darcy_3d_synthetic, ccsnet, fno4co2, adr_pe_da, fourier_mionet, newwell"
        exit 2
        ;;
esac

# Run preflight check NOW (before any phase). Fail-fast with actionable
# instructions if a required input is missing AND not auto-generable.
preflight_dataset

# ---- (1) Pretrain PI-JEPA per seed (cached) ----
if phase_enabled "1"; then
echo ""
echo "--- (1) Pretraining PI-JEPA encoder, $N_SEEDS seeds ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    CKPT="$OUTPUT_ROOT/pretrain/seed${s}/pretrain/checkpoint_final.pt"
    if [ -f "$CKPT" ]; then
        echo "  [seed $s] cached: $CKPT"
        continue
    fi
    "$PY" scripts/run_multiseed.py \
        --pretrain-config "$PRETRAIN_CONFIG" \
        --output "$OUTPUT_ROOT/pretrain" \
        --n-seeds 1 --seed-start "$s" \
        --note "focused-paper pretrain seed $s on $DATASET"
done
else echo ""; echo "--- (1) Pretrain SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (2) PI-JEPA fine-tune × N_l × seed ----
if phase_enabled "2"; then
echo ""
echo "--- (2) PI-JEPA fine-tune (sample efficiency curve) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    CKPT="$OUTPUT_ROOT/pretrain/seed${s}/pretrain/checkpoint_final.pt"
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_finetune/seed${s}_n${n_l}"
        mkdir -p "$OUT"
        if skip_if_done "$OUT/pijepa_result.json"; then
            echo "  [seed $s n_l=$n_l] cached: $OUT/pijepa_result.json"
            continue
        fi
        if [ "$DATASET" = "ccsnet" ]; then
            "$PY" scripts/finetune_pijepa.py \
                --pretrain-checkpoint "$CKPT" \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-x "$TRAIN_X" --train-y "$TRAIN_Y" \
                --test-x "$TEST_X"   --test-y "$TEST_Y" \
                $RESIZE_FLAG \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        else
            "$PY" scripts/finetune_pijepa.py \
                --pretrain-checkpoint "$CKPT" \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                $RESIZE_FLAG \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        fi
    done
done

else echo ""; echo "--- (2) Finetune SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (3) PI-JEPA from scratch (NO pretrain) ----
if phase_enabled "3"; then
echo ""
echo "--- (3) PI-JEPA from scratch (architecture-only baseline) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_scratch/seed${s}_n${n_l}"
        mkdir -p "$OUT"
        if skip_if_done "$OUT/pijepa_result.json"; then
            echo "  [scratch seed $s n_l=$n_l] cached"
            continue
        fi
        if [ "$DATASET" = "ccsnet" ]; then
            "$PY" scripts/finetune_pijepa.py \
                --from-scratch \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-x "$TRAIN_X" --train-y "$TRAIN_Y" \
                --test-x "$TEST_X"   --test-y "$TEST_Y" \
                $RESIZE_FLAG \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        else
            "$PY" scripts/finetune_pijepa.py \
                --from-scratch \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                $RESIZE_FLAG \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        fi
    done
done

else echo ""; echo "--- (3) Scratch SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (3b) PI-JEPA frozen-encoder ablation (YkpY Open Q1) ----
if phase_enabled "3.5"; then
echo ""
echo "--- (3b) Frozen-encoder finetune (freeze first $FREEZE_EPOCHS epochs) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    CKPT="$OUTPUT_ROOT/pretrain/seed${s}/pretrain/checkpoint_final.pt"
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_frozen/seed${s}_n${n_l}"
        mkdir -p "$OUT"
        if skip_if_done "$OUT/pijepa_result.json"; then
            echo "  [frozen seed $s n_l=$n_l] cached"
            continue
        fi
        if [ "$DATASET" = "ccsnet" ]; then
            "$PY" scripts/finetune_pijepa.py \
                --pretrain-checkpoint "$CKPT" \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-x "$TRAIN_X" --train-y "$TRAIN_Y" \
                --test-x "$TEST_X"   --test-y "$TEST_Y" \
                $RESIZE_FLAG \
                --freeze-encoder-epochs "$FREEZE_EPOCHS" \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        else
            "$PY" scripts/finetune_pijepa.py \
                --pretrain-checkpoint "$CKPT" \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                $RESIZE_FLAG \
                --freeze-encoder-epochs "$FREEZE_EPOCHS" \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        fi
    done
done

else echo ""; echo "--- (3b) Frozen-encoder SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (4) Supervised baselines (loops over $BASELINES) ----
if phase_enabled "4"; then
echo ""
echo "--- (4) Supervised baselines: $BASELINES ---"
for baseline in $BASELINES; do
    for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
        for n_l in $N_LABELED; do
            OUT="$OUTPUT_ROOT/baselines/${baseline}/seed${s}_n${n_l}"
            mkdir -p "$OUT"
            if skip_if_done "$OUT/baseline_result.json"; then
                echo "  [$baseline seed $s n_l=$n_l] cached"
                continue
            fi
            if [ "$DATASET" = "ccsnet" ]; then
                "$PY" scripts/train_baseline.py \
                    --baseline "$baseline" --dataset ccsnet \
                    --train-x "$TRAIN_X" --train-y "$TRAIN_Y" \
                    --test-x "$TEST_X"   --test-y "$TEST_Y" \
                    --resize-cube "$RESIZE_CUBE" \
                    --n-labeled "$n_l" --epochs "$EPOCHS_BASELINE" --seed "$s" \
                    --hidden-channels 32 --n-blocks 4 --modes 8 8 8 \
                    --output "$OUT" || true
            else
                "$PY" scripts/train_baseline.py \
                    --baseline "$baseline" --dataset darcy_3d_pt \
                    --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                    --resize-cube "$RESIZE_CUBE" \
                    --n-labeled "$n_l" --epochs "$EPOCHS_BASELINE" --seed "$s" \
                    --hidden-channels 32 --n-blocks 4 --modes 8 8 8 \
                    --output "$OUT" || true
            fi
        done
    done
done

else echo ""; echo "--- (4) Baselines SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (5) Focused 3-variant ablation at N_l = 100 (or middle of sweep) ----
if phase_enabled "5"; then
N_L_ABL="100"
# Pick middle of the N_l sweep if 100 not present
case " $N_LABELED " in
    *" 100 "*) ;;
    *)
        N_L_ABL=$(echo $N_LABELED | awk '{print $((NF+1)/2)}')
        ;;
esac
echo ""
echo "--- (5) Ablation (full grid: 7 variants × $N_SEEDS seeds at N_l=$N_L_ABL) ---"
# Each variant changes pretraining (own pretrain + finetune); reviewer-traceable.
# - full: reference
# - no_chain: num_predictors=1 (operator-split contribution — M2)
# - no_physics: pretrain without physics residual (M1, W1)
# - fd_physics vs spectral_physics: which residual implementation helps (Q2)
# - no_vicreg: VICReg contribution (M2)
# - no_per_stage_decoders: per-stage decoder contribution
ABLATION_VARIANTS="${ABLATION_VARIANTS:-full no_chain no_physics fd_physics spectral_physics no_vicreg no_per_stage_decoders}"
"$PY" scripts/run_ablations.py \
    --base-config "$PRETRAIN_CONFIG" \
    --output "$OUTPUT_ROOT/ablation" \
    --n-seeds "$N_SEEDS" --seed-start "$SEED_START" \
    --variants $ABLATION_VARIANTS || true

else echo ""; echo "--- (5) Ablation SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

# ---- (6) Aggregate into paper-ready tables ----
if phase_enabled "6"; then
echo ""
echo "--- (6) Generating paper-ready figures + tables ---"
mkdir -p "$OUTPUT_ROOT/figures"
"$PY" scripts/make_paper_figures.py data_eff \
    --input-dir "$OUTPUT_ROOT" \
    --metric relative_l2_mean \
    --out "$OUTPUT_ROOT/figures/sample_efficiency.png" || true
"$PY" scripts/make_paper_figures.py ablation \
    --input-json "$OUTPUT_ROOT/ablation/ablation_table.json" \
    --metric jepa \
    --out "$OUTPUT_ROOT/figures/ablation.png" || true

else echo ""; echo "--- (6) Figures SKIPPED (RESUME_FROM=$RESUME_FROM ONLY_PHASE=${ONLY_PHASE:-}) ---"
fi

echo ""
echo "=============================================================="
echo "  Focused paper run complete."
echo "  Outputs:        $OUTPUT_ROOT"
echo "  Sample-eff fig: $OUTPUT_ROOT/figures/sample_efficiency.png"
echo "  Ablation fig:   $OUTPUT_ROOT/figures/ablation.png"
echo "=============================================================="
