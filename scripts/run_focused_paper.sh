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
    *)
        echo "Unknown dataset: $DATASET"
        exit 2
        ;;
esac

# ---- (1) Pretrain PI-JEPA per seed (cached) ----
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

# ---- (2) PI-JEPA fine-tune × N_l × seed ----
echo ""
echo "--- (2) PI-JEPA fine-tune (sample efficiency curve) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    CKPT="$OUTPUT_ROOT/pretrain/seed${s}/pretrain/checkpoint_final.pt"
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_finetune/seed${s}_n${n_l}"
        mkdir -p "$OUT"
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

# ---- (3) PI-JEPA from scratch (NO pretrain) ----
echo ""
echo "--- (3) PI-JEPA from scratch (architecture-only baseline) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_scratch/seed${s}_n${n_l}"
        mkdir -p "$OUT"
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

# ---- (3b) PI-JEPA frozen-encoder ablation (YkpY Open Q1) ----
echo ""
echo "--- (3b) Frozen-encoder finetune (freeze first $FREEZE_EPOCHS epochs) ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    CKPT="$OUTPUT_ROOT/pretrain/seed${s}/pretrain/checkpoint_final.pt"
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/pijepa_frozen/seed${s}_n${n_l}"
        mkdir -p "$OUT"
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

# ---- (4) Supervised baselines (loops over $BASELINES) ----
echo ""
echo "--- (4) Supervised baselines: $BASELINES ---"
for baseline in $BASELINES; do
    for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
        for n_l in $N_LABELED; do
            OUT="$OUTPUT_ROOT/baselines/${baseline}/seed${s}_n${n_l}"
            mkdir -p "$OUT"
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

# ---- (5) Focused 3-variant ablation at N_l = 100 (or middle of sweep) ----
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

# ---- (6) Aggregate into paper-ready tables ----
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

echo ""
echo "=============================================================="
echo "  Focused paper run complete."
echo "  Outputs:        $OUTPUT_ROOT"
echo "  Sample-eff fig: $OUTPUT_ROOT/figures/sample_efficiency.png"
echo "  Ablation fig:   $OUTPUT_ROOT/figures/ablation.png"
echo "=============================================================="
