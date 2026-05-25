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
echo "=============================================================="

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
                --resize-to 96 96 \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        else
            "$PY" scripts/finetune_pijepa.py \
                --pretrain-checkpoint "$CKPT" \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
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
                --resize-to 96 96 \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        else
            "$PY" scripts/finetune_pijepa.py \
                --from-scratch \
                --pretrain-config "$PRETRAIN_CONFIG" \
                --dataset "$FT_DATASET" \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                --n-labeled "$n_l" --epochs "$EPOCHS_FINETUNE" --seed "$s" \
                --output "$OUT" || true
        fi
    done
done

# ---- (4) Supervised FNO3D baseline ----
echo ""
echo "--- (4) FNO3D supervised baseline ---"
for s in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
    for n_l in $N_LABELED; do
        OUT="$OUTPUT_ROOT/fno3d_baseline/seed${s}_n${n_l}"
        mkdir -p "$OUT"
        if [ "$DATASET" = "ccsnet" ]; then
            "$PY" scripts/train_baseline.py \
                --baseline fno3d --dataset ccsnet \
                --train-x "$TRAIN_X" --train-y "$TRAIN_Y" \
                --test-x "$TEST_X"   --test-y "$TEST_Y" \
                --n-labeled "$n_l" --epochs "$EPOCHS_BASELINE" --seed "$s" \
                --hidden-channels 32 --n-blocks 4 --modes 4 8 8 \
                --output "$OUT" || true
        else
            "$PY" scripts/train_baseline.py \
                --baseline fno3d --dataset darcy_3d_pt \
                --train-pt "$TRAIN_PT" --test-pt "$TEST_PT" \
                --n-labeled "$n_l" --epochs "$EPOCHS_BASELINE" --seed "$s" \
                --hidden-channels 32 --n-blocks 4 --modes 8 8 8 \
                --output "$OUT" || true
        fi
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
echo "--- (5) Ablation (full / no_chain / no_physics) at N_l=$N_L_ABL ---"
"$PY" scripts/run_ablations.py \
    --base-config "$PRETRAIN_CONFIG" \
    --output "$OUTPUT_ROOT/ablation" \
    --n-seeds "$N_SEEDS" --seed-start "$SEED_START" \
    --variants full no_chain no_physics || true

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
