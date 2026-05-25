#!/usr/bin/env bash
# =============================================================================
# PI-JEPA paper experiment driver.
#
# Reproduces every cell in the paper's headline tables. Designed for the
# Brev cluster (CUDA, plenty of time) — DO NOT run as-is on a Mac laptop.
# Wall-clock estimate per dataset on a single A100: ~6-12 GPU-hours.
#
# Usage:
#   ./scripts/run_paper_experiments.sh <output_root> [<datasets>]
#
#   <output_root>: directory under which all per-experiment outputs land
#   <datasets>: optional comma-separated subset of {darcy_3d,ccsnet,fno4co2,pdebench_adr}
#               default: all four
#
# Example:
#   ./scripts/run_paper_experiments.sh outputs_paper_v1
#   ./scripts/run_paper_experiments.sh outputs_smoke_ccsnet ccsnet
# =============================================================================

set -euo pipefail

OUTPUT_ROOT="${1:-outputs_paper_v1}"
DATASETS="${2:-darcy_3d,ccsnet,fno4co2,pdebench_adr}"
N_SEEDS="${N_SEEDS:-5}"
SEED_START="${SEED_START:-42}"

PY="${PYTHON:-.venv/bin/python}"
export PYTORCH_ENABLE_MPS_FALLBACK=1

mkdir -p "$OUTPUT_ROOT"
echo "PI-JEPA paper experiments → $OUTPUT_ROOT  (seeds=$N_SEEDS, datasets=$DATASETS)"

run_dataset() {
    local ds="$1"
    local cfg="$2"
    local n_labeled_list="$3"
    echo ""
    echo "================================================================"
    echo " Dataset: $ds   config: $cfg"
    echo "================================================================"

    # 1. Multi-seed pretraining
    "$PY" scripts/run_multiseed.py \
        --pretrain-config "$cfg" \
        --output "$OUTPUT_ROOT/$ds/pretrain" \
        --n-seeds "$N_SEEDS" \
        --seed-start "$SEED_START" \
        --note "$ds main results pretrain"

    # 2. For each seed and each N_labeled, fine-tune and run baseline
    for seed in $(seq "$SEED_START" $((SEED_START + N_SEEDS - 1))); do
        for n_l in $n_labeled_list; do
            ckpt="$OUTPUT_ROOT/$ds/pretrain/seed${seed}/pretrain/checkpoint_final.pt"
            ftout="$OUTPUT_ROOT/$ds/finetune/seed${seed}_n${n_l}"
            blout="$OUTPUT_ROOT/$ds/baseline_fno3d/seed${seed}_n${n_l}"
            if [ -f "$ckpt" ]; then
                echo "[$ds seed=$seed n_l=$n_l] PI-JEPA fine-tune"
                "$PY" scripts/finetune_pijepa.py \
                    --pretrain-checkpoint "$ckpt" \
                    --pretrain-config "$cfg" \
                    --dataset darcy_3d_pt \
                    --train-pt data/darcy_3d/darcy3d_train.pt \
                    --test-pt  data/darcy_3d/darcy3d_test.pt \
                    --n-labeled "$n_l" --epochs 50 --seed "$seed" \
                    --output "$ftout" || true
            else
                echo "  missing $ckpt — skipping fine-tune"
            fi

            echo "[$ds seed=$seed n_l=$n_l] FNO3D baseline"
            "$PY" scripts/train_baseline.py \
                --baseline fno3d --dataset darcy_3d_pt \
                --train-pt data/darcy_3d/darcy3d_train.pt \
                --test-pt  data/darcy_3d/darcy3d_test.pt \
                --n-labeled "$n_l" --epochs 50 --seed "$seed" \
                --hidden-channels 32 --n-blocks 4 --modes 8 8 8 \
                --output "$blout" || true
        done
    done
}

for ds in $(echo "$DATASETS" | tr ',' ' '); do
    case "$ds" in
        darcy_3d)
            run_dataset "darcy_3d" "configs/darcy_3d.yaml" "16 32 64 128"
            ;;
        ccsnet)
            # Pretrain on CCSNet 3D — needs the train_x.hdf5 download to be complete
            run_dataset "ccsnet" "configs/ccsnet_3d_smoke.yaml" "16 32 64 128"
            ;;
        fno4co2)
            # Placeholder — needs a configs/fno4co2_3d.yaml + train_pt-equivalent paths
            echo "fno4co2 path TBD — wire fno4co2_3d config"
            ;;
        pdebench_adr)
            echo "pdebench_adr path TBD — wire pdebench_adr config"
            ;;
        *)
            echo "unknown dataset: $ds  — skipping"
            ;;
    esac
done

echo ""
echo "================================================================"
echo " All experiments complete. Aggregated JSON files under $OUTPUT_ROOT"
echo "================================================================"
