#!/bin/bash
#SBATCH --job-name=v16-Sub
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=0
#SBATCH --exclude=g001,g002,g003,g004,g005,g006,g007,g008,g009,g010
#SBATCH --output=/net/scratch/caom/repo_project/logs/v16_SubComponent_only_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v16_SubComponent_only_%j.err

# ==============================================
# Per-objective: SubComponent-only active-GRPO (v16 full)
# Train + Eval in one job
# ==============================================

set -euo pipefail

SCRATCH=/net/scratch/caom/repo_project
REPO_DIR=/home/caom/rl-agent/RePO
ADA_DIR=/home/caom/rl-agent/agent_drug_discovery/adaptive_repo
ENV_DIR=${SCRATCH}/envs/repo_env
OUTPUT_DIR=${SCRATCH}/outputs/ada_repo_dsi_v16_SubComponent_only
PRED_DIR=${SCRATCH}/outputs/eval_v16_SubComponent_only/predictions/

export HF_HOME=${SCRATCH}/cache/huggingface
export TRANSFORMERS_CACHE=${SCRATCH}/cache/huggingface
export HF_DATASETS_CACHE=${SCRATCH}/cache/huggingface/datasets
export TORCH_HOME=${SCRATCH}/cache/torch
export TRITON_CACHE_DIR=${SCRATCH}/cache/triton
export VLLM_CACHE_DIR=${SCRATCH}/cache/vllm
export TMPDIR=${SCRATCH}/cache/tmp
export XDG_CACHE_HOME=${SCRATCH}/cache
export DEEPSPEED_CACHE_DIR=${SCRATCH}/cache/deepspeed
export WANDB_DIR=${SCRATCH}/logs/wandb
export WANDB_CACHE_DIR=${SCRATCH}/cache/wandb
export PIP_CACHE_DIR=${SCRATCH}/cache/pip
export WANDB_MODE=offline

export CUDA_HOME=/usr/local/cuda-12
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export DS_SKIP_CUDA_CHECK=1

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${ADA_DIR}:${PYTHONPATH:-}"
export NCCL_DEBUG=WARN
export MASTER_ADDR=localhost
export MASTER_PORT=$((29500 + RANDOM % 100))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=============================================="
echo "Per-objective: SubComponent-only active-GRPO (v16)"
echo "Node: $(hostname), Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

cd ${REPO_DIR}

# === TRAIN ===
ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${REPO_DIR}/recipes/zero3_dsi.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 2 \
    ${ADA_DIR}/ada_repo.py \
    --config ${ADA_DIR}/configs/DSI_v16_SubComponent_only.yaml \
    --output_dir ${OUTPUT_DIR} \
    --variant default \
    --run_name "adarepo_v16_SubComponent_only"

echo "=== Training complete at $(date) ==="

# === EVAL ===
MODEL_PATH=${OUTPUT_DIR}
if [ ! -f "${MODEL_PATH}/config.json" ]; then
    CKPT=$(ls -d ${OUTPUT_DIR}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
    if [ -n "$CKPT" ]; then
        MODEL_PATH=${CKPT}
    else
        echo "ERROR: No model found in ${OUTPUT_DIR}"
        exit 1
    fi
fi

echo "--- Evaluating SubComponent with model ${MODEL_PATH} ---"
python3 ${REPO_DIR}/generate_predictions.py \
    --model_path ${MODEL_PATH} \
    --benchmark open_generation \
    --task MolEdit \
    --subtask SubComponent \
    --output_dir ${PRED_DIR} \
    --gpu_memory_utilization 0.85 \
    --lang en

python3 ${REPO_DIR}/evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmark "open_generation" \
    --task "MolEdit" \
    --subtask "SubComponent" \
    --output_dir "${PRED_DIR}"

echo "=== All done at $(date) ==="
