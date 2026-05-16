#!/bin/bash
#SBATCH --job-name=v15.1-eval
#SBATCH --partition=general
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=0
#SBATCH --output=/net/scratch/caom/repo_project/logs/v15_1_resume_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v15_1_resume_%j.err

# Eval v15.1 checkpoint-700 (training reached 737/748 = epoch 3.92)
# Generate predictions on MolOpt benchmark + evaluate

set -euo pipefail

SCRATCH=/net/scratch/caom/repo_project
REPO_DIR=/home/caom/rl-agent/RePO
ADA_DIR=/home/caom/rl-agent/agent_drug_discovery/adaptive_repo
ENV_DIR=${SCRATCH}/envs/repo_env

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

CKPT_DIR=${SCRATCH}/outputs/ada_repo_dsi_v15_1_boosted
PRED_DIR=${SCRATCH}/outputs/eval_v15_1_boosted/predictions/

echo "=============================================="
echo "v15.1 Eval — checkpoint-700 (epoch 3.73)"
echo "Node: $(hostname), Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=============================================="

cd ${REPO_DIR}

# Use checkpoint-700 directly (optimizer states cleaned, cannot resume,
# but model weights are complete and training was 98.6% done)
FINAL_MODEL=${CKPT_DIR}/checkpoint-700
echo "Model: ${FINAL_MODEL}"

# ==================================================================
# 2. Generate predictions on MolOpt benchmark (LogP, MR, QED)
# ==================================================================
echo ""
echo "=== PHASE 2: Generating predictions ==="
echo "Model: ${FINAL_MODEL}"
echo "Start: $(date)"

for SUBTASK in LogP MR QED; do
    echo "--- Generating for ${SUBTASK} ---"
    python3 ${REPO_DIR}/generate_predictions.py \
        --model_path ${FINAL_MODEL} \
        --benchmark open_generation \
        --task MolOpt \
        --subtask ${SUBTASK} \
        --output_dir ${PRED_DIR} \
        --gpu_memory_utilization 0.85 \
        --lang en
    echo "${SUBTASK} predictions done at $(date)"
done

echo "=== All predictions generated at $(date) ==="

# ==================================================================
# 3. Evaluate predictions
# ==================================================================
echo ""
echo "=== PHASE 3: Evaluating predictions ==="
echo "Start: $(date)"

# Extract model name (same logic as generate_predictions.py)
MODEL_NAME=$(basename ${FINAL_MODEL} | sed 's/--/_/g; s/\//_/g')

for SUBTASK in LogP MR QED; do
    echo "--- Evaluating ${SUBTASK} ---"
    python3 ${REPO_DIR}/evaluate.py \
        --model_path "${FINAL_MODEL}" \
        --benchmark "open_generation" \
        --task "MolOpt" \
        --subtask "${SUBTASK}" \
        --output_dir "${PRED_DIR}"
    echo "${SUBTASK} evaluation done at $(date)"
done

echo ""
echo "=============================================="
echo "All done at $(date)"
echo "Predictions: ${PRED_DIR}"
echo "=============================================="
