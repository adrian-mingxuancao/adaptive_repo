#!/bin/bash
#SBATCH --job-name=zeroshot-eval
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=0
#SBATCH --exclude=g001,g002,g003,g004,g005,g006,g007,g008,g009,g010
#SBATCH --output=/net/scratch/caom/repo_project/logs/zeroshot_eval_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/zeroshot_eval_%j.err

# ==============================================
# Eval Qwen2.5-3B-Instruct ZERO-SHOT (no training)
# Generate predictions + evaluate on MolOpt benchmark
# ==============================================

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use the base Qwen2.5-3B-Instruct model (no fine-tuning)
BASE_MODEL=${SCRATCH}/cache/huggingface/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
PRED_DIR=${SCRATCH}/outputs/eval_zeroshot/predictions/

echo "=============================================="
echo "Zero-shot Eval — Qwen2.5-3B-Instruct (no training)"
echo "Node: $(hostname), Date: $(date)"
echo "Model: ${BASE_MODEL}"
echo "=============================================="

cd ${REPO_DIR}

# Generate predictions
for SUBTASK in LogP MR QED; do
    echo "--- Generating for ${SUBTASK} ---"
    python3 ${REPO_DIR}/generate_predictions.py \
        --model_path ${BASE_MODEL} \
        --benchmark open_generation \
        --task MolOpt \
        --subtask ${SUBTASK} \
        --output_dir ${PRED_DIR} \
        --gpu_memory_utilization 0.85 \
        --lang en
    echo "${SUBTASK} predictions done at $(date)"
done

echo "=== All predictions generated at $(date) ==="

# Evaluate
for SUBTASK in LogP MR QED; do
    echo "--- Evaluating ${SUBTASK} ---"
    python3 ${REPO_DIR}/evaluate.py \
        --model_path "${BASE_MODEL}" \
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
