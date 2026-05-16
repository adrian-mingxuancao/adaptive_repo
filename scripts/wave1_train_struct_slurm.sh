#!/bin/bash
#SBATCH --job-name=w1_struct
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclude=g003,g004,g005,g006,g007,g008,g009,g010
#SBATCH --output=logs/w1_struct_%j.log
#SBATCH --error=logs/w1_struct_%j.log

# ==============================================
# Wave 1 Part A — Structural training
#   1. RePO Structure (Add/Del/Sub) — 4 epochs
#   2. AdaRePO Structure (Add/Del/Sub) — 4 epochs
# ==============================================

set -euo pipefail

PROJECT_DIR=/net/scratch2/qinanh/agent_drug_discovery
REPO_DIR=${PROJECT_DIR}/RePO
ADA_DIR=${PROJECT_DIR}/adaptive_repo
ENV_DIR=/net/scratch2/qinanh/miniconda3/envs/adarepo

mkdir -p ${ADA_DIR}/logs

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${ADA_DIR}:${PYTHONPATH:-}"
export HF_HOME=/net/scratch2/qinanh/hf_cache
export TRANSFORMERS_CACHE=/net/scratch2/qinanh/hf_cache
export HF_DATASETS_CACHE=/net/scratch2/qinanh/hf_cache/datasets
export TRITON_CACHE_DIR=/net/scratch2/qinanh/.triton
export DS_SKIP_CUDA_CHECK=1
export WANDB_MODE=offline
export WANDB_DIR=${ADA_DIR}/logs
export NCCL_DEBUG=WARN
export MASTER_ADDR=localhost
export MASTER_PORT=29502
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${REPO_DIR}

echo "=============================================="
echo "Wave 1 Part A — Structural training"
echo "Node: $(hostname), Date: $(date)"
echo "=============================================="

nvidia-smi

# ------------------------------------------------------------------
# 1. RePO Structure (4 epochs)
# ------------------------------------------------------------------
echo ""
echo "=== TRAINING: RePO Structure (Add/Del/Sub) ==="
echo "  Start: $(date)"

ACCELERATE_LOG_LEVEL=info \
  accelerate launch \
    --config_file recipes/zero3.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 3 \
    ${REPO_DIR}/src/x_r1/repo.py \
    --config ${REPO_DIR}/recipes/local_3B_Structure.yaml \
    --variant default \
    --run_name "w1_repo_Structure"

echo "RePO Structure DONE at $(date)"

# ------------------------------------------------------------------
# 2. AdaRePO Structure (4 epochs)
# ------------------------------------------------------------------
echo ""
echo "=== TRAINING: AdaRePO Structure (Add/Del/Sub) ==="
echo "  Start: $(date)"

ACCELERATE_LOG_LEVEL=info \
  accelerate launch \
    --config_file recipes/zero3.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 3 \
    ${ADA_DIR}/ada_repo.py \
    --config ${ADA_DIR}/configs/ada_repo_3B_Structure.yaml \
    --variant default \
    --run_name "w1_ada_Structure"

echo "AdaRePO Structure DONE at $(date)"

echo ""
echo "=============================================="
echo "Wave 1 Part A complete at $(date)"
echo "=============================================="
