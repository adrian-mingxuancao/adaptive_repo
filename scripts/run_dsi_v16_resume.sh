#!/bin/bash
#SBATCH --job-name=v16-res
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=0
#SBATCH --exclude=g001,g002,g003,g004,g005,g006,g007,g008,g009,g010
#SBATCH --output=/net/scratch/caom/repo_project/logs/v16_resume_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v16_resume_%j.err

# ==============================================
# AdaRePO v16 — Resume from checkpoint-650
# ~98 steps remaining at ~65s/step ≈ 1.8h
# ==============================================

set -euo pipefail

SCRATCH=/net/scratch/caom/repo_project
REPO_DIR=/home/caom/rl-agent/RePO
ADA_DIR=/home/caom/rl-agent/agent_drug_discovery/adaptive_repo
ENV_DIR=${SCRATCH}/envs/repo_env
OUTPUT_DIR=${SCRATCH}/outputs/ada_repo_dsi_v16

# Cache directories
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

# Fix DeepSpeed CUDA mismatch
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
echo "AdaRePO v16 — Resume from checkpoint-650"
echo "Node: $(hostname), Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

cd ${REPO_DIR}

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${REPO_DIR}/recipes/zero3_dsi.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 2 \
    ${ADA_DIR}/ada_repo.py \
    --config ${ADA_DIR}/configs/DSI_v16_adaptive_curriculum.yaml \
    --output_dir ${OUTPUT_DIR} \
    --variant default \
    --run_name "adarepo_v16_resume"

echo "=== Training complete at $(date) ==="
