#!/bin/bash
#SBATCH --job-name=v15.1
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=0
#SBATCH --output=/net/scratch/caom/repo_project/logs/v15_1_boosted_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v15_1_boosted_%j.err

# ==============================================
# AdaRePO v15.1 Boosted — sigmoid_gap (beta_max=1.5, beta_min=0.3)
# Joint LogP+MR+QED, matched to existing DSI RePO ckpt-748
# Hardware: G=4, 2 train + 1 vLLM A100 80GB
# ==============================================

set -euo pipefail

SCRATCH=/net/scratch/caom/repo_project
REPO_DIR=/home/caom/rl-agent/RePO
ADA_DIR=/home/caom/rl-agent/agent_drug_discovery/adaptive_repo
ENV_DIR=${SCRATCH}/envs/repo_env

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

# Fix DeepSpeed CUDA mismatch: system default is 13.1, torch needs 12.1
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
echo "AdaRePO v15.1 Boosted — Joint LogP+MR+QED"
echo "Node: $(hostname), Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "=============================================="

cd ${REPO_DIR}

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${REPO_DIR}/recipes/zero3_dsi.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 2 \
    ${ADA_DIR}/ada_repo.py \
    --config ${ADA_DIR}/configs/DSI_v15_1_LogP_boosted.yaml \
    --variant default \
    --run_name "adarepo_v15_1_boosted_dsi"

echo "=== AdaRePO v15.1 Boosted DONE at $(date) ==="
