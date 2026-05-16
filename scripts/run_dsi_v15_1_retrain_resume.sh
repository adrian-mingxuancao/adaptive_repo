#!/bin/bash
#SBATCH --job-name=v15.1-res
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:3
#SBATCH --mem=0
#SBATCH --output=/net/scratch/caom/repo_project/logs/v15_1_retrain_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v15_1_retrain_%j.err

# ==============================================
# Resume v15.1 retrain from checkpoint-150
# Remaining: 598 steps (~11h at 67s/step)
# Fixed callback: pre-cleans optimizer before save
# ==============================================

set -euo pipefail

SCRATCH=/net/scratch/caom/repo_project
REPO_DIR=/home/caom/rl-agent/RePO
ADA_DIR=/home/caom/rl-agent/agent_drug_discovery/adaptive_repo
ENV_DIR=${SCRATCH}/envs/repo_env
OUTPUT_DIR=${SCRATCH}/outputs/ada_repo_dsi_v15_1_retrain

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

CKPT=${OUTPUT_DIR}/checkpoint-150

echo "=============================================="
echo "v15.1 Resume from checkpoint-150"
echo "Node: $(hostname), Date: $(date)"
echo "GPUs: $(nvidia-smi -L 2>/dev/null | wc -l)"
echo "Checkpoint: ${CKPT}"
echo "=============================================="

# Sanity check
if [ ! -f "${CKPT}/trainer_state.json" ]; then
    echo "ERROR: trainer_state.json not found"; exit 1
fi
if [ ! -d "${CKPT}/global_step150" ]; then
    echo "ERROR: optimizer states not found"; exit 1
fi
echo "Checkpoint sanity: OK"

cd ${REPO_DIR}

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file ${REPO_DIR}/recipes/zero3_dsi.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 2 \
    ${ADA_DIR}/ada_repo.py \
    --config ${ADA_DIR}/configs/DSI_v15_1_LogP_boosted.yaml \
    --output_dir ${OUTPUT_DIR} \
    --variant default \
    --run_name "adarepo_v15_1_retrain" \
    --resume_from_checkpoint ${CKPT}

echo "=== Training complete at $(date) ==="
