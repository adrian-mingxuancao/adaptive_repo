#!/bin/bash
#SBATCH --job-name=adarepo
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --exclude=g003,g004,g005,g006,g007,g008,g009,g010
#SBATCH --output=logs/adarepo_%j.log
#SBATCH --error=logs/adarepo_%j.log

# ==============================================
# AdaRePO Training on UChicago SLURM cluster
# Usage:
#   sbatch submit_slurm.sh                          # default: QED subtask
#   sbatch submit_slurm.sh LogP                     # LogP subtask
#   sbatch submit_slurm.sh MR                       # MR subtask
#   sbatch submit_slurm.sh QED ada_repo_3B_QED      # QED with specific config
# ==============================================

set -euo pipefail

SUBTASK=${1:-QED}
CONFIG_NAME=${2:-ada_repo_3B_${SUBTASK}}

# Paths
PROJECT_DIR=/net/scratch2/qinanh/agent_drug_discovery
ADA_DIR=${PROJECT_DIR}/adaptive_repo
REPO_DIR=${PROJECT_DIR}/RePO
ENV_DIR=/net/scratch2/qinanh/miniconda3/envs/adarepo

CONFIG=${ADA_DIR}/configs/${CONFIG_NAME}.yaml
ENTRY=${ADA_DIR}/ada_repo.py

# Create log dir
mkdir -p ${ADA_DIR}/logs

echo "=============================================="
echo "AdaRePO Training"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Subtask:   ${SUBTASK}"
echo "  Config:    ${CONFIG}"
echo "  Entry:     ${ENTRY}"
echo "  GPUs:      ${CUDA_VISIBLE_DEVICES:-all}"
echo "=============================================="

# Environment
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${ADA_DIR}:${PYTHONPATH:-}"
export HF_HOME=/net/scratch2/qinanh/hf_cache
export TRANSFORMERS_CACHE=/net/scratch2/qinanh/hf_cache
export HF_DATASETS_CACHE=/net/scratch2/qinanh/hf_cache/datasets
export TRITON_CACHE_DIR=/net/scratch2/qinanh/.triton
export WANDB_MODE=offline
export WANDB_DIR=${ADA_DIR}/logs
export NCCL_DEBUG=WARN
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1

# Verify setup
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
nvidia-smi

cd ${REPO_DIR}

# Launch with accelerate (3 training processes + 1 vLLM process = 4 GPUs)
ACCELERATE_LOG_LEVEL=info \
  accelerate launch \
    --config_file recipes/zero3.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 1 \
    ${ENTRY} \
    --config ${CONFIG}

echo "=============================================="
echo "AdaRePO ${SUBTASK} training complete at $(date)"
echo "=============================================="
