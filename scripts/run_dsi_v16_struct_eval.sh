#!/bin/bash
#SBATCH --job-name=v16s-eval
#SBATCH --partition=general
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=0
#SBATCH --exclude=g001,g010
#SBATCH --output=/net/scratch/caom/repo_project/logs/v16_struct_eval_%j.out
#SBATCH --error=/net/scratch/caom/repo_project/logs/v16_struct_eval_%j.err

# ==============================================
# Evaluate AdaRePO v16-struct on MolEdit benchmark
# Automatically finds latest checkpoint
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

TRAIN_OUTPUT=${SCRATCH}/outputs/ada_repo_dsi_v16_struct
PRED_DIR=${SCRATCH}/outputs/eval_v16_struct/predictions/

# Find latest checkpoint
CKPT=$(ls -d ${TRAIN_OUTPUT}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
if [ -z "$CKPT" ]; then
    echo "ERROR: No checkpoint found in ${TRAIN_OUTPUT}"
    exit 1
fi

echo "=============================================="
echo "AdaRePO v16-struct Eval"
echo "Checkpoint: ${CKPT}"
echo "Node: $(hostname), Date: $(date)"
echo "=============================================="

cd ${REPO_DIR}

for SUBTASK in AddComponent DelComponent SubComponent; do
    echo "--- Generating for ${SUBTASK} ---"
    python3 ${REPO_DIR}/generate_predictions.py \
        --model_path ${CKPT} \
        --benchmark open_generation \
        --task MolEdit \
        --subtask ${SUBTASK} \
        --output_dir ${PRED_DIR} \
        --gpu_memory_utilization 0.85 \
        --lang en
    echo "${SUBTASK} predictions done at $(date)"
done

for SUBTASK in AddComponent DelComponent SubComponent; do
    echo "--- Evaluating ${SUBTASK} ---"
    python3 ${REPO_DIR}/evaluate.py \
        --model_path "${CKPT}" \
        --benchmark "open_generation" \
        --task "MolEdit" \
        --subtask "${SUBTASK}" \
        --output_dir "${PRED_DIR}"
    echo "${SUBTASK} evaluation done at $(date)"
done

echo "=== All done at $(date) ==="
