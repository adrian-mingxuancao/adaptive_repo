#!/bin/bash
#SBATCH --partition=general
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --exclude=g003,g004,g005,g006,g007,g008,g009,g010

# ==============================================
# Unified experiment launcher
# Usage: sbatch --job-name=<name> -o logs/<name>_%j.log -e logs/<name>_%j.log \
#          scripts/run_experiment.sh <method> <seed>
#   method: repo | adarepo | adarepo_pw
#   seed: 42 | 123 | 456
# ==============================================

set -euo pipefail

METHOD=${1:?"Usage: $0 <repo|adarepo|adarepo_pw> <seed>"}
SEED=${2:?"Usage: $0 <repo|adarepo|adarepo_pw> <seed>"}

PROJECT_DIR=/net/scratch2/qinanh/agent_drug_discovery
REPO_DIR=${PROJECT_DIR}/RePO
ADA_DIR=${PROJECT_DIR}/adaptive_repo
ENV_DIR=/net/scratch2/qinanh/miniconda3/envs/adarepo

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
export MASTER_PORT=$((29500 + RANDOM % 100))
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd ${REPO_DIR}
mkdir -p ${ADA_DIR}/logs

# Select entry point + config based on method
case ${METHOD} in
    repo)
        ENTRY=${REPO_DIR}/src/x_r1/repo.py
        CONFIG=${REPO_DIR}/recipes/local_3B_QED.yaml
        OUTPUT_DIR=${REPO_DIR}/output/repo_QED_s${SEED}
        RUN_NAME="repo_QED_s${SEED}"
        ;;
    adarepo)
        ENTRY=${ADA_DIR}/ada_repo.py
        CONFIG=${ADA_DIR}/configs/ada_repo_3B_QED_nopw.yaml
        OUTPUT_DIR=${ADA_DIR}/output/adarepo_QED_s${SEED}
        RUN_NAME="adarepo_QED_s${SEED}"
        ;;
    adarepo_pw)
        ENTRY=${ADA_DIR}/ada_repo.py
        CONFIG=${ADA_DIR}/configs/ada_repo_3B_QED.yaml
        OUTPUT_DIR=${ADA_DIR}/output/adarepo_pw_QED_s${SEED}
        RUN_NAME="adarepo_pw_QED_s${SEED}"
        ;;
    *)
        echo "Unknown method: ${METHOD}"
        exit 1
        ;;
esac

# Clean old output to avoid resume issues
rm -rf ${OUTPUT_DIR} 2>/dev/null

echo "=============================================="
echo "Experiment: ${METHOD} / QED / seed=${SEED}"
echo "  Node:   $(hostname)"
echo "  Date:   $(date)"
echo "  Entry:  ${ENTRY}"
echo "  Config: ${CONFIG}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================================="

nvidia-smi

ACCELERATE_LOG_LEVEL=info accelerate launch \
    --config_file recipes/zero3.yaml \
    --main_process_port ${MASTER_PORT} \
    --num_processes 3 \
    ${ENTRY} \
    --config ${CONFIG} \
    --seed ${SEED} \
    --output_dir ${OUTPUT_DIR} \
    --run_name ${RUN_NAME}

echo "=== ${METHOD} QED s${SEED} Done at $(date) ==="
