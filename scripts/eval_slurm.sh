#!/bin/bash
#SBATCH --job-name=eval_ada
#SBATCH --partition=general
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.log

# ==============================================
# Evaluate AdaRePO QED model on TOMG-Bench
# Usage:
#   sbatch scripts/eval_slurm.sh                           # default: QED
#   sbatch scripts/eval_slurm.sh LogP /path/to/checkpoint  # custom
# ==============================================

set -euo pipefail

SUBTASK=${1:-QED}
MODEL_PATH=${2:-/net/scratch2/qinanh/agent_drug_discovery/adaptive_repo/output/ada_repo_3B_QED}

PROJECT_DIR=/net/scratch2/qinanh/agent_drug_discovery
REPO_DIR=${PROJECT_DIR}/RePO
ADA_DIR=${PROJECT_DIR}/adaptive_repo
ENV_DIR=/net/scratch2/qinanh/miniconda3/envs/adarepo

mkdir -p ${ADA_DIR}/logs

export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src:${ADA_DIR}:${PYTHONPATH:-}"
export HF_HOME=/net/scratch2/qinanh/hf_cache
export TRANSFORMERS_CACHE=/net/scratch2/qinanh/hf_cache
export DS_SKIP_CUDA_CHECK=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd ${REPO_DIR}

PRED_DIR=${ADA_DIR}/evaluation_results
MODEL_NAME=$(basename ${MODEL_PATH})
MODEL_OUT_DIR="${PRED_DIR}/${MODEL_NAME}"
mkdir -p "${MODEL_OUT_DIR}"

echo "=============================================="
echo "TOMG-Bench Evaluation"
echo "  Node:      $(hostname)"
echo "  Date:      $(date)"
echo "  Model:     ${MODEL_PATH}"
echo "  Subtask:   ${SUBTASK}"
echo "  Output:    ${MODEL_OUT_DIR}"
echo "=============================================="

nvidia-smi

# Generate predictions
echo "[$(date)] Generating predictions for ${SUBTASK}..."
python generate_predictions.py \
    --model_path "${MODEL_PATH}" \
    --benchmark open_generation \
    --task MolOpt \
    --subtask "${SUBTASK}" \
    --output_dir "${MODEL_OUT_DIR}/" \
    --lang en \
    --gpu_memory_utilization 0.85

# Evaluate
echo "[$(date)] Evaluating ${SUBTASK}..."
python evaluate.py \
    --model_path "${MODEL_PATH}" \
    --benchmark open_generation \
    --task MolOpt \
    --subtask "${SUBTASK}" \
    --output_dir "${MODEL_OUT_DIR}/"

echo ""
echo "=============================================="
echo "Evaluation complete at $(date)"
echo "=============================================="

# Print summary
SUMMARY_FILE="${MODEL_OUT_DIR}/${MODEL_NAME}/open_generation/MolOpt/${SUBTASK}_summary.csv"
if [[ -f "${SUMMARY_FILE}" ]]; then
    echo "Summary:"
    cat "${SUMMARY_FILE}"
else
    # Try alternate path patterns
    find "${MODEL_OUT_DIR}" -name "${SUBTASK}_summary.csv" -exec echo "Summary at {}:" \; -exec cat {} \;
fi
