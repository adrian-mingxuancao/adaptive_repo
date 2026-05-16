#!/bin/bash
#PBS -l select=1:system=polaris:ncpus=64:ngpus=1
#PBS -l walltime=03:00:00
#PBS -l filesystems=home:eagle
#PBS -q preemptable
#PBS -A IMPROVE_Aim1
#PBS -N tomg_eval
#PBS -j oe
#PBS -r y

# ==============================================
# TOMG-Bench Evaluation: RePO vs AdaRePO
# Generates predictions + evaluates on MolOpt test sets
# Uses 1 GPU with vLLM for inference
# ==============================================

set -euo pipefail

# ---- Paths ----
REPO_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/RePO
ADA_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo
ENV_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/repo_env

# ---- Log setup ----
LOG_DIR=${ADA_DIR}/logs
mkdir -p ${LOG_DIR}
exec > ${LOG_DIR}/tomg_eval_${PBS_JOBID}.log 2>&1

# ---- Environment ----
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${REPO_DIR}/src/x_r1:${ADA_DIR}:${PYTHONPATH:-}"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda
CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib
export LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_MATH_LIBS}:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_MATH_LIBS}:${LD_LIBRARY_PATH:-}

export HF_HOME=/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface

export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1

cd ${REPO_DIR}

echo "=============================================="
echo "TOMG-Bench Evaluation: RePO vs AdaRePO"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "=============================================="

nvidia-smi

# Output directory for all predictions
PRED_DIR=${ADA_DIR}/evaluation_results
mkdir -p ${PRED_DIR}

# Define checkpoints and sample counts
declare -A CHECKPOINTS
CHECKPOINTS["repo_LogP"]="${REPO_DIR}/output/repo_3B_LogP/checkpoint-60"
CHECKPOINTS["repo_MR"]="${REPO_DIR}/output/repo_3B_MR/checkpoint-60"
CHECKPOINTS["repo_QED"]="${REPO_DIR}/output/repo_3B_QED/checkpoint-60"
CHECKPOINTS["ada_repo_LogP"]="${ADA_DIR}/output/ada_repo_3B_LogP/checkpoint-60"
CHECKPOINTS["ada_repo_MR"]="${ADA_DIR}/output/ada_repo_3B_MR/checkpoint-60"
CHECKPOINTS["ada_repo_QED"]="${ADA_DIR}/output/ada_repo_3B_QED/checkpoint-60"

declare -A SAMPLES
SAMPLES["LogP"]=5000
SAMPLES["MR"]=1000
SAMPLES["QED"]=1000

# Run evaluation for each model
for key in repo_LogP ada_repo_LogP repo_MR ada_repo_MR repo_QED ada_repo_QED; do
    CKPT=${CHECKPOINTS[$key]}
    # Extract subtask from key (last part after _)
    SUBTASK=${key##*_}
    N_SAMPLES=${SAMPLES[$SUBTASK]}
    
    echo ""
    echo "=============================================="
    echo "Evaluating: ${key}"
    echo "  Checkpoint: ${CKPT}"
    echo "  Subtask: ${SUBTASK}"
    echo "  Samples: ${N_SAMPLES}"
    echo "=============================================="
    
    # Use unique output directory per model
    MODEL_OUT_DIR="${PRED_DIR}/${key}"
    mkdir -p "${MODEL_OUT_DIR}"
    
    # Generate predictions
    echo "[$(date)] Generating predictions..."
    python generate_predictions.py \
        --model_path "${CKPT}" \
        --benchmark open_generation \
        --task MolOpt \
        --subtask "${SUBTASK}" \
        --output_dir "${MODEL_OUT_DIR}/" \
        --lang en \
        --gpu_memory_utilization 0.85 \
        --max_samples ${N_SAMPLES}
    
    # Evaluate
    echo "[$(date)] Evaluating predictions..."
    python evaluate.py \
        --model_path "${CKPT}" \
        --benchmark open_generation \
        --task MolOpt \
        --subtask "${SUBTASK}" \
        --output_dir "${MODEL_OUT_DIR}/" \
        --max_samples ${N_SAMPLES}
    
    echo "[$(date)] Done with ${key}"
done

echo ""
echo "=============================================="
echo "All evaluations complete at $(date)"
echo "Results saved to: ${PRED_DIR}"
echo "=============================================="

# Generate summary table
echo ""
echo "=============================================="
echo "SUMMARY TABLE"
echo "=============================================="
echo "Model,Subtask,Success_Rate,Validity,Similarity"
for key in repo_LogP ada_repo_LogP repo_MR ada_repo_MR repo_QED ada_repo_QED; do
    SUBTASK=${key##*_}
    SUMMARY_FILE="${PRED_DIR}/${key}/checkpoint-60/open_generation/MolOpt/${SUBTASK}_summary.csv"
    if [[ -f "${SUMMARY_FILE}" ]]; then
        # Read CSV and extract values (skip header)
        tail -1 "${SUMMARY_FILE}" | while IFS=, read -r sr sim val total valid succ; do
            echo "${key},${SUBTASK},${sr},${val},${sim}"
        done
    else
        echo "${key},${SUBTASK},N/A,N/A,N/A"
    fi
done
