#!/bin/bash
# ==============================================
# Phase 1 — Undertraining Diagnosis Launcher
# Submits matched RePO vs AdaRePO runs at
# multiple step budgets (120, 240, 480) for MR and QED.
# Step-60 runs already exist from Phase 0.
#
# Usage:
#   ./launch_phase1.sh [wave]
#   wave 1 = 120 steps only (quick check)
#   wave 2 = 240 steps
#   wave 3 = 480 steps
#   wave all = all three
# ==============================================

set -euo pipefail

WAVE=${1:-1}

REPO_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/RePO
ADA_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo
ENV_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/repo_env
LOG_DIR=${ADA_DIR}/logs
QUEUE=preemptable

mkdir -p "${LOG_DIR}"

# Define step budgets per wave
case "$WAVE" in
    1)   STEPS_LIST="120" ;;
    2)   STEPS_LIST="240" ;;
    3)   STEPS_LIST="480" ;;
    all) STEPS_LIST="120 240 480" ;;
    *)   echo "Usage: $0 [1|2|3|all]"; exit 1 ;;
esac

# Subtasks for Phase 1
SUBTASKS="MR QED"

# Methods
METHODS="repo ada_repo"

submit_job() {
    local METHOD=$1
    local SUBTASK=$2
    local MAX_STEPS=$3

    # Set config and entry point
    if [[ "$METHOD" == "repo" ]]; then
        CONFIG=${REPO_DIR}/recipes/Polaris_3B_${SUBTASK}.yaml
        ENTRY=${REPO_DIR}/src/x_r1/repo.py
    else
        CONFIG=${ADA_DIR}/configs/ada_repo_3B_${SUBTASK}.yaml
        ENTRY=${ADA_DIR}/ada_repo.py
    fi

    JOB_NAME="p1_${METHOD}_${SUBTASK}_s${MAX_STEPS}"

    # Unique output directory per step budget
    if [[ "$METHOD" == "repo" ]]; then
        OUTPUT_DIR="${REPO_DIR}/output/p1_repo_3B_${SUBTASK}_s${MAX_STEPS}"
    else
        OUTPUT_DIR="${ADA_DIR}/output/p1_ada_repo_3B_${SUBTASK}_s${MAX_STEPS}"
    fi

    # Walltime estimate: ~6.5 min/step, add buffer
    # 120 steps ≈ 3.5h, 240 ≈ 6h, 480 ≈ 6h (will need resume)
    if [[ $MAX_STEPS -le 120 ]]; then
        WALLTIME="04:00:00"
    elif [[ $MAX_STEPS -le 240 ]]; then
        WALLTIME="06:00:00"
    else
        WALLTIME="06:00:00"
    fi

    echo "=============================================="
    echo "Submitting: ${JOB_NAME}"
    echo "  Method:    ${METHOD}"
    echo "  Subtask:   ${SUBTASK}"
    echo "  Steps:     ${MAX_STEPS}"
    echo "  Output:    ${OUTPUT_DIR}"
    echo "  Walltime:  ${WALLTIME}"
    echo "=============================================="

    # Generate PBS script
    PBS_SCRIPT=$(mktemp /tmp/${JOB_NAME}_XXXXXX.pbs)
    cat > "$PBS_SCRIPT" <<PBSEOF
#!/bin/bash
#PBS -l select=1:system=polaris:ncpus=64:ngpus=4
#PBS -l walltime=${WALLTIME}
#PBS -l filesystems=home:eagle
#PBS -q ${QUEUE}
#PBS -A IMPROVE_Aim1
#PBS -N ${JOB_NAME}
#PBS -j oe
#PBS -r y

set -euo pipefail

REPO_DIR=${REPO_DIR}
ADA_DIR=${ADA_DIR}
ENV_DIR=${ENV_DIR}

LOG_DIR=${LOG_DIR}
mkdir -p \${LOG_DIR}
exec > \${LOG_DIR}/${JOB_NAME}_\${PBS_JOBID}.log 2>&1

export PATH="\${ENV_DIR}/bin:\${PATH}"
export PYTHONPATH="\${REPO_DIR}/src/x_r1:\${ADA_DIR}:\${PYTHONPATH:-}"

export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda
CUDA_MATH_LIBS=/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/math_libs/12.9/targets/x86_64-linux/lib
export LIBRARY_PATH=\${CUDA_HOME}/lib64:\${CUDA_MATH_LIBS}:\${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=\${CUDA_HOME}/lib64:\${CUDA_MATH_LIBS}:\${LD_LIBRARY_PATH:-}
export DS_SKIP_CUDA_CHECK=1

export HF_HOME=/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface
export TRANSFORMERS_CACHE=/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface
export HF_DATASETS_CACHE=/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface/datasets

export HTTP_PROXY="http://proxy.alcf.anl.gov:3128"
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"
export ftp_proxy="http://proxy.alcf.anl.gov:3128"
export no_proxy="admin,polaris-adminvm-01,localhost,*.cm.polaris.alcf.anl.gov,polaris-*,*.polaris.alcf.anl.gov,*.alcf.anl.gov"

export WANDB_MODE=online
export WANDB_DIR=\${ADA_DIR}/logs
export NCCL_DEBUG=WARN
export NCCL_COLLNET_ENABLE=0
export MASTER_ADDR=localhost
export MASTER_PORT=29501
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OPENBLAS_NUM_THREADS=1

cd \${REPO_DIR}

echo "=============================================="
echo "Phase 1: ${METHOD} / ${SUBTASK} / ${MAX_STEPS} steps"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "Config: ${CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================="

python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || {
    pip install flash-attn --no-build-isolation 2>&1 | tail -5 || true
}

nvidia-smi

CUDA_VISIBLE_DEVICES=0,1,2,3 \\
NO_PROXY=localhost,127.0.0.1 \\
no_proxy=localhost,127.0.0.1 \\
ACCELERATE_LOG_LEVEL=info \\
  accelerate launch \\
    --config_file recipes/zero3_polaris.yaml \\
    --main_process_port \${MASTER_PORT} \\
    --num_processes 3 \\
    ${ENTRY} \\
    --config ${CONFIG} \\
    --variant default \\
    --max_steps ${MAX_STEPS} \\
    --output_dir ${OUTPUT_DIR} \\
    --save_strategy steps \\
    --save_steps 15 \\
    --save_total_limit 10 \\
    --run_name "${JOB_NAME}"

echo "=============================================="
echo "Phase 1 ${METHOD}/${SUBTASK}/${MAX_STEPS} complete at \$(date)"
echo "=============================================="
PBSEOF

    echo "Generated: $PBS_SCRIPT"
    qsub "$PBS_SCRIPT"
    echo ""
}

# Submit all jobs for the selected wave
for STEPS in $STEPS_LIST; do
    for SUBTASK in $SUBTASKS; do
        for METHOD in $METHODS; do
            submit_job "$METHOD" "$SUBTASK" "$STEPS"
        done
    done
done

echo "=============================================="
echo "All Phase 1 Wave ${WAVE} jobs submitted."
echo "=============================================="
