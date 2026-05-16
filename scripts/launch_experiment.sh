#!/bin/bash
# ==============================================
# Unified launcher for RePO / AdaRePO experiments
# Usage: ./launch_experiment.sh <method> <subtask> [queue] [max_steps]
#   method:   repo | ada_repo
#   subtask:  LogP | MR | QED
#   queue:    debug-scaling | preemptable (default: preemptable)
#   max_steps: -1 for full, or integer for pilot (default: -1)
#
# Examples:
#   ./launch_experiment.sh repo MR debug-scaling 10    # pilot
#   ./launch_experiment.sh ada_repo QED preemptable     # full run
# ==============================================

set -euo pipefail

METHOD=${1:?"Usage: $0 <repo|ada_repo> <LogP|MR|QED> [queue] [max_steps]"}
SUBTASK=${2:?"Usage: $0 <repo|ada_repo> <LogP|MR|QED> [queue] [max_steps]"}
QUEUE=${3:-preemptable}
MAX_STEPS=${4:--1}

REPO_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/RePO
ADA_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo
SCRIPT_DIR=${ADA_DIR}/scripts

# Validate inputs
if [[ "$METHOD" != "repo" && "$METHOD" != "ada_repo" ]]; then
    echo "ERROR: method must be 'repo' or 'ada_repo', got '$METHOD'"
    exit 1
fi

if [[ "$SUBTASK" != "LogP" && "$SUBTASK" != "MR" && "$SUBTASK" != "QED" ]]; then
    echo "ERROR: subtask must be 'LogP', 'MR', or 'QED', got '$SUBTASK'"
    exit 1
fi

# Set walltime based on queue
if [[ "$QUEUE" == "debug-scaling" ]]; then
    WALLTIME="01:00:00"
elif [[ "$QUEUE" == "preemptable" ]]; then
    WALLTIME="06:00:00"
else
    WALLTIME="06:00:00"
fi

# Select config and entry point
if [[ "$METHOD" == "repo" ]]; then
    CONFIG=${REPO_DIR}/recipes/Polaris_3B_${SUBTASK}.yaml
    ENTRY=${REPO_DIR}/src/x_r1/repo.py
    JOB_NAME="repo_3B_${SUBTASK}"
else
    CONFIG=${ADA_DIR}/configs/ada_repo_3B_${SUBTASK}.yaml
    ENTRY=${ADA_DIR}/ada_repo.py
    JOB_NAME="ada_repo_3B_${SUBTASK}"
fi

# Verify config exists
if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config not found: $CONFIG"
    exit 1
fi

echo "=============================================="
echo "Launching experiment:"
echo "  Method:    $METHOD"
echo "  Subtask:   $SUBTASK"
echo "  Queue:     $QUEUE"
echo "  Walltime:  $WALLTIME"
echo "  Max steps: $MAX_STEPS"
echo "  Config:    $CONFIG"
echo "  Entry:     $ENTRY"
echo "  Job name:  $JOB_NAME"
echo "=============================================="

# Generate PBS script on the fly
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

set -euo pipefail

REPO_DIR=${REPO_DIR}
ADA_DIR=${ADA_DIR}
ENV_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/repo_env

LOG_DIR=${ADA_DIR}/logs
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
echo "Experiment: ${METHOD} / ${SUBTASK}"
echo "Node: \$(hostname)"
echo "Date: \$(date)"
echo "Config: ${CONFIG}"
echo "Max steps: ${MAX_STEPS}"
echo "=============================================="

python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)" 2>/dev/null || {
    pip install flash-attn --no-build-isolation 2>&1 | tail -5 || true
}

nvidia-smi

# Build extra args
EXTRA_ARGS=""
if [[ "${MAX_STEPS}" != "-1" ]]; then
    EXTRA_ARGS="--max_steps ${MAX_STEPS}"
fi

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
    \${EXTRA_ARGS}

echo "=============================================="
echo "Experiment ${METHOD}/${SUBTASK} complete at \$(date)"
echo "=============================================="
PBSEOF

echo "Generated PBS script: $PBS_SCRIPT"
echo ""
cat "$PBS_SCRIPT"
echo ""
echo "Submitting..."
qsub "$PBS_SCRIPT"
