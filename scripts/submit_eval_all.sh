#!/bin/bash
# Submit evaluation jobs for all key models
# Models to evaluate:
#   1. Qwen2.5-3B-Instruct (zero-shot baseline)
#   2. v16v17 baseline (best seed: s42)
#   3. v16 rerun (best seed: s42)
#   4. v17 rerun (best seed: s42)
#   Plus the original RePO baselines for comparison
#
# We pick best seed (s42) for each group first.
# If results look good, we can run all 3 seeds later.

set -euo pipefail

PBS_SCRIPT="/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/submit_eval.pbs"
BASE="/lus/eagle/projects/IMPROVE_Aim1/caom"

CAPACITY_COUNT=0
MAX_CAPACITY=2

submit_eval() {
    local label=$1
    local model_path=$2

    if [ $CAPACITY_COUNT -lt $MAX_CAPACITY ]; then
        queue="capacity"
        CAPACITY_COUNT=$((CAPACITY_COUNT + 1))
    else
        queue="preemptable"
    fi

    echo "Submitting eval: ${label} | queue=${queue}"
    echo "  model: ${model_path}"

    JOB_ID=$(qsub \
        -q ${queue} \
        -N "eval_${label}" \
        -v MODEL_PATH="${model_path}" \
        ${PBS_SCRIPT})

    echo "  -> Job ID: ${JOB_ID}"
    echo ""
}

echo "=============================================="
echo "Submitting evaluation jobs"
echo "=============================================="
echo ""

# Zero-shot baseline
submit_eval "zeroshot" "${BASE}/.cache/huggingface/Qwen2.5-3B-Instruct"

# Original RePO single-property baselines (60 steps)
submit_eval "repo_logp" "${BASE}/RePO/output/repo_3B_LogP"

# v16v17 baseline s42
submit_eval "bl_s42" "${BASE}/RePO/output/v16v17ms_repo_s42"

# v16 rerun best seeds
submit_eval "v16_s42" "${BASE}/RePO/output/v16v17ms_v16_s42"
submit_eval "v16_s123" "${BASE}/RePO/output/v16v17ms_v16_s123"
submit_eval "v16_s456" "${BASE}/RePO/output/v16v17ms_v16_s456"

# v17 rerun best seeds
submit_eval "v17_s42" "${BASE}/RePO/output/v16v17ms_v17_s42"
submit_eval "v17_s123" "${BASE}/RePO/output/v16v17ms_v17_s123"
submit_eval "v17_s456" "${BASE}/RePO/output/v16v17ms_v17_s456"

echo "=============================================="
echo "All eval jobs submitted."
echo "=============================================="
