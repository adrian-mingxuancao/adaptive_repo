#!/bin/bash
# Submit GRPO baseline (no reference guidance) — 3 seeds
# Uses same PBS template as v16v17 rerun

set -euo pipefail

PBS_SCRIPT="/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/submit_v16v17_rerun.pbs"
CONFIG="recipes/grpo_baseline.yaml"
BASE_OUT="/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output"

echo "=============================================="
echo "Submitting GRPO baseline (no ref guidance)"
echo "Config: ${CONFIG}"
echo "=============================================="

for SEED in 42 123 456; do
    OUT_DIR="${BASE_OUT}/grpo_baseline_s${SEED}"
    RUN_NAME="grpo_s${SEED}"

    echo "Submitting: seed=${SEED} output=${OUT_DIR}"
    JOB_ID=$(qsub -q preemptable -N grpo_bl \
        -v CONFIG="${CONFIG}",SEED="${SEED}",OUTPUT_DIR="${OUT_DIR}",RUN_NAME="${RUN_NAME}" \
        ${PBS_SCRIPT})
    echo "  -> Job ID: ${JOB_ID}"
done

echo ""
echo "All 3 GRPO baseline jobs submitted."
