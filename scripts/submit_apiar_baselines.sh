#!/bin/bash
# ==============================================
# Master submission script for APIAR paper baselines
# Submits jobs in dependency order
# ==============================================

set -euo pipefail

BASE="/lus/eagle/projects/IMPROVE_Aim1/caom/RePO"

echo "=============================================="
echo "APIAR Paper — Baseline Job Submission"
echo "=============================================="
echo ""

# --- Step 1: C2 preprocessing (needs 1 GPU, ~2h) ---
echo "[Step 1] C2: Offline-strengthened reference generation"
C2_JOB=$(qsub -q preemptable ${BASE}/submit_c2_strengthen.pbs)
echo "  C2 preprocess job: ${C2_JOB}"
C2_JOBID=$(echo ${C2_JOB} | cut -d. -f1)

# --- Step 2: C2 training (depends on C2 preprocessing) ---
echo ""
echo "[Step 2] C2: Offline-strengthened RePO training (3 seeds, after C2 preprocess)"
PBS_SCRIPT="${BASE}/submit_v16v17_rerun.pbs"
CONFIG="recipes/offline_repo.yaml"

for SEED in 42 123 456; do
    OUT_DIR="${BASE}/output/offline_repo_s${SEED}"
    RUN_NAME="offline_repo_s${SEED}"
    JOB=$(qsub -q preemptable -N c2_train \
        -W depend=afterok:${C2_JOBID} \
        -v CONFIG="${CONFIG}",SEED="${SEED}",OUTPUT_DIR="${OUT_DIR}",RUN_NAME="${RUN_NAME}" \
        ${PBS_SCRIPT})
    echo "  C2 train seed=${SEED}: ${JOB}"
done

# --- Step 3: C1 Iterative SFT (3 seeds, independent) ---
echo ""
echo "[Step 3] C1: Iterative SFT baseline (3 seeds)"
for SEED in 42 123 456; do
    OUT_DIR="${BASE}/output/isft_s${SEED}"
    JOB=$(qsub -q preemptable \
        -v SEED="${SEED}",OUTPUT_DIR="${OUT_DIR}" \
        ${BASE}/submit_c1_isft.pbs)
    echo "  C1 ISFT seed=${SEED}: ${JOB}"
done

echo ""
echo "=============================================="
echo "All baseline jobs submitted."
echo ""
echo "After completion, run eval with:"
echo "  bash submit_eval_all.sh  (update with new model paths)"
echo "  python scripts/aggregate_results.py --print-table"
echo "=============================================="
