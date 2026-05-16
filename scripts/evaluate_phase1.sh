#!/bin/bash
# ==============================================
# Phase 1 — Evaluation Script
# Evaluates multiple checkpoints from Phase 1 runs.
#
# Usage:
#   ./evaluate_phase1.sh [subtask] [max_samples]
#   subtask:     MR | QED | all (default: all)
#   max_samples: number of test samples (default: 1000)
#
# Evaluates checkpoint-60, checkpoint-120, checkpoint-240, checkpoint-480
# for both RePO and AdaRePO, wherever checkpoints exist.
# ==============================================

set -euo pipefail

SUBTASK_FILTER=${1:-all}
MAX_SAMPLES=${2:-1000}

REPO_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/RePO
ADA_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo
ENV_DIR=/lus/eagle/projects/IMPROVE_Aim1/caom/repo_env
EVAL_BASE=${ADA_DIR}/evaluation_results/phase1

mkdir -p "${EVAL_BASE}"

# All step budgets and checkpoints to evaluate
STEP_BUDGETS="60 120 240 480"
CHECKPOINTS="60 120 240 480"

# Build list of subtasks
if [[ "$SUBTASK_FILTER" == "all" ]]; then
    SUBTASKS="MR QED"
else
    SUBTASKS="$SUBTASK_FILTER"
fi

echo "=============================================="
echo "Phase 1 Evaluation"
echo "  Subtasks:    ${SUBTASKS}"
echo "  Max samples: ${MAX_SAMPLES}"
echo "  Output base: ${EVAL_BASE}"
echo "=============================================="

evaluate_checkpoint() {
    local MODEL_PATH=$1
    local SUBTASK=$2
    local RUN_KEY=$3
    local CKPT=$4

    local CKPT_PATH="${MODEL_PATH}/checkpoint-${CKPT}"
    if [[ ! -d "$CKPT_PATH" ]]; then
        echo "SKIP: ${CKPT_PATH} does not exist"
        return 0
    fi

    local OUT_DIR="${EVAL_BASE}/${RUN_KEY}/checkpoint-${CKPT}"
    if [[ -f "${OUT_DIR}/open_generation/MolOpt/${SUBTASK}_summary.csv" ]]; then
        echo "SKIP: ${RUN_KEY}/checkpoint-${CKPT} already evaluated"
        return 0
    fi

    echo "----------------------------------------------"
    echo "Evaluating: ${RUN_KEY} @ checkpoint-${CKPT}"
    echo "  Model: ${CKPT_PATH}"
    echo "  Output: ${OUT_DIR}"
    echo "----------------------------------------------"

    mkdir -p "${OUT_DIR}"

    # Generate predictions
    python ${REPO_DIR}/generate_predictions.py \
        --model_path "${CKPT_PATH}" \
        --benchmark open_generation \
        --task MolOpt \
        --subtask "${SUBTASK}" \
        --output_dir "${OUT_DIR}" \
        --device cuda \
        --gpu_memory_utilization 0.85 \
        --lang en \
        --max_samples "${MAX_SAMPLES}"

    # Evaluate
    python ${REPO_DIR}/evaluate.py \
        --benchmark open_generation \
        --task MolOpt \
        --subtask "${SUBTASK}" \
        --output_dir "${OUT_DIR}" \
        --max_samples "${MAX_SAMPLES}"

    echo "Done: ${RUN_KEY} @ checkpoint-${CKPT}"
}

# Iterate over all combinations
for SUBTASK in $SUBTASKS; do
    for STEPS in $STEP_BUDGETS; do
        for CKPT in $CHECKPOINTS; do
            # Only evaluate checkpoints that exist within the step budget
            if [[ $CKPT -gt $STEPS ]]; then
                continue
            fi

            # RePO paths
            if [[ $STEPS -eq 60 ]]; then
                REPO_MODEL="${REPO_DIR}/output/repo_3B_${SUBTASK}"
            else
                REPO_MODEL="${REPO_DIR}/output/p1_repo_3B_${SUBTASK}_s${STEPS}"
            fi
            evaluate_checkpoint "${REPO_MODEL}" "${SUBTASK}" "repo_${SUBTASK}_s${STEPS}" "${CKPT}"

            # AdaRePO paths
            if [[ $STEPS -eq 60 ]]; then
                ADA_MODEL="${ADA_DIR}/output/ada_repo_3B_${SUBTASK}"
            else
                ADA_MODEL="${ADA_DIR}/output/p1_ada_repo_3B_${SUBTASK}_s${STEPS}"
            fi
            evaluate_checkpoint "${ADA_MODEL}" "${SUBTASK}" "ada_${SUBTASK}_s${STEPS}" "${CKPT}"
        done
    done
done

# Generate summary table
echo ""
echo "=============================================="
echo "Phase 1 Evaluation Summary"
echo "=============================================="

SUMMARY_FILE="${EVAL_BASE}/phase1_summary.csv"
echo "method,subtask,step_budget,checkpoint,success_rate,validity,similarity" > "${SUMMARY_FILE}"

for SUBTASK in $SUBTASKS; do
    for STEPS in $STEP_BUDGETS; do
        for CKPT in $CHECKPOINTS; do
            if [[ $CKPT -gt $STEPS ]]; then
                continue
            fi

            for PREFIX in repo ada; do
                KEY="${PREFIX}_${SUBTASK}_s${STEPS}"
                CSV="${EVAL_BASE}/${KEY}/checkpoint-${CKPT}/open_generation/MolOpt/${SUBTASK}_summary.csv"
                if [[ -f "$CSV" ]]; then
                    # Read the CSV (skip header)
                    METHOD=$([[ "$PREFIX" == "repo" ]] && echo "RePO" || echo "AdaRePO")
                    while IFS=, read -r sr sim val tot vmol sopt; do
                        echo "${METHOD},${SUBTASK},${STEPS},${CKPT},${sr},${val},${sim}" >> "${SUMMARY_FILE}"
                    done < <(tail -1 "$CSV")
                fi
            done
        done
    done
done

echo "Summary written to: ${SUMMARY_FILE}"
echo ""
column -t -s, "${SUMMARY_FILE}" 2>/dev/null || cat "${SUMMARY_FILE}"
echo ""
echo "Phase 1 Evaluation complete at $(date)"
