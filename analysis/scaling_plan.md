# Scaling Plan — AdaRePO Diagnostic Experiments

**Date:** 2026-03-13
**Prerequisite:** Phase 0 audit complete (see `current_setup_audit.md`)

---

## Phase 1 — Undertraining Diagnosis (Step Scaling)

### Goal
Determine whether the MR/QED gap between RePO and AdaRePO is caused by insufficient training steps.

### Design
- **Data:** Fixed at 500 examples per subtask (OpenMolIns light)
- **Subtasks:** MR, QED (LogP as anchor sanity check if needed)
- **Methods:** RePO vs AdaRePO
- **Step budgets:** 60, 120, 240, 480
- **Everything else:** Identical to Phase 0 setup

### Experiment Matrix (Phase 1)

| Run ID | Method | Subtask | max_steps | Config |
|--------|--------|---------|-----------|--------|
| p1_repo_MR_60 | RePO | MR | 60 | *(already done)* |
| p1_repo_MR_120 | RePO | MR | 120 | Polaris_3B_MR.yaml + --max_steps 120 |
| p1_repo_MR_240 | RePO | MR | 240 | Polaris_3B_MR.yaml + --max_steps 240 |
| p1_repo_MR_480 | RePO | MR | 480 | Polaris_3B_MR.yaml + --max_steps 480 |
| p1_ada_MR_60 | AdaRePO | MR | 60 | *(already done)* |
| p1_ada_MR_120 | AdaRePO | MR | 120 | ada_repo_3B_MR.yaml + --max_steps 120 |
| p1_ada_MR_240 | AdaRePO | MR | 240 | ada_repo_3B_MR.yaml + --max_steps 240 |
| p1_ada_MR_480 | AdaRePO | MR | 480 | ada_repo_3B_MR.yaml + --max_steps 480 |
| p1_repo_QED_60 | RePO | QED | 60 | *(already done)* |
| p1_repo_QED_120 | RePO | QED | 120 | Polaris_3B_QED.yaml + --max_steps 120 |
| p1_repo_QED_240 | RePO | QED | 240 | Polaris_3B_QED.yaml + --max_steps 240 |
| p1_repo_QED_480 | RePO | QED | 480 | Polaris_3B_QED.yaml + --max_steps 480 |
| p1_ada_QED_60 | AdaRePO | QED | 60 | *(already done)* |
| p1_ada_QED_120 | AdaRePO | QED | 120 | ada_repo_3B_QED.yaml + --max_steps 120 |
| p1_ada_QED_240 | AdaRePO | QED | 240 | ada_repo_3B_QED.yaml + --max_steps 240 |
| p1_ada_QED_480 | AdaRePO | QED | 480 | ada_repo_3B_QED.yaml + --max_steps 480 |

**Total new runs:** 12 (reusing 4 existing step-60 runs)
**Estimated wall time per run:**
- 60 steps ≈ 1.7 hours (already observed)
- 120 steps ≈ 3.4 hours
- 240 steps ≈ 6.8 hours (borderline for preemptable 6h, may need 2 submissions)
- 480 steps ≈ 13.6 hours (needs preemptable with resume)

### Config Modifications Required
No new config files needed. The existing configs work with `--max_steps N` override.
Key changes to `launch_experiment.sh`:
1. Override `max_steps` via command line arg
2. Set unique `output_dir` per step budget (e.g., `output/repo_3B_MR_s120`)
3. Set unique `wandb_run_name` per step budget
4. Checkpoint every 15 steps for learning curve extraction
5. Enable `--save_steps 15` alongside epoch saves

### Checkpoint Strategy
- Save every 15 steps: `--save_steps 15 --save_strategy steps`
- This gives checkpoints at: 15, 30, 45, 60, 75, 90, ..., 480
- For evaluation: evaluate at 60, 120, 240, 480 (and optionally intermediate)
- Each checkpoint ≈ 6 GB; 480-step run produces ~32 checkpoints → ~192 GB per run
- **Optimization:** Use `--save_total_limit 8` to keep only last 8 checkpoints per run

### Metrics to Capture (per step)
From trainer logs (automatic):
- loss, reward (rewards/smile_optimization), kl, s_loss, completion_length
- beta_guide_mean, beta_guide_std (AdaRePO only)
- v_top_minus_v_ref (AdaRePO only)
- s_loss_weighted (AdaRePO only)

From evaluation (at selected checkpoints):
- success_rate, validity, similarity

### HPC Plan
**Queue:** preemptable (6h walltime, rerunnable)
**Node count:** 1 node (4x A100 40GB) per run
**Strategy:**
1. Submit MR runs first (both methods, all step budgets) as separate jobs
2. Submit QED runs after MR completes (or in parallel if nodes available)
3. 240-step and 480-step runs: enable resume via `#PBS -r y`

**Batching:** Submit 4 jobs at a time (2 methods × 2 subtasks at one step budget)

### Conservative Run Plan

**Wave 1 (first submission):**
- p1_repo_MR_120, p1_ada_MR_120, p1_repo_QED_120, p1_ada_QED_120
- Walltime: 3.5h each, fits in preemptable
- Purpose: Quick check — does doubling steps change the gap?

**Wave 2 (after Wave 1 analysis):**
- p1_repo_MR_240, p1_ada_MR_240, p1_repo_QED_240, p1_ada_QED_240
- Walltime: ~6h each, needs careful walltime or resume

**Wave 3 (if warranted):**
- p1_repo_MR_480, p1_ada_MR_480, p1_repo_QED_480, p1_ada_QED_480
- Only run if 240-step results are inconclusive

---

## Phase 2 — Data Scaling (outline)

### Design
- Pick best step budget from Phase 1 (likely 240 or 480)
- Train sizes: 500, 1000, 2000, full (4500 for all subtasks combined, or use medium/large scale)
- Focus on MR and QED
- Same evaluation protocol

### Data Availability
- light scale: 500 per subtask (current)
- Need to verify: small/medium/large scales in OpenMolIns data directory

---

## Phase 3 — Beta Calibration (outline)

### Variants to Implement
| Variant | Description | Key Change |
|---------|-------------|------------|
| A (baseline) | Current sigmoid_gap | No change |
| B | Normalized-gap beta | Divide gap by running std |
| C | Top-k mean normalized | Use top-k mean + normalization |
| D | Warm-start beta | Beta floor for first N steps |
| E | Per-task calibration | Task-specific tau/beta_max/beta_min |

### Implementation Plan
1. Extend `dynamic_beta.py` with new modes
2. Add running statistics tracker for normalization
3. Add config parameters for warm-start and per-task settings
4. Smoke test each variant on 10-step runs
5. Pilot on 60-step MR/QED
6. Full comparison for best 1-2 variants

---

## Phase 4 — Multitask (outline)

### Design
- Single model trained on all 3 subtasks jointly
- Task identifier in prompt (already present in OpenMolIns instructions)
- Balanced sampling: 500 per subtask = 1500 total
- Per-task reward normalization
- Per-task beta calibration (if Phase 3 supports it)

---

## Files to Modify / Create

### Phase 1 (immediate)
1. **Modify:** `scripts/launch_experiment.sh` — add step-budget output dir logic, save_steps override
2. **Create:** `scripts/launch_phase1.sh` — batch launcher for all Phase 1 runs
3. **Create:** `scripts/evaluate_phase1.sh` — batch evaluation for Phase 1 checkpoints
4. **Create:** `analysis/phase1_analysis.py` — learning curve extraction and plotting
5. **Update:** `analysis/run_manifest.csv` — add Phase 1 run entries

### Phase 3 (later)
6. **Modify:** `dynamic_beta.py` — add normalized_gap, warm_start modes
7. **Modify:** `ada_repo_config.py` — add new config parameters
8. **Create:** New config files for beta variants

### Phase 4 (later)
9. **Create:** Multitask configs
10. **Modify:** Dataset loading to support balanced multitask sampling
