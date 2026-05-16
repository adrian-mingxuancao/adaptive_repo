# Experiment Plan: RePO vs AdaRePO on TOMG-Bench MolOpt

**Date**: 2026-03-10
**Status**: Phase 1 complete (audit + wiring), ready for Phase 2 (pilot runs)

---

## Prior Experiment Summary

**What we thought**: RePO vs AdaRePO on MolOpt/LogP
**What actually happened**: Both methods trained on **mixed LogP+MR+QED data (1500 samples)** but the reward function was **hardcoded to compute LogP improvement** for all samples, regardless of the prompt's intended property. This is now fixed.

| Item | Value |
|------|-------|
| RePO WandB | https://wandb.ai/adrian-caom/RePO-OpenMolIns-3B/runs/mpw42aej |
| AdaRePO WandB | https://wandb.ai/adrian-caom/AdaRePO-OpenMolIns-3B/runs/njz39gku |
| Both used | 1500 train samples (500 LogP + 500 MR + 500 QED) |
| Reward bug | `property_name="logP"` hardcoded in both `repo.py` and `ada_repo.py` |
| Bug status | **FIXED** — now reads `property_name` from config YAML |

---

## New Experiment Matrix (6 clean runs)

| Experiment | Subtask | Method | Data | Property | Steps (est.) |
|------------|---------|--------|------|----------|------------|
| repo_LogP | LogP | RePO | 500 | logP | ~64 |
| ada_repo_LogP | LogP | AdaRePO | 500 | logP | ~64 |
| repo_MR | MR | RePO | 500 | mr | ~64 |
| ada_repo_MR | MR | AdaRePO | 500 | mr | ~64 |
| repo_QED | QED | RePO | 500 | qed | ~64 |
| ada_repo_QED | QED | AdaRePO | 500 | qed | ~64 |

Steps calculation: 500 samples / (2 batch × 3 processes) ≈ 84 samples/step... actually:
- effective_batch = per_device_batch × num_processes × grad_accum × num_generations = 2 × 3 × 16 × 3 = 288
- But with 500 samples and 4 epochs: ~500/batch_per_step × 4 ≈ steps depends on exact batching
- The previous run with 1500 samples did 184 steps (46/epoch), so 500 samples → ~16 steps/epoch × 4 = ~64 steps

---

## Execution Plan

### Phase 1: Audit + Fix (DONE)
- [x] Verified previous training case from code, config, and logs
- [x] Found and fixed hardcoded `property_name="logP"` bug
- [x] Generalized dataset loading to accept single-subtask selections
- [x] Created 6 per-subtask config files (3 RePO + 3 AdaRePO)
- [x] Smoke-tested all 3 reward functions on real data
- [x] Created unified launcher script
- [x] Created run manifest and dataset inventory

### Phase 2: Pilot Runs
- [ ] Submit 10-step pilot for repo_MR (debug-scaling, 1hr)
- [ ] Submit 10-step pilot for ada_repo_MR (debug-scaling, 1hr)
- [ ] Verify logs show correct property name and reward values
- [ ] If clean, proceed to Phase 3

**Commands:**
```bash
cd /lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/scripts
./launch_experiment.sh repo MR debug-scaling 10
./launch_experiment.sh ada_repo MR debug-scaling 10
```

### Phase 3: Full Comparison Runs
- [ ] Submit all 6 full runs (preemptable, 6hr walltime)
- [ ] Monitor via logs and WandB

**Commands:**
```bash
cd /lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/scripts
./launch_experiment.sh repo LogP preemptable
./launch_experiment.sh ada_repo LogP preemptable
./launch_experiment.sh repo MR preemptable
./launch_experiment.sh ada_repo MR preemptable
./launch_experiment.sh repo QED preemptable
./launch_experiment.sh ada_repo QED preemptable
```

### Phase 4: Analysis
- [ ] Run per-subtask analysis (same style as original LogP analysis)
- [ ] Run cross-subtask comparison
- [ ] Generate final summary report

**Command:**
```bash
python analysis/cross_subtask_analysis.py --subtasks LogP MR QED
```

---

## Polaris Queue Strategy

| Phase | Queue | Walltime | Nodes | Purpose |
|-------|-------|----------|-------|---------|
| Smoke test | login node | N/A | 0 | Data/reward verification (DONE) |
| Pilot | debug-scaling | 1:00:00 | 1 | 10-step sanity check |
| Full | preemptable | 6:00:00 | 1 | Complete training |

- **Checkpoint cadence**: `save_strategy: "epoch"` + `save_steps: 50`
- **Resume**: automatic via `get_last_checkpoint()` in both `repo.py` and `ada_repo.py`
- **Preemption safety**: checkpoints every epoch (~16 steps) means max loss is ~16 steps on preemption

---

## Key Differences from Prior Run

| Parameter | Prior Run | New Runs |
|-----------|-----------|----------|
| subtask_selection | ["LogP", "MR", "QED"] | ["LogP"] or ["MR"] or ["QED"] |
| property_name | "logP" (hardcoded) | per-subtask from config |
| Training samples | 1500 (mixed) | 500 (single subtask) |
| Steps per epoch | ~46 | ~16 |
| Total steps | ~184 | ~64 |
| save_steps | epoch only | epoch + every 50 steps |
| wandb_project | per-method | unified "RePO-MolOpt-3B" / "AdaRePO-MolOpt-3B" |

---

## Files Created/Modified

### Modified
- `RePO/src/x_r1/repo.py` — added `property_name` arg, fixed reward wiring, generalized subtask loading
- `adaptive_repo/ada_repo.py` — same fixes

### New Configs
- `RePO/recipes/Polaris_3B_LogP.yaml`
- `RePO/recipes/Polaris_3B_MR.yaml`
- `RePO/recipes/Polaris_3B_QED.yaml`
- `adaptive_repo/configs/ada_repo_3B_LogP.yaml`
- `adaptive_repo/configs/ada_repo_3B_MR.yaml`
- `adaptive_repo/configs/ada_repo_3B_QED.yaml`

### New Scripts
- `adaptive_repo/scripts/launch_experiment.sh` — unified PBS launcher
- `adaptive_repo/smoke_test_rewards.py` — reward function smoke test (PASSED)
- `adaptive_repo/analysis/cross_subtask_analysis.py` — cross-subtask comparison
- `adaptive_repo/analysis/dataset_audit.md` — full audit report
- `adaptive_repo/analysis/dataset_inventory.csv` — dataset inventory
- `adaptive_repo/analysis/run_manifest.csv` — experiment manifest
- `adaptive_repo/analysis/EXPERIMENT_PLAN.md` — this file

---

## MR Reward Scale Note

MR (Molar Refractivity) values are much larger than LogP or QED (typical range 30-200 vs 0-5 for LogP or 0-1 for QED). The reward function computes `improvement_score = ref_value - gen_value` (or vice versa), so MR rewards will have larger magnitude. This is expected and does not break training, but means raw reward numbers are not directly comparable across subtasks. The relative comparison (RePO vs AdaRePO on the same subtask) remains valid.
