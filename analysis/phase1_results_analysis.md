# AdaRePO vs RePO — Paper-Aligned Comparison Report

> Last updated: 2026-03-23
> This is the single running report for the study. All phases append here.
> Goal: Reproduce RePO paper's main experimental coverage, comparing RePO vs AdaRePO.

---

# A. Paper-Aligned Experiment Inventory

The RePO paper's main results are in Tables 1, 2, and 10.

## Table 1 — Single-objective TOMG-Bench (6 tasks)

Table 1 reports RePO (Qwen2.5-3B) on all 6 TOMG-Bench open_generation tasks.
Two distinct training setups produce Table 1:

| Task group | Tasks | Reward function | Train data | Paper config |
|-----------|-------|----------------|------------|-------------|
| **MolEdit** (structural) | AddComponent, DelComponent, SubComponent | `structure_optimization` | `data/structural_opt_light.json` (1496 examples: ~498/task) | `OpenMolIns_3B_config_structure.yaml` |
| **MolOpt** (property) | LogP, MR, QED | `smile_optimization` | `data/OpenMolIns/light/train.csv` (500/task) | `Polaris_3B_{LogP,MR,QED}.yaml` (single-task) or `OpenMolIns_3B_config.yaml` (mixed) |

**Critical paper ambiguity**: The paper may train MolOpt tasks jointly (1500 examples, all 3 subtasks) or per-task (500 each). Our Polaris configs train per-task. The original `OpenMolIns_3B_config.yaml` trains all 3 jointly but hardcodes `property_name: "qed"`, which is a bug for LogP/MR.

| Row | Dataset path on disk | Exists | Train size | Reward function |
|-----|---------------------|--------|-----------|----------------|
| AddComponent | `data/structural_opt_light.json` | ✅ | 498 | `structure_optimization` |
| DelComponent | same JSON | ✅ | 498 | `structure_optimization` |
| SubComponent | same JSON | ✅ | 500 | `structure_optimization` |
| LogP | `data/OpenMolIns/light/train.csv` → filter SubTask=LogP | ✅ | 500 | `smile_optimization` (property_name=logP) |
| MR | same CSV → filter SubTask=MR | ✅ | 500 | `smile_optimization` (property_name=mr) |
| QED | same CSV → filter SubTask=QED | ✅ | 500 | `smile_optimization` (property_name=qed) |

**Eval benchmarks**: All 6 exist at `data/benchmarks/open_generation/{MolEdit,MolOpt}/{subtask}/test.csv`.
Eval code: `generate_predictions.py` + `evaluate.py` — supports all 6 tasks.

## Table 2 — Multi-objective MuMOInstruct (3 combos × seen/unseen)

| Setting | Train data | Exists | Train size | Reward | Eval data (seen) | Eval data (unseen) |
|---------|-----------|--------|-----------|--------|------------------|-------------------|
| BDP (bbbp+drd2+plogp) | `TRAIN_multi_prop/IND_sft_train_data_bbbp+drd2+plogp.json` | ✅ | 1500 | `multi_prop_optimization` | `TEST_multi_prop/IND_sft_seen_bbbp+drd2+plogp_test_data.json` (500) | `IND_sft_unseen_...` (500) |
| BDQ (bbbp+drd2+qed) | `TRAIN_multi_prop/IND_sft_train_data_bbbp+drd2+qed.json` | ✅ | 1500 | `multi_prop_optimization` | analogous seen (500) | analogous unseen (500) |
| BPQ (bbbp+plogp+qed) | `TRAIN_multi_prop/IND_sft_train_data_bbbp+plogp+qed.json` | ✅ | 1500 | `multi_prop_optimization` | analogous seen (500) | analogous unseen (500) |

**Config**: `MulProp_3B_config.yaml` — uses `variant: mumo`, `num_train_epochs: 1`, `warmup_ratio: 0.1`.

**Critical dependency**: The multi-prop reward function (`rewards_mumo.py`) and evaluation (`mumo_evaluate.py`) call **external ADMET and DRD2 API servers** at `localhost:10086` and `localhost:10087`. Server code is in `multiprop_utils/admetModel_api.py` and `drd2Model_api.py`. These must be running during both training AND evaluation.

## Table 10 — Multi-seed stability (LogP/QED/MR)

Table 10 reports mean ± std across multiple random seeds for the property optimization tasks. Same setup as Table 1 MolOpt rows, but repeated with different seeds.

---

# B. Table 1 Status — Which Cases Are Done / Partial / Missing

## RePO side

| Task | Training | Eval | Status | Notes |
|------|---------|------|--------|-------|
| **AddComponent** | ❌ not done | ❌ | **missing** | Need to train with `structure_optimization` reward |
| **DelComponent** | ❌ not done | ❌ | **missing** | Same training run as AddComponent (shared JSON) |
| **SubComponent** | ❌ not done | ❌ | **missing** | Same training run |
| **LogP** | ✅ 60-step run exists (`repo_3B_LogP`) | ✅ at ckpt-60 (SR=0.196) | **partial** | Only 60 steps; paper likely trains longer. No 120-step run. |
| **MR** | ✅ 60/120/240 steps | ✅ 20 evals | **done** | Best: SR=0.220 at s120/ckpt-120 |
| **QED** | ✅ 60/120/240 steps | ✅ 20 evals | **done** | Best: SR=0.116 at s120/ckpt-120 |

## AdaRePO side

| Task | Training | Eval | Status | Notes |
|------|---------|------|--------|-------|
| **AddComponent** | ❌ not done | ❌ | **missing** | `ada_repo.py` lacks structural data loading + multi-prop reward |
| **DelComponent** | ❌ not done | ❌ | **missing** | Same gap |
| **SubComponent** | ❌ not done | ❌ | **missing** | Same gap |
| **LogP** | ✅ 60-step run exists (`ada_repo_3B_LogP`) | ✅ at ckpt-60 (SR=0.235) | **partial** | Only 60 steps |
| **MR** | ✅ 60/120/240 steps | ✅ 20 evals | **done** | Best: SR=0.230 at s120/ckpt-120 |
| **QED** | ✅ 60/120/240 steps | ✅ 20 evals | **done** | Best: SR=0.150 at s240/ckpt-240 |

## Summary: Table 1 gap count

| | Done | Partial | Missing |
|-|------|---------|---------|
| RePO | 2 (MR, QED) | 1 (LogP) | 3 (Add/Del/Sub) |
| AdaRePO | 2 (MR, QED) | 1 (LogP) | 3 (Add/Del/Sub) |
| **Total** | 4/12 | 2/12 | 6/12 |

---

# C. Table 2 Status — Multi-Objective MuMOInstruct

| Setting | RePO training | RePO eval | AdaRePO training | AdaRePO eval | Status |
|---------|-------------|----------|-----------------|-------------|--------|
| BDP seen | ❌ | ❌ | ❌ | ❌ | **missing** |
| BDP unseen | ❌ | ❌ | ❌ | ❌ | **missing** |
| BDQ seen | ❌ | ❌ | ❌ | ❌ | **missing** |
| BDQ unseen | ❌ | ❌ | ❌ | ❌ | **missing** |
| BPQ seen | ❌ | ❌ | ❌ | ❌ | **missing** |
| BPQ unseen | ❌ | ❌ | ❌ | ❌ | **missing** |

**Total: 0/12 done.**

### Code/infrastructure gaps for Table 2

1. **`ada_repo.py`**: No `multi_prop_optimization` reward registered, no mumo dataset loading
2. **ADMET/DRD2 API servers**: Must be launched alongside training/eval on Polaris compute nodes
3. **MuMO prediction generation**: No dedicated generation script found (unlike `generate_predictions.py` for TOMG-Bench). The `mumo_evaluate.py` expects a JSON with `vllm_output` field — need to verify how predictions are generated.
4. **Environment**: `mumo_requirements.txt` lists `admet_ai==1.3.1` and `chemprop==1.6.1` — may need separate env or additions to `repo_env`

---

# D. Table 10 Status — Multi-Seed Stability

| Task | RePO seeds done | AdaRePO seeds done | Status |
|------|----------------|-------------------|--------|
| LogP | 1 (seed=42) | 1 (seed=42) | **missing** — need 3+ seeds |
| MR | 1 (seed=42) | 1 (seed=42) | **missing** — need 3+ seeds |
| QED | 1 (seed=42) | 1 (seed=42) | **missing** — need 3+ seeds |

**Total: 0/6 conditions done** (each condition = multiple seeds).
Current seed=42 runs can be reused as one of the seeds.

---

# E. Exact Gaps to Close

## Gap 1 — MolEdit structural tasks (Table 1, Priority A)

**What's needed**:
- 1 RePO training run with `structure_optimization` reward on `structural_opt_light.json` (1496 examples)
- 1 AdaRePO training run — same data, same reward
- Eval both on AddComponent, DelComponent, SubComponent test sets

**Code changes needed for AdaRePO**:
- Add structural data loading to `ada_repo.py` (mirror `repo.py` lines 240-254)
- `structure_optimization` reward is already imported in `ada_repo.py` and registered

**Estimated compute**: 2 training jobs × ~3h each + 1 eval job = ~7 node-hours

## Gap 2 — LogP extended training (Table 1, Priority A)

**What's needed**:
- 1 RePO LogP training run at 120 steps (new, or extend from ckpt-60)
- 1 AdaRePO LogP training run at 120 steps
- Eval both at ckpt-60 and ckpt-120

**No code changes needed** — existing per-task configs work. Just need to set `--max_steps 120`.

**Estimated compute**: 2 training jobs × ~2h + 1 eval job = ~5 node-hours

## Gap 3 — MuMOInstruct (Table 2, Priority B)

**What's needed**:
- 6 training runs: {RePO, AdaRePO} × {BDP, BDQ, BPQ}
- 12 evaluations: 6 runs × {seen, unseen}
- ADMET/DRD2 API servers running on compute nodes

**Code changes needed for AdaRePO**:
- Add `multi_prop_optimization` reward import + registration to `ada_repo.py`
- Add mumo dataset loading logic (mirror `repo.py` lines 255-282)
- Verify prediction generation pipeline for mumo_evaluate.py

**Critical blocker**: Need to verify ADMET/DRD2 API setup works on Polaris.
**Estimated compute**: 6 training × ~2h + 12 evals × ~0.5h = ~18 node-hours

## Gap 4 — Multi-seed stability (Table 10, Priority C)

**What's needed**:
- Repeat LogP/MR/QED training with seeds {42, 123, 456} (or similar) for both methods
- seed=42 already done → 4 new seeds × 6 tasks = ~24 runs if using 5 seeds
- Minimum viable: 3 seeds total → 2 new seeds × 6 = 12 new runs

**No code changes needed** — just vary `--seed` in launch scripts.

**Estimated compute**: 12 training × ~2h + 12 evals = ~30 node-hours

---

# F. Setup & Configuration Reference

All values verified from YAML config files on disk, not inferred.

## Shared hyperparameters (MolOpt per-task configs)

| Parameter | Value | Source |
|-----------|-------|--------|
| Base model | `Qwen2.5-3B-Instruct` | All YAML configs |
| Training data | `data/OpenMolIns/light/train.csv` | `data_scale: light` |
| Train examples per subtask | 500 | Verified: `wc -l` + SubTask counts |
| Reward function | `smile_optimization` (weight 1.0) | All MolOpt configs |
| Reward params | `similarity_weight=0.3`, `property_weight=0.7`, `min_similarity=0.1` | All configs |
| `per_device_train_batch_size` | 2 | All configs |
| `gradient_accumulation_steps` | 16 | Polaris configs |
| `num_processes` | 3 (GPUs 0-2 train, GPU 3 vLLM) | Polaris configs |
| `num_generations` | 3 | Polaris configs |
| `learning_rate` | 5e-6 | All configs |
| `lr_scheduler_type` | cosine | All configs |
| `warmup_ratio` | 0.15 | MolOpt configs (0.1 for MuMO) |
| `seed` | 42 | All configs |
| Effective batch per step | ~33 prompts (empirical) | trainer_state.json |

## Paper vs Polaris config differences

| Parameter | Paper config (`OpenMolIns_3B_config.yaml`) | Polaris config (`Polaris_3B_{task}.yaml`) |
|-----------|------------------------------------------|------------------------------------------|
| `num_processes` | 2 | 3 |
| `num_generations` | 4 | 3 |
| `gradient_accumulation_steps` | 8 | 16 |
| `max_completion_length` | 1024 | 512 |
| `attn_implementation` | flash_attention_2 | sdpa |
| `vllm_gpu_memory_utilization` | 0.8 | 0.6 |
| subtask_selection | `["LogP", "MR", "QED"]` (joint) | single-task per config |

**Assessment**: These are **not identical** to the paper setup. The Polaris configs were adapted for 4×A100-40GB constraints. Key differences:
- **num_generations 3 vs 4**: affects exploration diversity
- **grad_accum 16 vs 8 with different num_processes**: changes effective batch size
- **max_completion_length 512 vs 1024**: may truncate longer SMILES reasoning
- **single-task vs joint training**: paper may train on all 3 MolOpt tasks together

These differences mean our results **approximate** but do not exactly reproduce the paper. For the RePO-vs-AdaRePO comparison, this is acceptable since both use the same Polaris config.

## AdaRePO-only parameters

| Parameter | Value |
|-----------|-------|
| `beta_guide_mode` | `sigmoid_gap` |
| `beta_guide_max` | 1.0 |
| `beta_guide_alpha` | 5.0 |
| `beta_guide_top_k_frac` | 0.33 |
| `use_memory_bank` | false |
| `use_ensemble` | false |

---

# G. Prior Results (for reference)

## Phase 0 baseline (60 steps, checkpoint-60)

| Method | Subtask | Success Rate | Validity | Similarity |
|---------|---------|-------------|----------|------------|
| RePO | LogP | 0.196 | 0.607 | 0.832 |
| AdaRePO | LogP | 0.235 | 0.599 | 0.799 |
| RePO | MR | 0.209 | 0.674 | 0.867 |
| AdaRePO | MR | 0.135 | 0.622 | 0.886 |
| RePO | QED | 0.130 | 0.711 | 0.874 |
| AdaRePO | QED | 0.100 | 0.695 | 0.878 |

## Phase 1 best per method/subtask

| Method | Subtask | Best SR | At | SR×Sim |
|---------|---------|---------|-----|--------|
| RePO | MR | 0.220 | s120/ckpt-120 | 0.183 |
| AdaRePO | MR | **0.230** | s120/ckpt-120 | **0.193** |
| RePO | QED | 0.116 | s120/ckpt-120 | 0.101 |
| AdaRePO | QED | **0.150** | s240/ckpt-240 | **0.126** |

---

# H. Execution Plan

(To be filled after user approves the gap analysis and run priorities.)
