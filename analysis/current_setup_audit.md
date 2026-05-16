# Phase 0 — Current Setup Audit

**Date:** 2026-03-13
**Author:** Cascade (automated audit)

---

## A. Confirmed Current Setup

### 1. Model & Base Checkpoint
- **Base model:** Qwen2.5-3B-Instruct (`/lus/eagle/projects/IMPROVE_Aim1/caom/.cache/huggingface/Qwen2.5-3B-Instruct`)
- **Precision:** bfloat16, SDPA attention
- **No LoRA/PEFT:** Full fine-tuning

### 2. Training Data
- **Source:** `data/OpenMolIns/light/train.csv` (relative to RePO root)
- **Total rows:** 4500 (9 subtasks × 500 each)
- **Per-subtask breakdown:**
  | SubTask | Count |
  |---------|-------|
  | AddComponent | 500 |
  | AtomNum | 500 |
  | BondNum | 500 |
  | DelComponent | 500 |
  | FunctionalGroup | 500 |
  | **LogP** | **500** |
  | **MR** | **500** |
  | **QED** | **500** |
  | SubComponent | 500 |
- **Filtering:** Each single-subtask run filters to exactly **500 training examples**

### 3. Test Data
- **Source:** `data/benchmarks/open_generation/MolOpt/{LogP,MR,QED}/test.csv`
- **Size:** 5000 rows per subtask (headers + 5000 data rows)

### 4. Training Hyperparameters (Shared)

| Parameter | Value |
|-----------|-------|
| per_device_train_batch_size | 2 |
| gradient_accumulation_steps | 16 |
| num_processes (training GPUs) | 3 |
| num_generations (G) | 3 |
| learning_rate | 5e-6 |
| lr_scheduler | cosine |
| warmup_ratio | 0.15 |
| num_train_epochs | 4 |
| max_steps | -1 (epoch-controlled) |
| seed | 42 |
| use_vllm | true |
| vllm_gpu_memory_utilization | 0.6 |
| max_prompt_length | 256 |
| max_completion_length | 512 |
| save_strategy | epoch |

**Effective batch size:** 2 × 16 = 32 samples/step (grad accum on each process)
**Steps per epoch:** ~15 (500 / 32 ≈ 15.6, rounded)
**Total steps:** 60 (confirmed from all 6 trainer_state.json files)
**Epochs completed:** ~3.96 (60 steps × 0.066 epoch/step)

### 5. Reward Function

All 6 runs use the **same** reward function: `smile_optimization`

| Parameter | LogP | MR | QED |
|-----------|------|-----|-----|
| property_name | logP | mr | qed |
| target_direction | increase | increase | increase |
| similarity_weight | 0.3 | 0.3 | 0.3 |
| property_weight | 0.7 | 0.7 | 0.7 |
| min_similarity | 0.1 | 0.1 | 0.1 |
| validity_weight | 1.0 | 1.0 | 1.0 |
| reward_mode | average | average | average |

### 6. RePO vs AdaRePO — Exact Differences

**RePO (baseline):**
- Entry: `RePO/src/x_r1/repo.py` → uses `XGRPOTrainer`
- Loss: `loss = L_RL + s_loss` (beta_guide implicitly = 1.0)
- No dynamic beta, no memory bank, no ensemble

**AdaRePO:**
- Entry: `adaptive_repo/ada_repo.py` → uses `AdaRePOTrainer`
- Loss: `loss = L_RL + beta_guide_mean * s_loss`
- Dynamic beta controller with:
  | Parameter | Value |
  |-----------|-------|
  | beta_guide_mode | sigmoid_gap |
  | beta_guide_max | 1.0 |
  | beta_guide_alpha | 5.0 |
  | beta_guide_top_k_frac | 0.33 |
  | beta_guide_softmax_tau | 1.0 |
  | use_memory_bank | **false** |
  | use_ensemble | **false** |

**Key observation:** Memory bank and ensemble are both **disabled**. The only active AdaRePO feature is the dynamic beta controller (sigmoid_gap mode). This means AdaRePO = RePO + adaptive guidance scaling.

### 7. Checkpoints Used for Evaluation

All evaluations used **checkpoint-60** (final checkpoint, end of epoch 4):

| Model | Checkpoint Path |
|-------|----------------|
| repo_LogP | `/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output/repo_3B_LogP/checkpoint-60` |
| repo_MR | `/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output/repo_3B_MR/checkpoint-60` |
| repo_QED | `/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output/repo_3B_QED/checkpoint-60` |
| ada_repo_LogP | `/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/output/ada_repo_3B_LogP/checkpoint-60` |
| ada_repo_MR | `/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/output/ada_repo_3B_MR/checkpoint-60` |
| ada_repo_QED | `/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/output/ada_repo_3B_QED/checkpoint-60` |

Intermediate checkpoints available: 15, 30, 45, 60 (per-epoch saves).

### 8. WandB Runs & PBS Jobs

**Training PBS jobs (all completed Mar 10):**
| Run | PBS Job ID |
|-----|------------|
| repo_3B_LogP | 6958401 |
| ada_repo_3B_LogP | 6958402 |
| repo_3B_MR | 6958403 |
| ada_repo_3B_MR | 6958404 |
| repo_3B_QED | 6958405 |
| ada_repo_3B_QED | 6958406 |

**WandB projects:**
- RePO runs: `RePO-MolOpt-3B`
- AdaRePO runs: `AdaRePO-MolOpt-3B`

**Evaluation PBS job:** 6959360 (completed Mar 11)

---

## B. Evidence — Training Dynamics Summary

### Loss Trajectories (step 1 → step 60)

| Model | First Loss | Last Loss | Mean Loss |
|-------|-----------|----------|-----------|
| repo_LogP | 1.168 | 0.367 | 0.537 |
| repo_MR | 1.120 | 0.332 | 0.484 |
| repo_QED | 1.440 | 0.412 | 0.518 |
| ada_LogP | 0.653 | 0.192 | 0.288 |
| ada_MR | 0.554 | 0.273 | 0.255 |
| ada_QED | 0.476 | 0.153 | 0.231 |

**Note:** AdaRePO losses are lower because beta_guide < 1.0 scales down the guidance loss component. This does NOT mean AdaRePO trains better — it means the loss landscape is different due to the adaptive weighting.

### Reward Trajectories

| Model | First Reward | Last Reward | Mean Reward |
|-------|-------------|-------------|-------------|
| repo_LogP | 0.160 | 0.198 | 0.182 |
| repo_MR | 1.743 | 0.474 | 0.977 |
| repo_QED | 0.148 | 0.097 | 0.107 |
| ada_LogP | 0.160 | 0.301 | 0.210 |
| ada_MR | 1.743 | -0.174 | 1.407 |
| ada_QED | 0.148 | 0.110 | 0.138 |

**Key observations:**
- **MR rewards are on a fundamentally different scale** (~1.7 at start vs ~0.15 for LogP/QED). This is because MR (Molar Refractivity) values are numerically larger than logP or QED values.
- **ada_MR shows reward COLLAPSE** (1.743 → -0.174), while repo_MR also declines but stays positive (1.743 → 0.474). Both drop from high initial rewards, but AdaRePO drops much more.
- LogP rewards increase for both methods, with AdaRePO showing a slightly higher final reward.
- QED rewards are low and roughly comparable.

### Beta Dynamics (AdaRePO only)

| Model | First Beta | Last Beta | Mean Beta |
|-------|-----------|----------|-----------|
| ada_LogP | 0.696 | 0.603 | 0.666 |
| ada_MR | 0.795 | 0.723 | 0.765 |
| ada_QED | 0.479 | 0.486 | 0.503 |

**Key observations:**
- Beta values are high (0.5–0.8), meaning the controller is keeping guidance relatively strong throughout training.
- Beta for MR is the highest (~0.76), suggesting the model struggles to outperform the reference on MR and the controller compensates by keeping guidance strong.
- Beta for QED is lowest (~0.50), meaning the controller has partially switched away from guidance — but this hasn't helped.
- **Beta values are NOT adapting dramatically across training** — they stay in a narrow band. The sigmoid_gap controller may not be providing enough dynamic range given these reward scales.

### KL Divergence

| Model | Last KL |
|-------|---------|
| repo_LogP | 0.279 |
| repo_MR | 0.563 |
| repo_QED | 0.499 |
| ada_LogP | 0.145 |
| ada_MR | 0.331 |
| ada_QED | 0.268 |

AdaRePO consistently has lower KL than RePO, which is expected because the reduced guidance loss (beta < 1) means less aggressive policy change.

---

## C. Evaluation Results (checkpoint-60)

| Model | Subtask | Samples | Success Rate | Validity | Similarity |
|-------|---------|---------|-------------|----------|------------|
| repo | LogP | 5000 | 0.196 | 0.607 | 0.832 |
| ada_repo | LogP | 5000 | **0.235** | 0.599 | 0.799 |
| repo | MR | 1000 | **0.209** | **0.674** | 0.867 |
| ada_repo | MR | 1000 | 0.135 | 0.622 | 0.886 |
| repo | QED | 1000 | **0.130** | **0.711** | 0.874 |
| ada_repo | QED | 1000 | 0.100 | 0.695 | 0.878 |

---

## D. Ambiguities & Concerns

### D.1 Reward Scale Mismatch
MR rewards are ~10× larger than LogP/QED rewards. The sigmoid_gap beta controller uses **raw reward gaps** (`v_top - v_ref`) without normalization. This means:
- For MR: gap is large → sigmoid(-alpha * gap) saturates → beta stays high or low depending on sign
- For LogP/QED: gap is small → sigmoid stays near 0.5

**This is the most likely cause of miscalibration.** The beta controller is effectively operating in different regimes for different tasks, despite using the same alpha=5.0.

### D.2 MR Reward Collapse
ada_MR's reward drops to -0.174 by step 60. This suggests the model is generating molecules that are valid but have WORSE MR than the reference. The high beta (~0.76) means it's still being strongly guided by the reference, but the RL loss may be fighting the guidance direction.

### D.3 Only Checkpoint-60 Evaluated
We have checkpoints at 15, 30, 45, 60 but only evaluated at 60. It's possible AdaRePO peaks earlier and degrades, while RePO is more monotonic. **Phase 1 will directly test this.**

### D.4 Memory Bank Disabled
The memory bank (self-distillation) feature that was designed to improve reference quality over time is disabled. This means AdaRePO is only getting the single adaptive beta benefit, not the full method.

### D.5 num_train_epochs vs max_steps
Configs say `max_steps: -1` and `num_train_epochs: 4`, but all trainer_states show `max_steps: 60`. This appears correct — the trainer resolves epoch-based training to 60 total steps given 500 samples and batch size 32.

---

## E. Reproducibility Assessment

### Can current results be reproduced as-is?
**YES, with caveats:**

1. ✅ All configs, code, and checkpoints are present and documented
2. ✅ Launch script (`scripts/launch_experiment.sh`) can regenerate any run
3. ✅ Evaluation script (`scripts/run_evaluation.sh`) can re-evaluate any checkpoint
4. ✅ Seeds are fixed (42)
5. ⚠️ vLLM generation may introduce minor non-determinism (GPU scheduling)
6. ⚠️ PBS preemptable queue means runs could be interrupted
7. ✅ All 6 training runs completed successfully with matching step counts

### Files required for reproduction:
- **RePO configs:** `RePO/recipes/Polaris_3B_{LogP,MR,QED}.yaml`
- **AdaRePO configs:** `adaptive_repo/configs/ada_repo_3B_{LogP,MR,QED}.yaml`
- **Training code:** `RePO/src/x_r1/repo.py`, `adaptive_repo/ada_repo.py`
- **Trainer code:** `RePO/src/x_r1/x_repo_trainer.py`, `adaptive_repo/ada_repo_trainer.py`
- **Beta controller:** `adaptive_repo/dynamic_beta.py`
- **Reward function:** `RePO/src/x_r1/rewards.py` → `get_smile_optimization_reward`
- **Launch script:** `adaptive_repo/scripts/launch_experiment.sh`
- **Evaluation scripts:** `RePO/generate_predictions.py`, `RePO/evaluate.py`
- **Accelerate config:** `RePO/recipes/zero3_polaris.yaml`

---

## F. Summary of Key Findings for Phase 1 Design

1. **60 steps (4 epochs) may be insufficient** — especially for MR where reward collapses
2. **Reward scales differ dramatically** — MR ~10× larger than LogP/QED
3. **Beta controller operates in a narrow band** (0.5–0.8) and may not be adapting enough
4. **MR shows reward collapse in AdaRePO** — this needs investigation
5. **Intermediate checkpoints (15, 30, 45) exist** and can provide learning curve data without new training
6. **Memory bank is disabled** — enabling it could be a Phase 3 intervention
