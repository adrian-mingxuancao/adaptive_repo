# Dataset & Pipeline Audit: AdaRePO vs RePO Experiments

**Date**: 2026-03-10
**Author**: Cascade (automated audit)

---

## A. Confirmed Previous Training Case

### Conclusion

The previous RePO and AdaRePO runs trained on **OpenMolIns "light" scale**, using **all three MolOpt subtasks (LogP + MR + QED) simultaneously** in a single mixed dataset. The reward function, however, was **hardcoded to compute only LogP property improvement** regardless of the actual subtask of each sample. This means:

- For the 500 LogP samples: reward was correct (LogP improvement + similarity)
- For the 500 MR samples: reward was **incorrect** — it computed LogP improvement when MR improvement was intended
- For the 500 QED samples: reward was **incorrect** — it computed LogP improvement when QED improvement was intended

**The previous comparison is therefore a "LogP-reward on mixed-subtask data" experiment, not a clean single-property benchmark.**

### Evidence

| Evidence Item | Source | Value |
|---|---|---|
| Dataset path | `repo.py` line 273, `ada_repo.py` line 142 | `data/OpenMolIns/light/train.csv` |
| Subtask filter | Config YAML line 19 | `subtask_selection: ["LogP", "MR", "QED"]` |
| Filtered dataset size | RePO log, AdaRePO log | 1500 examples (500 per subtask) |
| Reward function | `repo.py` line 434, `ada_repo.py` line 212 | `get_smile_optimization_reward(property_name="logP", ...)` |
| Config `property_name` | Both YAML configs line 99/122 | `property_name: "qed"` — **UNUSED** |
| Actual property in logs | `grep "Property:" <log>` | `Property: logP` (confirmed from both logs) |
| Direction inference | `rewards.py` line 910-916 | Inferred from prompt text ("increase"/"decrease"/"higher"/"lower") — this IS correct per-sample |
| Reference SMILES | `rewards.py` line 874-908 | Extracted from prompt — this IS correct per-sample |
| Train split | Code | `train` only, no validation or test split used |
| Eval split | Config | `do_eval: false`, `eval_strategy: "no"` |
| RePO WandB | Log | `https://wandb.ai/adrian-caom/RePO-OpenMolIns-3B/runs/mpw42aej` |
| AdaRePO WandB | Log | `https://wandb.ai/adrian-caom/AdaRePO-OpenMolIns-3B/runs/njz39gku` |
| RePO checkpoint | Filesystem | `/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output/repo_polaris_3B/checkpoint-184` |
| AdaRePO checkpoint | Filesystem | `/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/output/ada_repo_3B/checkpoint-184` |
| RePO PBS | Filesystem | `/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/submit_repo_polaris.pbs` |
| AdaRePO PBS | Filesystem | `/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/submit_ada_repo_polaris.pbs` |
| Model | Config | Qwen2.5-3B-Instruct |
| DeepSpeed | Config | ZeRO-3 with CPU offload (`recipes/zero3_polaris.yaml`) |

### Ambiguity / Mismatch Found

1. **CRITICAL**: `property_name: "qed"` in config YAML is **never passed** to `get_smile_optimization_reward()`. The reward registry in both `repo.py` (line 434) and `ada_repo.py` (line 212) hardcodes `property_name="logP"`. The config field is dead code.

2. **Mixed-subtask training**: The model sees LogP, MR, and QED prompts, but the reward function always evaluates LogP improvement. For MR/QED samples, the reward signal is nonsensical (LogP change on molecules where the prompt asks for MR/QED change).

3. **No validation/test evaluation**: `do_eval: false` in both configs. Training metrics are the only signal.

4. **No TOMG-Bench provenance**: The data file `data/OpenMolIns/light/train.csv` appears to be derived from the OpenMolIns dataset, which is the training dataset companion to TOMG-Bench. It is NOT the TOMG-Bench test set. TOMG-Bench test sets exist separately in `data/benchmarks/open_generation/MolOpt/{LogP,MR,QED}/test.csv`.

### Reproducibility

The previous run IS reproducible as-is, with one caveat: the reward function bug means the results reflect "LogP-reward on mixed data", not "per-subtask property optimization". This is a valid experiment but should be labeled correctly.

---

## B. Dataset Inventory

### Training Data: OpenMolIns (light scale)

| File | Path | Rows | Columns | Subtasks |
|---|---|---|---|---|
| train.csv | `RePO/data/OpenMolIns/light/train.csv` | 4500 | SubTask, Instruction, molecule | 9 subtasks × 500 each |

Subtask breakdown:
- **MolCustom**: AtomNum (500), BondNum (500), FunctionalGroup (500)
- **MolEdit**: AddComponent (500), SubComponent (500), DelComponent (500)
- **MolOpt**: LogP (500), MR (500), QED (500)

### TOMG-Bench Test Sets

| Task Family | Subtask | Test File | Rows | Columns | Has Ground Truth |
|---|---|---|---|---|---|
| MolCustom | AtomNum | `benchmarks/open_generation/MolCustom/AtomNum/test.csv` | varies | Instruction | No |
| MolCustom | BondNum | `benchmarks/open_generation/MolCustom/BondNum/test.csv` | varies | Instruction | No |
| MolCustom | FunctionalGroup | `benchmarks/open_generation/MolCustom/FunctionalGroup/test.csv` | varies | Instruction | No |
| MolEdit | AddComponent | `benchmarks/open_generation/MolEdit/AddComponent/test.csv` | varies | Instruction, molecule | Yes (molecule) |
| MolEdit | SubComponent | `benchmarks/open_generation/MolEdit/SubComponent/test.csv` | varies | Instruction, molecule | Yes (molecule) |
| MolEdit | DelComponent | `benchmarks/open_generation/MolEdit/DelComponent/test.csv` | varies | Instruction, molecule | Yes (molecule) |
| MolOpt | LogP | `benchmarks/open_generation/MolOpt/LogP/test.csv` | 5000 | index, Instruction, molecule, logP | Yes (molecule + property) |
| MolOpt | MR | `benchmarks/open_generation/MolOpt/MR/test.csv` | 5000 | index, Instruction, molecule, MR | Yes (molecule + property) |
| MolOpt | QED | `benchmarks/open_generation/MolOpt/QED/test.csv` | 5000 | index, Instruction, molecule, QED | Yes (molecule + property) |

### Other Data

| File | Purpose |
|---|---|
| `structural_opt_light.json` | MolEdit structural optimization (AddComponent/SubComponent/DelComponent) |
| `SFT_formatted_data.json` | SFT-formatted version of the light training set |
| `TRAIN_multi_prop/` | Multi-property optimization (bbbp+drd2+plogp, etc.) |
| `TEST_multi_prop/` | Multi-property test sets |

---

## C. Reward Function Support

### Current support in `rewards.py`

The `get_smile_optimization_reward()` function (line 784) supports:
- **logP** → `Descriptors.MolLogP(mol)` ✅
- **qed** → `Descriptors.qed(mol)` ✅
- **mr** → `Descriptors.MolMR(mol)` ✅
- **tpsa** → `Descriptors.TPSA(mol)` ✅

Direction is inferred from prompt text: "increase"/"higher"/"maximize" → increase; "decrease"/"lower"/"minimize" → decrease.

**The reward function itself is fully capable of MR and QED. The only bug is the hardcoded `property_name="logP"` in the reward registry.**

### Fix Required

Change both `repo.py` and `ada_repo.py` reward registries from:
```python
"smile_optimization": lambda ...: get_smile_optimization_reward(
    property_name="logP",  # BUG: hardcoded
    ...
```
to:
```python
"smile_optimization": lambda ...: get_smile_optimization_reward(
    property_name=script_args.property_name,  # from config
    ...
```

Or, for per-subtask runs, set `property_name` in the config YAML to the correct property.

---

## D. Extension Plan: Per-Subtask MR and QED Experiments

### Design Decisions

1. **Single-subtask training**: Instead of mixed LogP+MR+QED with a single reward, run separate experiments per subtask. This is the clean comparison.

2. **Config-driven property_name**: Fix the hardcoded reward wiring so property_name comes from config.

3. **Per-subtask data filtering**: Add `subtask_selection: ["LogP"]` (or `["MR"]`, `["QED"]`) to select only 500 samples of that subtask.

4. **Matched hyperparameters**: All hyperparameters identical across subtasks and methods. Only dataset filter and property_name change.

### Files to Modify

| File | Change |
|---|---|
| `repo.py` | Wire `property_name` from config instead of hardcoded "logP" |
| `ada_repo.py` | Same fix |
| New: `configs/repo_3B_LogP.yaml` | LogP-only config |
| New: `configs/repo_3B_MR.yaml` | MR-only config |
| New: `configs/repo_3B_QED.yaml` | QED-only config |
| New: `configs/ada_repo_3B_LogP.yaml` | AdaRePO LogP-only config |
| New: `configs/ada_repo_3B_MR.yaml` | AdaRePO MR-only config |
| New: `configs/ada_repo_3B_QED.yaml` | AdaRePO QED-only config |
| New: PBS scripts per subtask/method | 6 total (3 subtasks × 2 methods) |
| New: `smoke_test_rewards.py` | Verifies reward computation on sample data |
| New: `run_manifest.csv` | Tracks all experiments |

### Polaris Execution Plan

| Phase | Queue | Walltime | Nodes | Purpose |
|---|---|---|---|---|
| Smoke test | login node | N/A | 0 | Verify reward functions, data loading |
| Pilot (10 steps) | debug-scaling | 1:00:00 | 1 | Verify training starts, logs correct |
| Full (184 steps) | preemptable | 6:00:00 | 1 | Full comparison |

Checkpoint cadence: every epoch (save_strategy: "epoch"), plus save_steps: 50 for preemptable safety.

### Experiment Matrix (6 runs)

| experiment_name | task | subtask | method | property_name | train_rows |
|---|---|---|---|---|---|
| repo_LogP | MolOpt | LogP | RePO | logP | 500 |
| repo_MR | MolOpt | MR | RePO | mr | 500 |
| repo_QED | MolOpt | QED | RePO | qed | 500 |
| ada_repo_LogP | MolOpt | LogP | AdaRePO | logP | 500 |
| ada_repo_MR | MolOpt | MR | AdaRePO | mr | 500 |
| ada_repo_QED | MolOpt | QED | AdaRePO | qed | 500 |

Note: With 500 samples / (2 batch × 16 grad_accum) = ~16 steps/epoch × 4 epochs = ~64 steps total per run. Each step ~100s → ~1.8hr per run. Well within 6hr walltime.
