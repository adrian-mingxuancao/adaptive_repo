# AdaRePO → DSI Cluster Handoff

## A. Current Main Goal

Move from a relative-method win (AdaRePO vs local RePO) toward a **full reproduction-level win** — matching or exceeding the published RePO paper numbers on TOMG-Bench.

---

## B. Current Status Summary

### Wave 1 / Table 1 Final (single-seed, all N=5000)

| Task | Paper RePO SR×Sim | Local RePO SR×Sim | Local AdaRePO SR×Sim | AdaRePO vs Local RePO |
|------|-------------------|-------------------|----------------------|-----------------------|
| AddComponent | 0.239 | 0.198 | **0.220** | +11.1% |
| DelComponent | 0.140 | 0.128 | 0.069 | −46% REGRESSION |
| SubComponent | 0.344 | 0.254 | 0.258 | ~tie |
| LogP | 0.297 | 0.219 | 0.234 | +6.8% (seed-42 only) |
| MR | 0.294 | 0.182 | 0.194 | +6.6% |
| QED | 0.236 | 0.123 | 0.111 | −9.8% |

**Both methods are 17–57% below paper RePO.** The reproduction gap is the dominant problem.

---

## C. Key Findings

### LogP Robustness — NOT ROBUST

Multi-seed test (seeds 42, 123, 456; 120 steps; N=5000 eval):

| Seed | RePO SR×Sim | AdaRePO SR×Sim | Δ |
|------|-------------|----------------|---|
| 42 | 0.219 | 0.234 | +6.7% |
| 123 | 0.205 | 0.207 | +0.6% |
| 456 | 0.229 | 0.185 | −19.0% |
| **Mean** | **0.218** | **0.209** | **−4.1%** |

The seed-42 "AdaRePO > RePO on LogP" was a seed artifact. Across 3 seeds, RePO has higher mean and lower variance.

### DelComponent Failure — CONFIRMED as Input Copying

- 80% of AdaRePO's valid DelComponent outputs are near-identical to the input (Tanimoto > 0.99)
- vs 63% for RePO
- Model collects 0.3×similarity reward without attempting deletion
- Low-beta fix (β_max: 1.0 → 0.5) made ALL MolEdit tasks WORSE, not better
- Root cause is in the reward structure + guidance architecture, not a hyperparameter knob

### Reproduction-Gap Diagnosis

Top ranked causes (local vs paper config):

| Rank | Factor | Ours | Paper | Impact |
|------|--------|------|-------|--------|
| 1 | num_generations | 3 | 4 | HIGH — 33% less exploration. Cannot use G=4 (batch divisibility: global_batch=3 not divisible by 4) |
| 2 | max_completion_length | 512 | 1024 | HIGH — truncates reasoning chains. **Test in flight (jobs 6978443 + 6978447)** |
| 3 | gradient_accumulation | 16 | 8 | MEDIUM |
| 4 | attn_implementation | sdpa | flash_attention_2 | LOW-MEDIUM |
| 5 | single-task vs joint | 500 samples | 1500 samples | LOW-MEDIUM |

### QED / MR

- QED: AdaRePO loses by 9.8% at seed-42. Not tested multi-seed yet. Lower priority.
- MR: AdaRePO wins by 6.6% at seed-42. Not tested multi-seed. Likely same seed-variance issue.

---

## D. Paper RePO Table 1 Numbers (Reference)

| Task | SR | Sim | SR×Sim |
|------|-----|------|--------|
| AddComponent | 0.307 | 0.778 | 0.239 |
| DelComponent | 0.158 | 0.887 | 0.140 |
| SubComponent | 0.429 | 0.802 | 0.344 |
| QED | 0.312 | 0.756 | 0.236 |
| LogP | 0.415 | 0.715 | 0.297 |
| MR | 0.399 | 0.736 | 0.294 |

---

## E. Exact Locations (Polaris Paths)

All under `/lus/eagle/projects/IMPROVE_Aim1/caom/`.

### Key Repos
- **RePO** (upstream + our Polaris configs): `RePO/`
- **AdaRePO** (our method): `agent_drug_discovery/adaptive_repo/`
- **Git remote**: `git@github.com:qinanh/agent_drug_discovery.git` branch `main`

### Core Code
- `adaptive_repo/ada_repo.py` — main AdaRePO training entry point
- `adaptive_repo/ada_repo_trainer.py` — custom GRPO trainer with adaptive beta
- `adaptive_repo/dynamic_beta.py` — sigmoid-gap beta controller
- `adaptive_repo/memory_bank.py` — memory bank module (currently disabled)
- `adaptive_repo/ada_repo_config.py` — config dataclass
- `RePO/src/x_r1/repo.py` — baseline RePO training entry point
- `RePO/src/x_r1/rewards.py` — all reward functions (shared)
- `RePO/evaluate.py` — evaluation script
- `RePO/generate_predictions.py` — inference/generation script

### Configs
- AdaRePO: `adaptive_repo/configs/ada_repo_3B_{LogP,MR,QED,Structure}.yaml`
- RePO: `RePO/recipes/Polaris_3B_{LogP,MR,QED,Structure}.yaml`
- Paper reference: `RePO/recipes/OpenMolIns_3B_config.yaml`
- DeepSpeed: `RePO/recipes/zero3_polaris.yaml`

### Launch Scripts
- Unified launcher: `adaptive_repo/scripts/launch_experiment.sh`
- Wave 2 PBS scripts: `adaptive_repo/scripts/wave2_*.pbs`

### Key Checkpoints (on Polaris Eagle filesystem)
- RePO LogP seed-42 ckpt-120: `RePO/output/repo_polaris_3B_LogP/checkpoint-120/`
- AdaRePO LogP seed-42 ckpt-120: `adaptive_repo/output/ada_repo_3B_LogP/checkpoint-120/`
- RePO Structure ckpt-184: `RePO/output/repo_polaris_3B_Structure/checkpoint-184/`
- AdaRePO Structure: `adaptive_repo/output/ada_repo_3B_Structure/`
- Multi-seed LogP: `RePO/output/ms_repo_3B_LogP_s{123,456}/`, `adaptive_repo/output/ms_ada_repo_3B_LogP_s{123,456}/`
- Low-beta structural: `adaptive_repo/output/ada_repo_3B_Structure_lowbeta/`

### Key Evaluation Summaries
- Wave 1: `adaptive_repo/evaluation_results/wave1/`
- Wave 2: `adaptive_repo/evaluation_results/wave2/`
- Summary CSVs in `open_generation/MolOpt/LogP_summary.csv` or `MolEdit/{Add,Del,Sub}Component_summary.csv`

### Main Analysis
- `adaptive_repo/analysis/phase1_results_analysis.md`
- `adaptive_repo/analysis/run_manifest.csv`
- `adaptive_repo/analysis/current_setup_audit.md`

---

## F. Jobs History

### Wave 1 — Single-seed Table 1
| Job | What | Result |
|-----|------|--------|
| 6958401-6958406 | 6 single-subtask runs (repo/ada × LogP/MR/QED) | Completed, 60 steps |
| 6962932-6962935 | Phase 1 re-run at 120 steps | Completed |
| 6963296-6963299 | Clean 120-step runs | Completed |
| 6963502-6963505 | 240-step extension runs | Completed |
| 6973269-6973270 | Wave 1 LogP + Structure training | Completed |
| 6974299 | AdaRePO Structure training | Completed |
| Various eval jobs | Full N=5000 evaluation | Completed |

### Wave 2 — Multi-seed + Diagnosis
| Job | What | Result |
|-----|------|--------|
| 6976289 | LogP seed-123 (RePO+AdaRePO sequential) | RePO done, AdaRePO hit walltime at ckpt-75 |
| 6977533 | Resume AdaRePO seed-123 + eval both | **Success** — AdaRePO SR×Sim=0.207, RePO=0.205 |
| 6977534 | AdaRePO LogP seed-456 + eval | **Success** — SR×Sim=0.185 |
| 6977536 | RePO LogP seed-456 + eval | **Success** — SR×Sim=0.229 |
| 6977537 | AdaRePO Structure β_max=0.5 + eval | **Success** — DelComp SR=0.034 (WORSE) |
| 6977538 | Reprogap G=4 maxlen=1024 | **FAILED** — batch not divisible by G=4 |
| 6978443 | Reprogap RePO maxlen=1024 (fixed) | **Queued/Running** |
| 6978447 | Reprogap AdaRePO maxlen=1024 | **Queued/Running** |

---

## G. Known Pitfalls

1. **Eval output dir naming bug**: `--output_dir "foo"` with `--model_path "bar/checkpoint-120"` produces `foocheckpoint-120/` (missing slash). Always add trailing slash or use explicit separate dirs.

2. **Walltime for sequential jobs**: One 120-step training run takes ~3–3.5h. Two sequential runs do NOT fit in 6h walltime. Use separate PBS scripts per method.

3. **Structural reward format mismatch**: `get_molecular_structure_reward()` in `rewards.py` expects `completions` as `[[{"role":"assistant","content":...}]]` but AdaRePO's eval produces plain strings. **Local fix applied** in `ada_repo.py` line 255-257 — wraps strings into chat format. This fix is NOT in `rewards.py` (shared with RePO).

4. **num_generations=4 impossible**: With 3 training processes, global_batch must be divisible by G. batch_size=2 → global=6, not divisible by 4. batch_size=4 → global=12, divisible by 4, but likely OOM with maxlen=1024.

5. **DelComponent input-copying**: 80% of AdaRePO valid outputs are near-identical to input. Low-beta fix made it worse. The reward gives 0.3×similarity even for zero-edit outputs, creating incentive to copy.

6. **MR reward scale**: MR values (30–200) much larger than LogP (0–5) or QED (0–1). Raw reward curves not comparable across subtasks.

7. **vLLM version detection**: vLLM 0.6.3.post1 reports as "dev" causing routing failures. Patched in site-packages `verl/third_party/vllm/__init__.py` and `verl/workers/rollout/vllm_rollout/__init__.py`. These patches are in the **conda env, not in git**.

8. **CUDA/DeepSpeed**: `DS_SKIP_CUDA_CHECK=1` needed (system CUDA 12.9 vs torch 12.4). `CUDA_HOME` must point to `/opt/nvidia/hpc_sdk/Linux_x86_64/25.5/cuda`. `LIBRARY_PATH` must include math_libs for `libcurand`.

---

## H. Best Next Execution Steps

### BEST: Reproduction-gap test — max_completion_length=1024

**Why**: The reproduction gap (both methods 17-57% below paper) is the dominant problem. AdaRePO's relative advantage collapsed under multi-seed testing. Closing the gap is prerequisite for any credible claim.

**Status**: Jobs 6978443 (RePO) and 6978447 (AdaRePO) are queued/running on Polaris. Check results first. If they ran on Polaris, collect and compare:
- `evaluation_results/wave2/reprogap_repo_logp/` (RePO maxlen=1024)
- `evaluation_results/wave2/reprogap_ada_logp/` (AdaRePO maxlen=1024)

**If migrating to DSI before those complete**, re-run on DSI:
- Config: `RePO/recipes/Polaris_3B_LogP.yaml` with `--max_completion_length 1024`
- Entry: `RePO/src/x_r1/repo.py`
- 120 steps, eval on full N=5000
- Compare against baseline (SR×Sim=0.218 mean) and paper (0.297)

**If DSI has more GPU memory**: Also test `num_generations=4` with `per_device_train_batch_size=4` and `max_completion_length=1024`. This requires global_batch divisible by 4.

### SECOND BEST: Fix the G=4 constraint

On DSI, if GPUs have >40GB or if you can use more training processes (e.g., 4 training + 1 vLLM on 5+ GPUs):
- Set `num_generations=4`, `max_completion_length=1024`, `per_device_train_batch_size=4`
- This matches the paper setup exactly
- Run both RePO and AdaRePO LogP
- Compare to paper Table 1

---

## I. Migration Notes

### Move with git (already committed)
- All AdaRePO code: `ada_repo.py`, `ada_repo_trainer.py`, `dynamic_beta.py`, `memory_bank.py`, `ada_repo_config.py`
- All configs: `configs/*.yaml`
- All PBS/launch scripts: `scripts/`
- All analysis: `analysis/`
- This handoff file

### Move with Globus (large files)
- **Checkpoints** (~4TB total in `adaptive_repo/output/`): Transfer only what you need:
  - Best single-seed: `ada_repo_3B_LogP/checkpoint-120/`, `repo_polaris_3B_LogP/checkpoint-120/`
  - Multi-seed: `ms_repo_3B_LogP_s{123,456}/checkpoint-120/`, `ms_ada_repo_3B_LogP_s{123,456}/checkpoint-120/`
  - Structural: `ada_repo_3B_Structure/`, `repo_polaris_3B_Structure/checkpoint-184/`
- **Evaluation results** (~526MB in `adaptive_repo/evaluation_results/`): Transfer all — these are the summary CSVs and detailed results.
- **RePO codebase**: `RePO/` (the upstream repo with our Polaris configs and the shared reward/eval code).
- **Model weights**: `~/.cache/huggingface/Qwen2.5-3B-Instruct/`
- **Conda env**: Rebuild on DSI; do NOT transfer. Key packages: `trl`, `deepspeed`, `vllm==0.6.3.post1`, `rdkit`, `transformers`, `accelerate`.

### Do NOT move
- `adaptive_repo/logs/` (1.7GB of training logs — regenerated on new runs)
- PBS `.o*` output files
- `__pycache__/`
- `AgentFlow/.venv/`
