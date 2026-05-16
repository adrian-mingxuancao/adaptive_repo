# AdaRePO / APIAR

AdaRePO, also referred to as APIAR in the experiment notes, extends RePO for instruction-conditioned molecular optimization. The method keeps RePO's answer-level reference guidance and adds adaptive guidance strength, optional memory-bank self-distillation, adaptive temperature, and related ablations.

This repository is the standalone snapshot of `agent_drug_discovery/adaptive_repo`. It contains the AdaRePO training entry point, experiment configs, submission scripts, analysis scripts, generated figures, and writing-ready result tables.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `ada_repo.py` | Main AdaRePO training entry point. |
| `ada_repo_trainer.py` | Custom GRPO/RePO trainer with adaptive beta, memory bank, experience buffer, and adaptive temperature hooks. |
| `ada_repo_config.py` | TRL `GRPOConfig` extension with AdaRePO/APIAR hyperparameters. |
| `dynamic_beta.py` | Dynamic beta controller implementations. |
| `memory_bank.py` | Per-query best-molecule memory bank for self-distillation. |
| `experience_buffer.py` | Stable example replay buffer. |
| `configs/` | Training configs for v15-v18, ablations, GRPO, offline RePO, and iterative SFT baselines. |
| `scripts/` | Training, evaluation, aggregation, significance, and cluster submission scripts. |
| `analysis/` | Analysis scripts plus generated figures and CSV assets. |
| `results/` | Paper-ready aggregate result tables. |
| `RESULTS_SUMMARY.md` | Writing-ready summary of completed APIAR experiments. |
| `apiar_experiment_plan.md` | Experimental plan and acceptance criteria. |
| `experiment_audit.md` | Historical run inventory and status notes. |

## What Is Reproducible From This Repo

The checked-in analysis artifacts are directly reproducible from the CSVs and generated result files included here. Full model training and evaluation require the upstream RePO checkout because this repo imports RePO utilities:

- `x_r1.rewards`
- `x_r1.x_repo_trainer`
- `x_r1.utils.callbacks`
- RePO datasets under `data/OpenMolIns`
- RePO evaluation/generation scripts for several historical workflows

Expected sibling layout:

```text
<workspace>/
  RePO/
    src/x_r1/
    data/OpenMolIns/
    data/benchmarks/open_generation/MolOpt/
    recipes/
  adaptive_repo/
    ada_repo.py
    configs/
    scripts/
```

For this workspace, `<workspace>` was `/home/caom/rl-agent` locally and `/lus/eagle/projects/IMPROVE_Aim1/caom` or `/net/scratch/...` on clusters. Historical scripts preserve those absolute paths for auditability. Before rerunning them elsewhere, update `REPO_DIR`, `ADA_DIR`, `ENV_DIR`, scratch/output paths, and cache directories.

## Environment

Use the same CUDA/PyTorch stack as the target cluster. The project was run with:

- Python 3.10/3.12 compatible code
- PyTorch with bf16 GPU support
- TRL GRPO support
- Accelerate + DeepSpeed ZeRO-3
- vLLM for generation
- RDKit for molecule rewards/evaluation
- pandas/numpy/matplotlib for analysis

Install the Python packages with:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On HPC systems, prefer the site-provided CUDA-enabled PyTorch/vLLM/DeepSpeed modules or prebuilt environment over installing GPU packages on a login node.

## Required External Assets

Full training/evaluation assumes these assets exist outside this repo:

| Asset | Expected location in sibling RePO checkout |
| --- | --- |
| OpenMolIns training CSVs | `RePO/data/OpenMolIns/{light,...}/train.csv`, `train_hard.csv` |
| MolOpt benchmark data | `RePO/data/benchmarks/open_generation/MolOpt/` |
| RePO reward/trainer code | `RePO/src/x_r1/` |
| DeepSpeed configs | `RePO/recipes/zero3_dsi.yaml` or cluster-specific equivalent |
| Base model | The `model_name_or_path` in each config, usually a local Qwen2.5-3B-Instruct snapshot |

If paths differ, edit the corresponding YAML config and launcher script.

## Quick Sanity Check

Run this after cloning or after edits:

```bash
python scripts/sanity_check.py
```

The script checks required files, Python syntax, result CSV readability, config shape, generated-cache cleanliness, and README reproducibility notes. It deliberately does not import `torch`, `trl`, `vllm`, or `x_r1`, so it can run on a login node.

## Analysis Reproduction

The main paper tables are already checked in:

```text
results/aggregated_results.csv
results/table1_main_srxsim.csv
results/table2_ablation.csv
results/table_significance.csv
analysis/e1a_significance_tests.csv
analysis/e2_gap_bucketed_analysis.csv
```

To regenerate aggregate tables from evaluation outputs on the original cluster layout:

```bash
python scripts/aggregate_results.py --output_dir results --print-table
```

Several analysis scripts still point to the original absolute experiment roots because they scan large external prediction/checkpoint directories that are not included in git. Update `REPO_ROOT`, `ADA_ROOT`, or script-local path constants before running them in a new layout.

## Training Examples

Set `PYTHONPATH` so AdaRePO can import both this repo and RePO:

```bash
export WORKSPACE=/path/to/workspace
export REPO_DIR=${WORKSPACE}/RePO
export ADA_DIR=${WORKSPACE}/adaptive_repo
export PYTHONPATH="${REPO_DIR}/src:${ADA_DIR}:${PYTHONPATH:-}"
```

Example local/cluster training command:

```bash
cd "${REPO_DIR}"
ACCELERATE_LOG_LEVEL=info accelerate launch \
  --config_file "${REPO_DIR}/recipes/zero3_dsi.yaml" \
  --num_processes 2 \
  "${ADA_DIR}/ada_repo.py" \
  --config "${ADA_DIR}/configs/DSI_v16_adaptive_curriculum.yaml" \
  --output_dir /path/to/output/ada_repo_dsi_v16 \
  --variant default \
  --run_name adarepo_v16
```

Important config knobs:

| Field | Meaning |
| --- | --- |
| `beta_guide_mode` | Dynamic beta strategy, e.g. `sigmoid_gap`, `sample_sigmoid`, `fixed`. |
| `beta_guide_max`, `beta_guide_min`, `beta_guide_alpha` | Guidance strength range and sigmoid sharpness. |
| `use_memory_bank` | Enables self-distillation from best generated molecules. |
| `use_adaptive_temperature` | Raises decoding temperature for collapsed prompts. |
| `disable_reference_guidance` | Produces GRPO-style baseline behavior when true. |
| `subtask_selection` | Selects MolOpt subtasks such as `LogP`, `MR`, `QED`. |

## Cluster Scripts

The `scripts/` directory includes historical SLURM/PBS launchers for DSI, Polaris, and earlier experiments. Treat them as reproducibility records plus templates. Before reuse:

1. Update account/queue/partition directives.
2. Update `REPO_DIR`, `ADA_DIR`, `ENV_DIR`, `SCRATCH`, and cache locations.
3. Confirm the config's `model_name_or_path` exists on the target system.
4. Confirm `num_processes`, GPU count, and DeepSpeed config match the node.
5. Run `python scripts/sanity_check.py`.

## Main Results

The current writing-ready summary is in `RESULTS_SUMMARY.md`. The headline Table 1 metric is SR x Sim over 3 seeds:

| Method | Avg SR x Sim |
| --- | ---: |
| Zero-shot | 0.1376 |
| GRPO | 0.0959 |
| RePO | 0.1665 |
| C1: Iterative SFT | 0.1627 |
| C2: Offline-Strengthened | 0.1652 |
| APIAR / AdaRePO | 0.1773 |

See `results/table1_main_srxsim.csv` and `RESULTS_SUMMARY.md` for full per-subtask values, standard errors, ablations, significance tests, and qualitative summaries.

## Cleanliness Policy

Keep source, configs, analysis scripts, small CSV summaries, and final figures in git. Do not commit:

- checkpoints or model weights
- raw WandB runs
- local caches
- `__pycache__` or `*.pyc`
- `slurm-*.out`, `*.log`, `*.err`, or large transient outputs
- external RePO datasets and benchmark directories

The `.gitignore` enforces the common generated-file cases; use external storage for large model/evaluation artifacts.

## Current Known Limitations

- Full training is not self-contained because the upstream RePO code and datasets are external dependencies.
- Several historical scripts contain absolute cluster paths by design. They need path edits for new machines.
- `requirements.txt` records package names, not exact locked binary builds. CUDA/vLLM/DeepSpeed compatibility should be pinned through the target environment module or cluster image.
- Some analysis scripts regenerate figures from external prediction directories that are not committed here; the final generated CSVs/figures are committed for inspection.

## Citation / Naming

Use `AdaRePO` for the implementation and `APIAR` for the paper framing when needed. The repository contains both names because the method evolved from AdaRePO implementation work into the APIAR experiment package.
