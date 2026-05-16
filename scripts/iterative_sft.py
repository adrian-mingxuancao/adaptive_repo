#!/usr/bin/env python3
"""
C1: Iterative SFT / ReST-style baseline.

Pipeline:
  1. Start from base model.
  2. Run RePO training for T steps (RL phase).
  3. Use current policy to sample N candidates per training example.
  4. Keep best-of-N by reward (valid only).
  5. Create updated training CSV with best candidates as new `molecule` column.
  6. Run one SFT epoch on the updated targets (SFT phase).
  7. Repeat from step 2 with the updated model.

This script orchestrates the iterations. Each phase (RL or SFT) is a separate
accelerate launch, resuming from the previous checkpoint.

Usage:
    python scripts/iterative_sft.py \
        --base_model /path/to/Qwen2.5-3B-Instruct \
        --repo_config recipes/isft_repo_phase.yaml \
        --train_csv data/OpenMolIns/light/train.csv \
        --output_dir output/isft_s42 \
        --seed 42 \
        --n_iterations 4 \
        --steps_per_iteration 30 \
        --n_candidates 8 \
        --wandb_project APIAR-baselines
"""

import argparse
import json
import os
import shutil
import re
import subprocess
import sys

import pandas as pd
from tqdm import tqdm


def extract_smiles(text):
    """Extract SMILES from model output text."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"```(?:smiles)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line and re.match(r"^[A-Za-z0-9@+\-\[\]\(\)\\/#=.%]+$", line):
            return line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def compute_reward(smiles, original_mol, instruction, subtask,
                   similarity_weight=0.5, property_weight=0.5, min_similarity=0.2):
    """Compute reward for a candidate SMILES (same as strengthen_references.py)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, AllChem, DataStructs

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0

    orig_mol = Chem.MolFromSmiles(original_mol)
    if orig_mol is None:
        return -1.0

    fp1 = AllChem.GetMorganFingerprintAsBitVect(orig_mol, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    if similarity < min_similarity:
        return -0.5

    if subtask == "LogP":
        orig_val = Descriptors.MolLogP(orig_mol)
        new_val = Descriptors.MolLogP(mol)
    elif subtask == "MR":
        orig_val = Descriptors.MolMR(orig_mol)
        new_val = Descriptors.MolMR(mol)
    elif subtask == "QED":
        orig_val = QED.qed(orig_mol)
        new_val = QED.qed(mol)
    else:
        return -1.0

    instruction_lower = instruction.lower()
    if "increase" in instruction_lower or "higher" in instruction_lower or "improve" in instruction_lower:
        direction = 1
    elif "decrease" in instruction_lower or "lower" in instruction_lower or "reduce" in instruction_lower:
        direction = -1
    else:
        direction = 1

    delta = (new_val - orig_val) * direction
    prop_reward = min(max(delta / (abs(orig_val) + 1e-6), 0.0), 1.0)
    reward = similarity_weight * similarity + property_weight * prop_reward

    return reward


def sample_best_of_n(model_path, train_df, n_candidates, temperature=0.7,
                     gpu_memory_utilization=0.90, similarity_weight=0.5,
                     property_weight=0.5, min_similarity=0.2):
    """
    Use current policy to generate N candidates per training example,
    return updated DataFrame with best candidates as molecule column.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    # Build prompts
    prompts = []
    for _, row in train_df.iterrows():
        instruction = row["Instruction"]
        molecule = row["molecule"]
        prompt = f"{instruction}\nMolecule SMILES: {molecule}\n\nPlease provide the optimized molecule in SMILES format."
        prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        max_tokens=512,
        n=n_candidates,
    )

    print(f"Generating {n_candidates} candidates per example ({len(prompts)} examples)...")
    outputs = llm.generate(prompts, sampling_params)

    # Score and select best
    new_molecules = []
    stats = {"updated": 0, "kept_original": 0, "avg_reward_gap": 0.0}
    reward_gaps = []

    for idx in tqdm(range(len(train_df)), desc="Scoring candidates"):
        row = train_df.iloc[idx]
        original_mol = row["molecule"]
        instruction = row["Instruction"]
        subtask = row["SubTask"]
        candidates = outputs[idx].outputs

        # Score original
        orig_reward = compute_reward(
            original_mol, original_mol, instruction, subtask,
            similarity_weight, property_weight, min_similarity,
        )

        best_reward = orig_reward
        best_smiles = original_mol

        for cand in candidates:
            smiles = extract_smiles(cand.text)
            if not smiles:
                continue
            reward = compute_reward(
                smiles, original_mol, instruction, subtask,
                similarity_weight, property_weight, min_similarity,
            )
            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles

        new_molecules.append(best_smiles)
        gap = best_reward - orig_reward
        reward_gaps.append(gap)

        if best_smiles != original_mol:
            stats["updated"] += 1
        else:
            stats["kept_original"] += 1

    stats["avg_reward_gap"] = sum(reward_gaps) / len(reward_gaps) if reward_gaps else 0.0

    df_out = train_df.copy()
    df_out["molecule"] = new_molecules
    return df_out, stats


def run_repo_phase(config_path, model_path, output_dir, train_csv, seed,
                   max_steps, run_name, num_processes=3):
    """Run one RL (RePO) training phase via accelerate launch."""
    cmd = [
        "accelerate", "launch",
        "--config_file", "recipes/zero3_polaris.yaml",
        "--num_processes", str(num_processes),
        "src/x_r1/repo.py",
        "--config", config_path,
        "--model_name_or_path", model_path,
        "--output_dir", output_dir,
        "--train_data_file", os.path.basename(train_csv),
        "--seed", str(seed),
        "--max_steps", str(max_steps),
        "--run_name", run_name,
    ]
    print(f"[RL Phase] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=os.environ.get("PROJECT_DIR", "."))
    if result.returncode != 0:
        raise RuntimeError(f"RL phase failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser(description="C1: Iterative SFT baseline")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--repo_config", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_iterations", type=int, default=4,
                        help="Number of RL-SFT iterations (total steps = n_iterations * steps_per_iteration)")
    parser.add_argument("--steps_per_iteration", type=int, default=30,
                        help="RL steps per iteration")
    parser.add_argument("--n_candidates", type=int, default=8,
                        help="Best-of-N candidates for SFT target selection")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--similarity_weight", type=float, default=0.5)
    parser.add_argument("--property_weight", type=float, default=0.5)
    parser.add_argument("--wandb_project", type=str, default="APIAR-baselines")
    parser.add_argument("--num_processes", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Track progress
    progress_file = os.path.join(args.output_dir, "isft_progress.json")
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"completed_iterations": 0, "current_model": args.base_model,
                     "current_train_csv": args.train_csv, "iteration_stats": []}

    for iteration in range(progress["completed_iterations"], args.n_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.n_iterations}")
        print(f"{'='*60}")

        current_model = progress["current_model"]
        current_train_csv = progress["current_train_csv"]

        # --- RL Phase ---
        iter_output = os.path.join(args.output_dir, f"iter{iteration}")
        run_name = f"isft_iter{iteration}_s{args.seed}"
        run_repo_phase(
            config_path=args.repo_config,
            model_path=current_model,
            output_dir=iter_output,
            train_csv=current_train_csv,
            seed=args.seed,
            max_steps=args.steps_per_iteration,
            run_name=run_name,
            num_processes=args.num_processes,
        )

        # --- Best-of-N Phase ---
        # Find best checkpoint in iter_output
        ckpt_dirs = sorted(
            [d for d in os.listdir(iter_output) if d.startswith("checkpoint-")],
            key=lambda x: int(x.split("-")[1]),
        )
        best_ckpt = os.path.join(iter_output, ckpt_dirs[-1]) if ckpt_dirs else iter_output
        print(f"\n[Best-of-N] Sampling {args.n_candidates} candidates from {best_ckpt}")
        train_df = pd.read_csv(args.train_csv)  # Always score against ORIGINAL molecules
        # Filter to subtasks
        train_df = train_df[train_df["SubTask"].isin(["LogP", "MR", "QED"])]

        updated_df, stats = sample_best_of_n(
            model_path=best_ckpt,
            train_df=train_df,
            n_candidates=args.n_candidates,
            temperature=args.temperature,
            similarity_weight=args.similarity_weight,
            property_weight=args.property_weight,
        )

        # Save updated training CSV
        updated_csv = os.path.join(args.output_dir, f"train_iter{iteration}.csv")
        updated_df.to_csv(updated_csv, index=False)
        # Copy to data dir so repo.py can find it via train_data_file basename
        data_dir = os.path.join(os.environ.get("PROJECT_DIR", "."), "data", "OpenMolIns", "light")
        dest = os.path.join(data_dir, f"train_iter{iteration}.csv")
        shutil.copy2(updated_csv, dest)
        print(f"[Best-of-N] Saved updated CSV: {updated_csv} -> {dest}")
        print(f"[Best-of-N] Stats: {stats}")

        # Update progress
        progress["completed_iterations"] = iteration + 1
        progress["current_model"] = best_ckpt
        progress["current_train_csv"] = updated_csv
        progress["iteration_stats"].append({
            "iteration": iteration,
            "model": iter_output,
            "stats": stats,
        })
        with open(progress_file, "w") as f:
            json.dump(progress, f, indent=2)

    # Copy final model to top-level output_dir
    final_model = progress["current_model"]
    print(f"\n{'='*60}")
    print(f"Iterative SFT complete. Final model: {final_model}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
