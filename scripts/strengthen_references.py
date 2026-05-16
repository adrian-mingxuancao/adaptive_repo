#!/usr/bin/env python3
"""
C2: Offline-strengthened reference generation.

For each training example, sample M candidates from the base policy,
select the highest-reward valid candidate, and save as a new training CSV
with the `molecule` column replaced by the strengthened reference.

Usage:
    python scripts/strengthen_references.py \
        --model_path /path/to/Qwen2.5-3B-Instruct \
        --train_csv data/OpenMolIns/light/train.csv \
        --output_csv data/OpenMolIns/light/train_strengthened.csv \
        --num_candidates 64 \
        --temperature 1.0 \
        --similarity_weight 0.5 \
        --property_weight 0.5

Produces:
    - train_strengthened.csv: same schema as train.csv but molecule = best candidate
    - train_strengthened_stats.json: per-example stats (reward gap, etc.)
"""

import argparse
import json
import os
import re
import sys

import pandas as pd
from tqdm import tqdm

# Add project source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "x_r1"))


def extract_smiles(text):
    """Extract SMILES from model output (same logic as generate_predictions.py)."""
    # Try <answer>...</answer> first
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try ```...```
    m = re.search(r"```(?:smiles)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try last line that looks like SMILES
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if line and re.match(r"^[A-Za-z0-9@+\-\[\]\(\)\\/#=.%]+$", line):
            return line
    # Fallback: return last non-empty line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


def compute_reward_for_candidate(smiles, original_mol, instruction, subtask,
                                  similarity_weight, property_weight, min_similarity):
    """Compute reward for a single candidate SMILES."""
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED, AllChem, DataStructs
    except ImportError:
        raise ImportError("RDKit is required. Install with: pip install rdkit")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0, {"valid": False}

    orig_mol = Chem.MolFromSmiles(original_mol)
    if orig_mol is None:
        return -1.0, {"valid": False}

    # Compute Tanimoto similarity
    fp1 = AllChem.GetMorganFingerprintAsBitVect(orig_mol, 2, nBits=1024)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

    if similarity < min_similarity:
        return -0.5, {"valid": True, "similarity": similarity, "below_threshold": True}

    # Compute property based on subtask
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
        return -1.0, {"valid": True, "unknown_subtask": subtask}

    # Determine direction from instruction
    instruction_lower = instruction.lower()
    if "increase" in instruction_lower or "higher" in instruction_lower or "improve" in instruction_lower:
        direction = 1
    elif "decrease" in instruction_lower or "lower" in instruction_lower or "reduce" in instruction_lower:
        direction = -1
    else:
        direction = 1  # default

    # Property improvement reward
    delta = (new_val - orig_val) * direction
    prop_reward = min(max(delta / (abs(orig_val) + 1e-6), 0.0), 1.0)

    # Combined reward
    reward = similarity_weight * similarity + property_weight * prop_reward

    return reward, {
        "valid": True,
        "similarity": similarity,
        "orig_val": orig_val,
        "new_val": new_val,
        "delta": delta,
        "prop_reward": prop_reward,
        "reward": reward,
    }


def main():
    parser = argparse.ArgumentParser(description="C2: Generate offline-strengthened references")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--num_candidates", type=int, default=64, help="M: candidates per example")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--similarity_weight", type=float, default=0.5)
    parser.add_argument("--property_weight", type=float, default=0.5)
    parser.add_argument("--min_similarity", type=float, default=0.2)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Process examples in batches (default: all at once)")
    args = parser.parse_args()

    print(f"Loading training data from {args.train_csv}")
    df = pd.read_csv(args.train_csv)
    print(f"Loaded {len(df)} examples")

    # Build prompts (same as generate_predictions.py English MolOpt template)
    prompts_all = []
    for _, row in df.iterrows():
        instruction = row["Instruction"]
        molecule = row["molecule"]
        prompt = f"{instruction}\nMolecule SMILES: {molecule}\n\nPlease provide the optimized molecule in SMILES format."
        prompts_all.append(prompt)

    # Load model with vLLM
    print(f"Loading model: {args.model_path}")
    from vllm import LLM, SamplingParams

    import torch
    tp_size = torch.cuda.device_count()
    print(f"Using tensor_parallel_size={tp_size}")
    llm = LLM(
        model=args.model_path,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=tp_size,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        n=args.num_candidates,  # Generate M candidates per prompt
    )

    # Generate all candidates
    print(f"Generating {args.num_candidates} candidates per example...")
    outputs = llm.generate(prompts_all, sampling_params)
    print(f"Generation complete: {len(outputs)} prompts processed")

    # For each example, select the best candidate
    strengthened_molecules = []
    stats = []

    for idx in tqdm(range(len(df)), desc="Scoring candidates"):
        row = df.iloc[idx]
        original_mol = row["molecule"]
        instruction = row["Instruction"]
        subtask = row["SubTask"]

        candidates = outputs[idx].outputs  # list of M completions

        best_reward = -float("inf")
        best_smiles = original_mol  # fallback to original
        best_info = None
        n_valid = 0

        for cand in candidates:
            smiles = extract_smiles(cand.text)
            if not smiles:
                continue

            reward, info = compute_reward_for_candidate(
                smiles, original_mol, instruction, subtask,
                args.similarity_weight, args.property_weight, args.min_similarity,
            )

            if info.get("valid", False):
                n_valid += 1

            if reward > best_reward:
                best_reward = reward
                best_smiles = smiles
                best_info = info

        # Compute original reference reward for comparison
        orig_reward, orig_info = compute_reward_for_candidate(
            original_mol, original_mol, instruction, subtask,
            args.similarity_weight, args.property_weight, args.min_similarity,
        )

        strengthened_molecules.append(best_smiles)
        stats.append({
            "idx": idx,
            "subtask": subtask,
            "original_mol": original_mol,
            "strengthened_mol": best_smiles,
            "original_reward": orig_reward,
            "best_reward": best_reward,
            "reward_gap": best_reward - orig_reward,
            "n_valid_candidates": n_valid,
            "n_total_candidates": len(candidates),
            "kept_original": best_smiles == original_mol,
        })

    # Save strengthened CSV
    df_out = df.copy()
    df_out["molecule"] = strengthened_molecules
    df_out["original_molecule"] = df["molecule"]  # keep original for reference
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df_out.to_csv(args.output_csv, index=False)
    print(f"Saved strengthened training data to {args.output_csv}")

    # Save stats
    stats_path = args.output_csv.replace(".csv", "_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved stats to {stats_path}")

    # Print summary
    reward_gaps = [s["reward_gap"] for s in stats]
    kept_count = sum(1 for s in stats if s["kept_original"])
    avg_gap = sum(reward_gaps) / len(reward_gaps)
    print(f"\n--- Summary ---")
    print(f"Total examples: {len(stats)}")
    print(f"Kept original (no improvement): {kept_count} ({100*kept_count/len(stats):.1f}%)")
    print(f"Average reward gap (strengthened - original): {avg_gap:.4f}")
    print(f"Mean valid candidates per example: {sum(s['n_valid_candidates'] for s in stats)/len(stats):.1f}")


if __name__ == "__main__":
    main()
