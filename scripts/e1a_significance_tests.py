#!/usr/bin/env python3
"""
E1a: Statistical significance tests for APIAR paper.
Paired bootstrap test + paired t-test for:
  - APIAR-full vs RePO
  - APIAR-full vs strongest baseline (C2)
  - APIAR-full vs each ablation
"""
import csv
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import sys

np.random.seed(42)

# ---- Config ----
N_BOOTSTRAP = 10000
ALPHA = 0.05

# Eval result locations (two possible dirs)
PRED_DIRS = [
    "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/predictions",
    "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/evaluation_results",
]

# Method -> list of (dir_pattern_fragments, seed)
# We need per-example results to do paired tests
METHODS = {
    "APIAR (v16)": [
        ("v16v17ms_v16_s42", "v16_s42checkpoint-120"), 
        ("v16v17ms_v16_s123", "v16_s123checkpoint-120"),
        ("v16v17ms_v16_s456", "v16_s456checkpoint-120"),
    ],
    "RePO": [
        ("v16v17ms_repo_s42", "repo_s42checkpoint-120"),
        ("v16v17ms_repo_s123", "repo_s123checkpoint-120"),
        ("v16v17ms_repo_s456", "repo_s456checkpoint-120"),
    ],
    "GRPO": [
        ("grpo_baseline_s42/checkpoint-120",),
        ("grpo_baseline_s123/checkpoint-120",),
        ("grpo_baseline_s456/checkpoint-120",),
    ],
    "C1: Iterative SFT": [
        ("isft_s42/checkpoint-30",),
        ("isft_s123/checkpoint-30",),
        ("isft_s456/checkpoint-30",),
    ],
    "C2: Offline-Str.": [
        ("c2_offline_s42/checkpoint-120",),
        ("c2_offline_s123/checkpoint-120",),
        ("c2_offline_s456/checkpoint-120",),
    ],
    "Abl: β-only": [
        ("ablation_beta_only_s42checkpoint-120", "ablation_beta_only_s42/checkpoint-120"),
        ("ablation_beta_only_s123checkpoint-120", "ablation_beta_only_s123/checkpoint-120"),
        ("ablation_beta_only_s456/checkpoint-120",),
    ],
    "Abl: Bank-only": [
        ("ablation_bank_only_s42checkpoint-120", "ablation_bank_only_s42/checkpoint-120"),
        ("ablation_bank_only_s123/checkpoint-120",),
        ("ablation_bank_only_s456/checkpoint-120",),
    ],
}

SUBTASKS = ["LogP", "MR", "QED"]

def find_detailed_csv(method_fragments, subtask):
    """Find the detailed results CSV for a method+subtask."""
    for base_dir in PRED_DIRS:
        for frag in method_fragments:
            # Try multiple path patterns
            candidates = [
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv",
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}.csv",
            ]
            # Also try with subdirs
            for subdir in Path(base_dir).iterdir() if Path(base_dir).exists() else []:
                if subdir.is_dir():
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv")
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}.csv")
            
            for c in candidates:
                if c.exists():
                    return c
    return None


def load_per_example(csv_path):
    """Load per-example success and similarity from detailed CSV."""
    successes = []
    similarities = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = float(row.get("success", 0))
            sim = float(row.get("similarity", 0))
            successes.append(s)
            similarities.append(sim)
    return np.array(successes), np.array(similarities)


def paired_bootstrap(metric_a, metric_b, n_bootstrap=N_BOOTSTRAP):
    """
    Paired bootstrap test: H0: mean(A) <= mean(B).
    Returns p-value (fraction of bootstraps where mean(A) - mean(B) <= 0).
    """
    n = len(metric_a)
    assert len(metric_b) == n
    observed_diff = np.mean(metric_a) - np.mean(metric_b)
    
    count = 0
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        diff = np.mean(metric_a[idx]) - np.mean(metric_b[idx])
        if diff <= 0:
            count += 1
    
    p_value = count / n_bootstrap
    return observed_diff, p_value


def paired_ttest(metric_a, metric_b):
    """Paired t-test on per-example differences."""
    diffs = metric_a - metric_b
    n = len(diffs)
    mean_diff = np.mean(diffs)
    se = np.std(diffs, ddof=1) / np.sqrt(n)
    if se == 0:
        return mean_diff, 0.0
    t_stat = mean_diff / se
    # Two-sided p-value approximation using normal (n is large, ~5000)
    from math import erfc, sqrt
    p_value = erfc(abs(t_stat) / sqrt(2))
    return mean_diff, p_value


def seed_level_bootstrap(seed_means_a, seed_means_b, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap test at seed level (n=3 seeds)."""
    diffs = np.array(seed_means_a) - np.array(seed_means_b)
    observed = np.mean(diffs)
    n = len(diffs)
    count = 0
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        if np.mean(diffs[idx]) <= 0:
            count += 1
    return observed, count / n_bootstrap


def main():
    # Load all per-example data
    print("Loading per-example results...")
    data = {}  # method -> subtask -> seed_idx -> (success, sim)
    
    for method, seed_list in METHODS.items():
        data[method] = {}
        for subtask in SUBTASKS:
            data[method][subtask] = []
            for seed_idx, fragments in enumerate(seed_list):
                csv_path = find_detailed_csv(fragments, subtask)
                if csv_path is None:
                    print(f"  WARNING: missing {method} seed{seed_idx} {subtask}", file=sys.stderr)
                    data[method][subtask].append((np.array([]), np.array([])))
                else:
                    succ, sim = load_per_example(csv_path)
                    data[method][subtask].append((succ, sim))
                    # print(f"  {method} seed{seed_idx} {subtask}: n={len(succ)}, SR={np.mean(succ):.4f}")
    
    # ---- Comparisons ----
    apiar = "APIAR (v16)"
    comparisons = [
        ("APIAR vs RePO", "RePO"),
        ("APIAR vs C2 (strongest baseline)", "C2: Offline-Str."),
        ("APIAR vs C1", "C1: Iterative SFT"),
        ("APIAR vs GRPO", "GRPO"),
        ("APIAR vs Abl:β-only", "Abl: β-only"),
        ("APIAR vs Abl:Bank-only", "Abl: Bank-only"),
    ]
    
    print("\n" + "=" * 90)
    print(f"{'Comparison':<35} | {'Subtask':<6} | {'Δ SR':>8} | {'Δ SR×Sim':>10} | {'p(boot)':>8} | {'p(t)':>8}")
    print("=" * 90)
    
    summary_rows = []
    
    for comp_label, other_method in comparisons:
        for subtask in SUBTASKS:
            # Collect per-seed metrics
            sr_a_seeds, sr_b_seeds = [], []
            srxsim_a_seeds, srxsim_b_seeds = [], []
            
            # Also accumulate all per-example for pooled test
            all_sr_a, all_sr_b = [], []
            all_srxsim_a, all_srxsim_b = [], []
            
            for seed_idx in range(3):
                succ_a, sim_a = data[apiar][subtask][seed_idx]
                succ_b, sim_b = data[other_method][subtask][seed_idx]
                
                if len(succ_a) == 0 or len(succ_b) == 0:
                    continue
                
                sr_a_seeds.append(np.mean(succ_a))
                sr_b_seeds.append(np.mean(succ_b))
                srxsim_a = succ_a * sim_a
                srxsim_b = succ_b * sim_b
                srxsim_a_seeds.append(np.mean(srxsim_a))
                srxsim_b_seeds.append(np.mean(srxsim_b))
                
                all_sr_a.append(succ_a)
                all_sr_b.append(succ_b)
                all_srxsim_a.append(srxsim_a)
                all_srxsim_b.append(srxsim_b)
            
            if not sr_a_seeds:
                continue
            
            # Pooled per-example paired bootstrap (use seed 0 for per-example test)
            pool_a = np.concatenate(all_srxsim_a)
            pool_b = np.concatenate(all_srxsim_b)
            
            sr_diff = np.mean(sr_a_seeds) - np.mean(sr_b_seeds)
            srxsim_diff = np.mean(srxsim_a_seeds) - np.mean(srxsim_b_seeds)
            
            # Paired bootstrap on pooled examples
            _, p_boot = paired_bootstrap(pool_a, pool_b)
            # Paired t-test on pooled examples
            _, p_t = paired_ttest(pool_a, pool_b)
            
            sig = "***" if p_boot < 0.001 else "**" if p_boot < 0.01 else "*" if p_boot < 0.05 else ""
            print(f"{comp_label:<35} | {subtask:<6} | {sr_diff:>+7.4f} | {srxsim_diff:>+9.4f} | {p_boot:>7.4f}{sig} | {p_t:>7.4f}")
            
            summary_rows.append({
                "comparison": comp_label, "subtask": subtask,
                "delta_sr": sr_diff, "delta_srxsim": srxsim_diff,
                "p_bootstrap": p_boot, "p_ttest": p_t,
            })
        print("-" * 90)
    
    # Seed-level aggregate test (across subtasks)
    print(f"\n{'='*70}")
    print(f"{'Seed-level test (n=3 seeds, Avg SR×Sim across subtasks)'}")
    print(f"{'='*70}")
    print(f"{'Comparison':<35} | {'Δ Avg':>8} | {'p(seed-boot)':>12}")
    print("-" * 60)
    
    for comp_label, other_method in comparisons:
        a_means, b_means = [], []
        for seed_idx in range(3):
            seed_srxsim_a, seed_srxsim_b = [], []
            for subtask in SUBTASKS:
                succ_a, sim_a = data[apiar][subtask][seed_idx]
                succ_b, sim_b = data[other_method][subtask][seed_idx]
                if len(succ_a) > 0 and len(succ_b) > 0:
                    seed_srxsim_a.append(np.mean(succ_a * sim_a))
                    seed_srxsim_b.append(np.mean(succ_b * sim_b))
            if seed_srxsim_a:
                a_means.append(np.mean(seed_srxsim_a))
                b_means.append(np.mean(seed_srxsim_b))
        
        if len(a_means) >= 2:
            diff, p = seed_level_bootstrap(a_means, b_means)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"{comp_label:<35} | {diff:>+7.4f} | {p:>11.4f}{sig}")
    
    # Save to CSV
    out_path = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/e1a_significance_tests.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["comparison", "subtask", "delta_sr", "delta_srxsim", "p_bootstrap", "p_ttest"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
