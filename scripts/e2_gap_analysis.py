#!/usr/bin/env python3
"""
E2: Optimization-headroom-conditional analysis.

Buckets test examples by how much "room to improve" the source molecule has,
then compares APIAR vs RePO per-bucket.

For MolOpt tasks:
  - "decrease LogP": gap = original_logP (higher = more room)
  - "decrease MR": gap = original_MR
  - "increase QED": gap = 1 - original_QED (higher = more room)
  
Produces:
  - Per-quintile SR, SR×Sim, and APIAR-RePO gap
  - CSV + console table
"""
import csv
import numpy as np
import os
import sys
from pathlib import Path
from collections import defaultdict

np.random.seed(42)

PRED_DIRS = [
    "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/predictions",
    "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/evaluation_results",
]

# APIAR v16 and RePO detailed result dirs per seed
APIAR_SEEDS = [
    ("v16v17ms_v16_s42", "v16_s42checkpoint-120"),
    ("v16v17ms_v16_s123", "v16_s123checkpoint-120"),
    ("v16v17ms_v16_s456", "v16_s456checkpoint-120"),
]
REPO_SEEDS = [
    ("v16v17ms_repo_s42", "repo_s42checkpoint-120"),
    ("v16v17ms_repo_s123", "repo_s123checkpoint-120"),
    ("v16v17ms_repo_s456", "repo_s456checkpoint-120"),
]

TEST_DIR = "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/data/benchmarks/open_generation/MolOpt"

SUBTASKS = {
    "LogP": {"test_col": "logP", "direction": "decrease", "gap_fn": lambda v: v},
    "MR":   {"test_col": "MR",   "direction": "decrease", "gap_fn": lambda v: v},
    "QED":  {"test_col": "QED",  "direction": "increase", "gap_fn": lambda v: 1.0 - v},
}

N_BINS = 5  # quintiles


def find_detailed_csv(fragments, subtask):
    for base_dir in PRED_DIRS:
        for frag in fragments:
            candidates = [
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv",
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}.csv",
            ]
            for subdir in Path(base_dir).iterdir() if Path(base_dir).exists() else []:
                if subdir.is_dir():
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv")
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}.csv")
            for c in candidates:
                if c.exists():
                    return c
    return None


def load_test_properties(subtask, test_col):
    """Load source molecule properties from the test set CSV."""
    test_csv = Path(TEST_DIR) / subtask / "test.csv"
    props = []
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            props.append(float(row[test_col]))
    return np.array(props)


def load_detailed(csv_path):
    """Load per-example data. Returns arrays: success, similarity."""
    successes, similarities = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = float(row.get("success", 0))
            sim = float(row.get("similarity", 0))
            successes.append(s)
            similarities.append(sim)
    return np.array(successes), np.array(similarities)


def main():
    out_dir = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis"
    os.makedirs(out_dir, exist_ok=True)
    
    all_rows = []
    
    for subtask, cfg in SUBTASKS.items():
        test_col = cfg["test_col"]
        gap_fn = cfg["gap_fn"]
        
        print(f"\n{'='*80}")
        print(f"  {subtask} — bucketed by optimization headroom (source molecule {cfg['direction']} task)")
        print(f"{'='*80}")
        
        # Load test set properties (same for all seeds)
        test_props = load_test_properties(subtask, test_col)
        gap = gap_fn(test_props)
        print(f"  Test set: n={len(test_props)}, prop range=[{test_props.min():.3f}, {test_props.max():.3f}], gap range=[{gap.min():.3f}, {gap.max():.3f}]")
        
        # Compute quintile boundaries from test set
        quantiles = np.percentile(gap, np.linspace(0, 100, N_BINS + 1))
        # Ensure unique boundaries
        quantiles = np.unique(quantiles)
        actual_bins = len(quantiles) - 1
        bin_idx = np.digitize(gap, quantiles[1:-1])  # 0..actual_bins-1
        
        # Collect per-seed results
        for seed_idx in range(3):
            apiar_path = find_detailed_csv(APIAR_SEEDS[seed_idx], subtask)
            repo_path = find_detailed_csv(REPO_SEEDS[seed_idx], subtask)
            
            if apiar_path is None or repo_path is None:
                print(f"  WARNING: missing seed {seed_idx} for {subtask}")
                continue
            
            a_succ, a_sim = load_detailed(apiar_path)
            r_succ, r_sim = load_detailed(repo_path)
            
            if len(a_succ) != len(test_props):
                print(f"  WARNING: length mismatch seed {seed_idx}: predictions={len(a_succ)}, test={len(test_props)}")
                continue
            
            if seed_idx == 0:
                print(f"\n  {'Bin':<20} | {'Gap Range':<18} | {'n':>5} | {'APIAR SR':>9} | {'RePO SR':>8} | {'Δ SR':>7} | {'APIAR SR×S':>11} | {'RePO SR×S':>10} | {'Δ SR×S':>8}")
                print(f"  {'-'*105}")
            
            for b in range(actual_bins):
                mask = bin_idx == b
                n = mask.sum()
                if n == 0:
                    continue
                
                a_sr = np.mean(a_succ[mask])
                r_sr = np.mean(r_succ[mask])
                a_srxsim = np.mean(a_succ[mask] * a_sim[mask])
                r_srxsim = np.mean(r_succ[mask] * r_sim[mask])
                
                lo, hi = quantiles[b], quantiles[b+1]
                label = f"Q{b+1} ({lo:.2f}, {hi:.2f}]"
                
                if seed_idx == 0:
                    print(f"  {label:<20} | {f'{lo:.2f} - {hi:.2f}':<18} | {n:>5} | {a_sr:>8.4f} | {r_sr:>7.4f} | {a_sr-r_sr:>+6.4f} | {a_srxsim:>10.4f} | {r_srxsim:>9.4f} | {a_srxsim-r_srxsim:>+7.4f}")
                
                all_rows.append({
                    "subtask": subtask, "seed": [42, 123, 456][seed_idx],
                    "bin": b+1, "bin_label": label, "gap_lo": lo, "gap_hi": hi, "n": n,
                    "apiar_sr": a_sr, "repo_sr": r_sr, "delta_sr": a_sr - r_sr,
                    "apiar_srxsim": a_srxsim, "repo_srxsim": r_srxsim,
                    "delta_srxsim": a_srxsim - r_srxsim,
                })
    
    # Aggregate across seeds: mean per bin
    print(f"\n\n{'='*90}")
    print(f"  AGGREGATE (3-seed mean)")
    print(f"{'='*90}")
    
    agg = defaultdict(lambda: defaultdict(list))
    for r in all_rows:
        key = (r["subtask"], r["bin"])
        for col in ["apiar_sr", "repo_sr", "delta_sr", "apiar_srxsim", "repo_srxsim", "delta_srxsim", "n"]:
            agg[key][col].append(r[col])
    
    print(f"\n  {'Subtask':<8} | {'Bin':<5} | {'n':>5} | {'APIAR SR':>9} | {'RePO SR':>8} | {'Δ SR':>7} | {'APIAR SR×S':>11} | {'RePO SR×S':>10} | {'Δ SR×S':>8}")
    print(f"  {'-'*90}")
    
    for subtask in SUBTASKS:
        for b in range(1, N_BINS+1):
            key = (subtask, b)
            if key not in agg:
                continue
            d = agg[key]
            print(f"  {subtask:<8} | Q{b:<4} | {np.mean(d['n']):>5.0f} | {np.mean(d['apiar_sr']):>8.4f} | {np.mean(d['repo_sr']):>7.4f} | {np.mean(d['delta_sr']):>+6.4f} | {np.mean(d['apiar_srxsim']):>10.4f} | {np.mean(d['repo_srxsim']):>9.4f} | {np.mean(d['delta_srxsim']):>+7.4f}")
        print(f"  {'-'*90}")
    
    # Save
    out_csv = os.path.join(out_dir, "e2_gap_bucketed_analysis.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
