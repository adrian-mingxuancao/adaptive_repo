#!/usr/bin/env python3
"""
C0: Single result-aggregation script for APIAR paper.

Scans ALL evaluation result directories (predictions/, evaluation_results/)
and produces paper-ready tables with mean ± SE (3 seeds).

Outputs:
  results/aggregated_results.csv        — full table with mean, std, ci95, n_seeds
  results/table1_main_srxsim.csv        — paper Table 1 (SR×Sim, mean±SE)
  results/table1b_per_metric.csv        — appendix per-metric SR, Sim
  results/table2_ablation.csv           — ablation factorization
  results/table_significance.csv        — copy of e1a significance tests

Usage:
    python scripts/aggregate_results.py [--output_dir ./results]
    python scripts/aggregate_results.py --print-table
"""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

# t-values for 95% CI (two-tailed) by df, avoids scipy dependency on login node
_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
        7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        30: 2.042, 50: 2.009, 100: 1.984}

def _t_val(df):
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95.keys())
    for k in reversed(keys):
        if k <= df:
            return _T95[k]
    return 1.96


REPO_ROOT = Path("/lus/eagle/projects/IMPROVE_Aim1/caom")
ADA_ROOT = REPO_ROOT / "agent_drug_discovery" / "adaptive_repo"

# All directories to scan for *_summary.csv
EVAL_DIRS = [
    REPO_ROOT / "RePO" / "predictions",
    ADA_ROOT / "evaluation_results" / "v16v17ms",
    ADA_ROOT / "evaluation_results" / "v18ms",
    ADA_ROOT / "evaluation_results" / "abl_beta",
    ADA_ROOT / "evaluation_results" / "abl_bank",
    ADA_ROOT / "evaluation_results" / "batch_eval",
    ADA_ROOT / "evaluation_results" / "repo_hard",
]

# ---- Method registry: map dir name fragments to (paper_label, seed) ----
METHOD_REGISTRY = {
    # Zero-shot
    "Qwen2.5-3B-Instruct": ("Zero-shot", None),
    # RePO baseline
    "v16v17ms_repo_s42":  ("RePO", 42),
    "v16v17ms_repo_s123": ("RePO", 123),
    "v16v17ms_repo_s456": ("RePO", 456),
    "repo_s42checkpoint-120":  ("RePO", 42),
    "repo_s123checkpoint-120": ("RePO", 123),
    "repo_s456checkpoint-120": ("RePO", 456),
    # GRPO
    "grpo_baseline_s42":  ("GRPO", 42),
    "grpo_baseline_s123": ("GRPO", 123),
    "grpo_baseline_s456": ("GRPO", 456),
    # C1: Iterative SFT
    "isft_s42":  ("C1: Iterative SFT", 42),
    "isft_s123": ("C1: Iterative SFT", 123),
    "isft_s456": ("C1: Iterative SFT", 456),
    # C2: Offline-Strengthened
    "c2_offline_s42":  ("C2: Offline-Str.", 42),
    "c2_offline_s123": ("C2: Offline-Str.", 123),
    "c2_offline_s456": ("C2: Offline-Str.", 456),
    # APIAR (v16 = full)
    "v16v17ms_v16_s42":  ("APIAR", 42),
    "v16v17ms_v16_s123": ("APIAR", 123),
    "v16v17ms_v16_s456": ("APIAR", 456),
    # Ablation: beta-only
    "ablation_beta_only_s42":  ("Abl: beta-only", 42),
    "ablation_beta_only_s123": ("Abl: beta-only", 123),
    "ablation_beta_only_s456": ("Abl: beta-only", 456),
    "ablation_beta_only_s42checkpoint-120":  ("Abl: beta-only", 42),
    "ablation_beta_only_s123checkpoint-120": ("Abl: beta-only", 123),
    "ablation_beta_only_s456checkpoint-120": ("Abl: beta-only", 456),
    # Ablation: bank-only
    "ablation_bank_only_s42":  ("Abl: bank-only", 42),
    "ablation_bank_only_s123": ("Abl: bank-only", 123),
    "ablation_bank_only_s456": ("Abl: bank-only", 456),
    "ablation_bank_only_s42checkpoint-120":  ("Abl: bank-only", 42),
    "ablation_bank_only_s123checkpoint-120": ("Abl: bank-only", 123),
    "ablation_bank_only_s456checkpoint-120": ("Abl: bank-only", 456),
    # APIAR-Hard (v18)
    "v18_hard_s42checkpoint-120":  ("APIAR-Hard", 42),
    "v18_hard_s123checkpoint-120": ("APIAR-Hard", 123),
    "v18_hard_s456checkpoint-120": ("APIAR-Hard", 456),
    "v18ms_hard_s42":  ("APIAR-Hard", 42),
    "v18ms_hard_s123": ("APIAR-Hard", 123),
    "v18ms_hard_s456": ("APIAR-Hard", 456),
    # RePO-Hard
    "repo_hard_s42":  ("RePO-Hard", 42),
    "repo_hard_s123": ("RePO-Hard", 123),
    "repo_hard_s456": ("RePO-Hard", 456),
    "repo_hard_s42checkpoint-120":  ("RePO-Hard", 42),
    "repo_hard_s123checkpoint-120": ("RePO-Hard", 123),
    "repo_hard_s456checkpoint-120": ("RePO-Hard", 456),
}


def read_summary(csv_path):
    """Read a single summary CSV and return dict of metrics."""
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    row = rows[0]
    return {k: float(v) for k, v in row.items() if v}


def _identify(parts):
    """Try to identify (method, seed) from path components."""
    # Strategy 1: direct match on any ancestor
    for p in parts:
        if p in METHOD_REGISTRY and METHOD_REGISTRY[p] is not None:
            return METHOD_REGISTRY[p]
    # Strategy 2: parent of checkpoint-NNN
    for i, p in enumerate(parts):
        if p.startswith("checkpoint-") and i > 0:
            parent = parts[i - 1]
            if parent in METHOD_REGISTRY and METHOD_REGISTRY[parent] is not None:
                return METHOD_REGISTRY[parent]
    # Strategy 3: parent of open_generation
    for i, p in enumerate(parts):
        if p == "open_generation" and i > 0:
            c = parts[i - 1]
            if c in METHOD_REGISTRY and METHOD_REGISTRY[c] is not None:
                return METHOD_REGISTRY[c]
    return None, None


def scan_all_eval_dirs():
    """
    Scan ALL eval directories and collect summary CSVs.
    Returns: list of dicts with keys: method, seed, subtask, metrics
    """
    records = []
    seen = set()  # (method, seed, subtask) dedup
    for base_dir in EVAL_DIRS:
        if not base_dir.exists():
            continue
        for summary_csv in sorted(base_dir.rglob("*_summary.csv")):
            parts = summary_csv.parts
            subtask = summary_csv.stem.replace("_summary", "")
            if subtask not in ("LogP", "MR", "QED"):
                continue

            method, seed = _identify(parts)
            if method is None:
                continue

            key = (method, seed, subtask)
            if key in seen:
                continue
            seen.add(key)

            metrics = read_summary(summary_csv)
            if not metrics:
                continue

            sr = metrics.get("success_rate", 0)
            sim = metrics.get("similarity", 0)
            metrics["sr_x_sim"] = sr * sim

            records.append({
                "method": method,
                "seed": seed,
                "subtask": subtask,
                "metrics": metrics,
            })

    return records


def aggregate(records):
    """
    Group records by (method, subtask, metric) and compute mean, std, 95% CI.
    """
    groups = defaultdict(list)
    for r in records:
        key = (r["method"], r["subtask"])
        groups[key].append((r["seed"], r["metrics"]))

    results = []
    for (method, subtask), entries in sorted(groups.items()):
        seeds = [e[0] for e in entries]
        metric_keys = set()
        for _, m in entries:
            metric_keys.update(m.keys())

        for metric in sorted(metric_keys):
            values = [e[1].get(metric, None) for e in entries]
            values = [v for v in values if v is not None]
            if not values:
                continue

            n = len(values)
            mean = sum(values) / n
            if n > 1:
                std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
                se = std / math.sqrt(n)
                t_val = _t_val(n - 1)
                ci_lo = mean - t_val * se
                ci_hi = mean + t_val * se
            else:
                std = 0.0
                ci_lo = mean
                ci_hi = mean

            results.append({
                "method": method,
                "subtask": subtask,
                "metric": metric,
                "mean": mean,
                "std": std,
                "ci95_lo": ci_lo,
                "ci95_hi": ci_hi,
                "n_seeds": n,
                "seed_list": ",".join(str(s) for s in sorted(seeds) if s is not None),
            })

    return results


def write_csv(results, output_path):
    """Write aggregated results to CSV."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["method", "subtask", "metric",
                  "mean", "std", "ci95_lo", "ci95_hi", "n_seeds", "seed_list"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v
                             for k, v in row.items()})
    print(f"Wrote {len(results)} rows to {output_path}")


def print_table(results):
    """Pretty-print paper-ready comparison table with mean ± SE."""
    # Build: method -> subtask -> metric -> (mean, std, n)
    table = defaultdict(lambda: defaultdict(dict))
    for r in results:
        table[r["method"]][r["subtask"]][r["metric"]] = (r["mean"], r["std"], r["n_seeds"])

    METHODS_ORDER = [
        "Zero-shot", "GRPO", "C1: Iterative SFT", "C2: Offline-Str.",
        "RePO", "Abl: beta-only", "Abl: bank-only", "APIAR",
        "APIAR-Hard", "RePO-Hard",
    ]
    present = [m for m in METHODS_ORDER if m in table]
    for m in sorted(table.keys()):
        if m not in present:
            present.append(m)

    subtasks = ["LogP", "MR", "QED"]

    def _fmt(m, s, n):
        se = s / math.sqrt(n) if n > 1 else 0
        if n > 1:
            return f"{m:.4f}\u00b1{se:.4f}"
        return f"{m:.4f}"

    # ---- Table 1: SR×Sim ----
    print("\n" + "=" * 90)
    print("Table 1: SR×Sim (mean ± SE, 3 seeds)")
    print("=" * 90)
    header = f"{'Method':25s}"
    for st in subtasks:
        header += f" | {st:>16s}"
    header += " |   Avg"
    print(header)
    print("-" * len(header))
    for method in present:
        row = f"{method:25s}"
        vals = []
        for st in subtasks:
            e = table[method].get(st, {}).get("sr_x_sim")
            if e:
                row += f" | {_fmt(*e):>16s}"
                vals.append(e[0])
            else:
                row += f" |    {'—':>12s}"
        avg = sum(vals) / len(vals) if vals else 0
        row += f" | {avg:.4f}"
        print(row)

    # ---- Table 1b: SR and Sim ----
    for metric, label in [("success_rate", "SR"), ("similarity", "Sim")]:
        print(f"\n--- {label} ---")
        header = f"{'Method':25s}"
        for st in subtasks:
            header += f" | {st:>16s}"
        print(header)
        print("-" * len(header))
        for method in present:
            row = f"{method:25s}"
            for st in subtasks:
                e = table[method].get(st, {}).get(metric)
                if e:
                    row += f" | {_fmt(*e):>16s}"
                else:
                    row += f" |    {'—':>12s}"
            print(row)

    print("\n" + "=" * 90)


def write_paper_tables(results, output_dir):
    """Write per-table CSVs for direct LaTeX ingestion."""
    os.makedirs(output_dir, exist_ok=True)
    table = defaultdict(lambda: defaultdict(dict))
    for r in results:
        n = r["n_seeds"]
        se = r["std"] / math.sqrt(n) if n > 1 else 0
        table[r["method"]][r["subtask"]][r["metric"]] = {
            "mean": r["mean"], "se": se, "n": n,
            "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
        }

    MAIN_METHODS = [
        "Zero-shot", "GRPO", "C1: Iterative SFT", "C2: Offline-Str.",
        "RePO", "APIAR",
    ]
    ABL_METHODS = ["APIAR", "Abl: beta-only", "Abl: bank-only", "RePO"]
    subtasks = ["LogP", "MR", "QED"]

    def _row(method, metric):
        vals = []
        for st in subtasks:
            e = table.get(method, {}).get(st, {}).get(metric, {})
            vals.append(e.get("mean", 0))
        avg = sum(vals) / len(vals) if vals else 0
        return {
            "method": method,
            **{f"{st}_mean": table.get(method, {}).get(st, {}).get(metric, {}).get("mean", "")
               for st in subtasks},
            **{f"{st}_se": table.get(method, {}).get(st, {}).get(metric, {}).get("se", "")
               for st in subtasks},
            "avg": avg,
        }

    # Table 1: main SR×Sim
    t1_path = os.path.join(output_dir, "table1_main_srxsim.csv")
    fields = ["method"] + [f"{st}_{x}" for st in subtasks for x in ("mean", "se")] + ["avg"]
    with open(t1_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in MAIN_METHODS:
            if m in table:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in _row(m, "sr_x_sim").items()})
    print(f"  Wrote {t1_path}")

    # Table 2: ablation
    t2_path = os.path.join(output_dir, "table2_ablation.csv")
    with open(t2_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in ABL_METHODS:
            if m in table:
                w.writerow({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in _row(m, "sr_x_sim").items()})
    print(f"  Wrote {t2_path}")

    # Copy significance table if it exists
    sig_src = ADA_ROOT / "analysis" / "e1a_significance_tests.csv"
    sig_dst = os.path.join(output_dir, "table_significance.csv")
    if sig_src.exists():
        import shutil
        shutil.copy2(sig_src, sig_dst)
        print(f"  Copied {sig_dst}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate per-seed evaluation results")
    parser.add_argument("--output_dir", default=str(ADA_ROOT / "results"), help="Output dir")
    parser.add_argument("--print-table", action="store_true", help="Pretty-print table")
    args = parser.parse_args()

    print("Scanning all eval directories...")
    records = scan_all_eval_dirs()
    print(f"Found {len(records)} summary records")

    if not records:
        print("No records found. Exiting.")
        return

    agg = aggregate(records)

    output_csv = os.path.join(args.output_dir, "aggregated_results.csv")
    write_csv(agg, output_csv)
    write_paper_tables(agg, args.output_dir)

    if args.print_table:
        print_table(agg)


if __name__ == "__main__":
    main()
