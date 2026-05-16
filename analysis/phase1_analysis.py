#!/usr/bin/env python3
"""
Phase 1 Analysis — Learning Curves & Diagnostic Plots

Extracts training metrics from trainer_state.json files across
multiple step budgets and generates comparison plots.

Usage:
    python phase1_analysis.py [--output_dir plots/phase1]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_DIR = "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO"
ADA_DIR = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo"

# Mapping of run configurations
RUNS = {
    # Phase 0 (step-60) runs
    ("RePO", "MR", 60): f"{REPO_DIR}/output/repo_3B_MR",
    ("RePO", "QED", 60): f"{REPO_DIR}/output/repo_3B_QED",
    ("RePO", "LogP", 60): f"{REPO_DIR}/output/repo_3B_LogP",
    ("AdaRePO", "MR", 60): f"{ADA_DIR}/output/ada_repo_3B_MR",
    ("AdaRePO", "QED", 60): f"{ADA_DIR}/output/ada_repo_3B_QED",
    ("AdaRePO", "LogP", 60): f"{ADA_DIR}/output/ada_repo_3B_LogP",
    # Phase 1 (step-scaling) runs
    ("RePO", "MR", 120): f"{REPO_DIR}/output/p1_repo_3B_MR_s120",
    ("RePO", "MR", 240): f"{REPO_DIR}/output/p1_repo_3B_MR_s240",
    ("RePO", "MR", 480): f"{REPO_DIR}/output/p1_repo_3B_MR_s480",
    ("AdaRePO", "MR", 120): f"{ADA_DIR}/output/p1_ada_repo_3B_MR_s120",
    ("AdaRePO", "MR", 240): f"{ADA_DIR}/output/p1_ada_repo_3B_MR_s240",
    ("AdaRePO", "MR", 480): f"{ADA_DIR}/output/p1_ada_repo_3B_MR_s480",
    ("RePO", "QED", 120): f"{REPO_DIR}/output/p1_repo_3B_QED_s120",
    ("RePO", "QED", 240): f"{REPO_DIR}/output/p1_repo_3B_QED_s240",
    ("RePO", "QED", 480): f"{REPO_DIR}/output/p1_repo_3B_QED_s480",
    ("AdaRePO", "QED", 120): f"{ADA_DIR}/output/p1_ada_repo_3B_QED_s120",
    ("AdaRePO", "QED", 240): f"{ADA_DIR}/output/p1_ada_repo_3B_QED_s240",
    ("AdaRePO", "QED", 480): f"{ADA_DIR}/output/p1_ada_repo_3B_QED_s480",
}


def load_trainer_state(output_dir):
    """Load trainer_state.json and extract per-step metrics."""
    ts_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(ts_path):
        return None

    with open(ts_path) as f:
        state = json.load(f)

    records = []
    for entry in state.get("log_history", []):
        if "loss" in entry and entry["loss"] is not None:
            records.append(entry)

    if not records:
        return None

    df = pd.DataFrame(records)
    return df


def collect_all_metrics():
    """Collect metrics from all available runs."""
    all_data = []

    for (method, subtask, max_steps), output_dir in RUNS.items():
        df = load_trainer_state(output_dir)
        if df is None:
            continue

        df["method"] = method
        df["subtask"] = subtask
        df["max_steps"] = max_steps
        all_data.append(df)

    if not all_data:
        print("No trainer states found!")
        return pd.DataFrame()

    combined = pd.concat(all_data, ignore_index=True)
    return combined


def load_eval_results():
    """Load evaluation results from Phase 0 and Phase 1 directories."""
    records = []

    # Phase 1 eval results: phase1/{run_key}/checkpoint-{N}/open_generation/MolOpt/{Sub}_summary.csv
    p1_base = os.path.join(ADA_DIR, "evaluation_results/phase1")
    if os.path.isdir(p1_base):
        for run_key in os.listdir(p1_base):
            run_dir = os.path.join(p1_base, run_key)
            if not os.path.isdir(run_dir):
                continue
            method = "AdaRePO" if "ada" in run_key else "RePO"
            # parse step budget from run_key, e.g. repo_MR_s120 -> 120
            parts = run_key.split("_s")
            step_budget = int(parts[-1]) if len(parts) > 1 else 60
            subtask_key = run_key.split("_")[1] if "ada" not in run_key else run_key.split("_")[2]
            for csv_path in Path(run_dir).rglob("*_summary.csv"):
                subtask = csv_path.stem.replace("_summary", "")
                # extract checkpoint number from path
                for p in csv_path.parts:
                    if p.startswith("checkpoint-"):
                        ckpt = int(p.split("-")[1])
                        break
                else:
                    ckpt = step_budget
                try:
                    row = pd.read_csv(csv_path).iloc[0]
                    records.append({
                        "method": method,
                        "subtask": subtask,
                        "step_budget": step_budget,
                        "checkpoint": ckpt,
                        "success_rate": row["success_rate"],
                        "validity": row["validity"],
                        "similarity": row["similarity"],
                    })
                except Exception:
                    continue

    # Phase 0 eval results (legacy dirs): {method}_{subtask}/checkpoint-60/...
    eval_base = os.path.join(ADA_DIR, "evaluation_results")
    for name in os.listdir(eval_base):
        if name == "phase1":
            continue
        summary_glob = Path(eval_base) / name
        if not summary_glob.is_dir():
            continue
        for csv_path in summary_glob.rglob("*_summary.csv"):
            subtask = csv_path.stem.replace("_summary", "")
            method = "AdaRePO" if "ada" in name else "RePO"
            for p in csv_path.parts:
                if p.startswith("checkpoint-"):
                    ckpt = int(p.split("-")[1])
                    break
            else:
                ckpt = 60
            try:
                row = pd.read_csv(csv_path).iloc[0]
                records.append({
                    "method": method,
                    "subtask": subtask,
                    "step_budget": 60,
                    "checkpoint": ckpt,
                    "success_rate": row["success_rate"],
                    "validity": row["validity"],
                    "similarity": row["similarity"],
                })
            except Exception:
                continue

    # Deduplicate (prefer Phase 1 over Phase 0 for same method/subtask/checkpoint)
    if records:
        df = pd.DataFrame(records)
        df = df.drop_duplicates(subset=["method", "subtask", "step_budget", "checkpoint"], keep="first")
        return df
    return pd.DataFrame()


def plot_learning_curves(df, output_dir, subtask):
    """Plot loss, reward, KL, and beta curves for a single subtask."""
    sub_df = df[df["subtask"] == subtask]
    if sub_df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Phase 1 Learning Curves — {subtask}", fontsize=16)

    metrics = [
        ("loss", "Loss", axes[0, 0]),
        ("rewards/smile_optimization", "Reward (smile_opt)", axes[0, 1]),
        ("kl", "KL Divergence", axes[0, 2]),
        ("s_loss", "Guidance Loss (s_loss)", axes[1, 0]),
        ("beta_guide_mean", "Beta Guide Mean", axes[1, 1]),
        ("v_top_minus_v_ref", "v_top - v_ref", axes[1, 2]),
    ]

    colors = {"RePO": "#2196F3", "AdaRePO": "#FF5722"}
    linestyles = {60: "-", 120: "--", 240: "-.", 480: ":"}

    for col, label, ax in metrics:
        if col not in sub_df.columns:
            ax.set_visible(False)
            continue

        for (method, max_steps), group in sub_df.groupby(["method", "max_steps"]):
            valid = group.dropna(subset=[col])
            if valid.empty:
                continue
            ax.plot(
                valid["step"], valid[col],
                color=colors.get(method, "gray"),
                linestyle=linestyles.get(max_steps, "-"),
                label=f"{method} (s={max_steps})",
                alpha=0.8,
                linewidth=1.5,
            )

        ax.set_xlabel("Step")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, f"learning_curves_{subtask}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_eval_scaling(eval_df, output_dir):
    """Plot success rate vs step budget from evaluation results."""
    if eval_df.empty:
        print("  No evaluation results to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Phase 1 — Success Rate vs Training Steps", fontsize=14)

    colors = {"RePO": "#2196F3", "AdaRePO": "#FF5722"}

    for i, subtask in enumerate(["MR", "QED"]):
        ax = axes[i]
        sub = eval_df[eval_df["subtask"] == subtask]
        if sub.empty:
            ax.set_title(f"{subtask} — No data")
            continue

        for method, group in sub.groupby("method"):
            # Use final checkpoint for each step budget
            final = group.loc[group.groupby("step_budget")["checkpoint"].idxmax()]
            final = final.sort_values("step_budget")
            ax.plot(
                final["step_budget"], final["success_rate"],
                "o-", color=colors.get(method, "gray"),
                label=method, linewidth=2, markersize=8,
            )

        ax.set_xlabel("Training Steps")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"{subtask}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([60, 120, 240, 480])

    plt.tight_layout()
    path = os.path.join(output_dir, "eval_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def generate_summary_table(df, eval_df, output_dir):
    """Generate a markdown summary table of all available results."""
    lines = ["# Phase 1 — Results Summary\n"]

    # Training metrics summary
    lines.append("## Training Metrics (final step per run)\n")
    lines.append("| Method | Subtask | Steps | Final Loss | Final Reward | Final KL | Final Beta |")
    lines.append("|--------|---------|-------|-----------|-------------|----------|------------|")

    for (method, subtask, max_steps), group in df.groupby(["method", "subtask", "max_steps"]):
        last = group.iloc[-1]
        loss = f"{last.get('loss', 'N/A'):.4f}" if pd.notna(last.get("loss")) else "N/A"
        reward = f"{last.get('rewards/smile_optimization', 'N/A'):.4f}" if pd.notna(last.get("rewards/smile_optimization")) else "N/A"
        kl = f"{last.get('kl', 'N/A'):.4f}" if pd.notna(last.get("kl")) else "N/A"
        beta = f"{last.get('beta_guide_mean', 'N/A'):.4f}" if pd.notna(last.get("beta_guide_mean")) else "N/A"
        lines.append(f"| {method} | {subtask} | {max_steps} | {loss} | {reward} | {kl} | {beta} |")

    # Evaluation results
    if not eval_df.empty:
        lines.append("\n## Evaluation Results\n")
        lines.append("| Method | Subtask | Steps | Checkpoint | Success Rate | Validity | Similarity |")
        lines.append("|--------|---------|-------|-----------|-------------|----------|------------|")

        for _, row in eval_df.iterrows():
            lines.append(
                f"| {row['method']} | {row['subtask']} | {row.get('step_budget', 60)} | "
                f"{row.get('checkpoint', 60)} | {row['success_rate']:.3f} | "
                f"{row['validity']:.3f} | {row['similarity']:.3f} |"
            )

    path = os.path.join(output_dir, "phase1_results.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Analysis")
    parser.add_argument("--output_dir", default=os.path.join(ADA_DIR, "analysis/plots/phase1"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Phase 1 Analysis")
    print("=" * 50)

    # Collect training metrics
    print("\n1. Collecting training metrics...")
    df = collect_all_metrics()
    if df.empty:
        print("  No training data found!")
        return

    available = df.groupby(["method", "subtask", "max_steps"]).size()
    print(f"  Found {len(available)} runs:")
    for (method, subtask, steps), count in available.items():
        print(f"    {method}/{subtask}/s{steps}: {count} log entries")

    # Load evaluation results
    print("\n2. Loading evaluation results...")
    eval_df = load_eval_results()
    if not eval_df.empty:
        print(f"  Found {len(eval_df)} evaluation entries")
    else:
        print("  No evaluation results found (run evaluate_phase1.sh first)")

    # Generate plots
    print("\n3. Generating plots...")
    for subtask in ["MR", "QED", "LogP"]:
        if subtask in df["subtask"].values:
            plot_learning_curves(df, args.output_dir, subtask)

    plot_eval_scaling(eval_df, args.output_dir)

    # Generate summary table
    print("\n4. Generating summary table...")
    generate_summary_table(df, eval_df, args.output_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
