#!/usr/bin/env python3
"""Generate paper/writing figures from finalized analysis CSVs."""
import csv
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path("/lus/eagle/projects/IMPROVE_Aim1/caom")
ADA_ROOT = REPO_ROOT / "agent_drug_discovery" / "adaptive_repo"
ANALYSIS_DIR = ADA_ROOT / "analysis"
FIGURE_DIR = ANALYSIS_DIR / "figures"

PRED_DIRS = [
    REPO_ROOT / "RePO" / "predictions",
    ADA_ROOT / "evaluation_results",
]

SUBTASKS = ["LogP", "MR", "QED"]
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

COLORS = {"LogP": "#2f6f9f", "MR": "#b15d2a", "QED": "#3f8f5f"}


def find_detailed_csv(fragments, subtask):
    for base_dir in PRED_DIRS:
        if not base_dir.exists():
            continue
        for frag in fragments:
            candidates = [
                base_dir / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv",
                base_dir / frag / "open_generation" / "MolOpt" / f"{subtask}.csv",
            ]
            for subdir in base_dir.iterdir():
                if subdir.is_dir():
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv")
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}.csv")
            for path in candidates:
                if path.exists():
                    return path
    return None


def mean_and_sem(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, sem


def load_headroom_rows():
    rows = list(csv.DictReader(open(ANALYSIS_DIR / "e2_gap_bucketed_analysis.csv")))
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["subtask"], int(row["bin"]))].append(float(row["delta_srxsim"]))
    return grouped


def plot_headroom():
    grouped = load_headroom_rows()
    x = np.arange(1, 6)

    def series(subtask):
        means, sems = [], []
        for b in x:
            mean, sem = mean_and_sem(grouped[(subtask, int(b))])
            means.append(mean)
            sems.append(sem)
        return np.asarray(means), np.asarray(sems)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    logp_mean, logp_sem = series("LogP")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.errorbar(
        x,
        logp_mean,
        yerr=logp_sem,
        color=COLORS["LogP"],
        marker="o",
        lw=2,
        capsize=3,
        label="LogP",
    )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Headroom quintile")
    ax.set_ylabel("APIAR - RePO SR x Sim")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x])
    ax.set_title("Headroom-conditioned gain")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_headroom_curve.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "e2_headroom_curve.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for subtask in SUBTASKS:
        mean, sem = series(subtask)
        ax.errorbar(
            x,
            mean,
            yerr=sem,
            marker="o",
            lw=1.8,
            capsize=3,
            color=COLORS[subtask],
            label=subtask,
        )
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xlabel("Headroom quintile")
    ax.set_ylabel("APIAR - RePO SR x Sim")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i}" for i in x])
    ax.set_title("Headroom-conditioned gain by subtask")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "e2_headroom_curve_all_subtasks_appendix.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "e2_headroom_curve_all_subtasks_appendix.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def noop_rate(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    flags = []
    for row in rows:
        sim = float(row.get("similarity", 0) or 0)
        success = int(float(row.get("success", 0) or 0))
        flags.append(sim >= 0.98 and success == 0)
    return float(np.mean(flags))


def collect_noop_rates():
    records = []
    for subtask in SUBTASKS:
        for method, seeds in [("APIAR", APIAR_SEEDS), ("RePO", REPO_SEEDS)]:
            for seed, fragments in zip([42, 123, 456], seeds):
                csv_path = find_detailed_csv(fragments, subtask)
                if csv_path is None:
                    print(f"WARNING: missing {method} {subtask} seed {seed}")
                    continue
                records.append(
                    {
                        "method": method,
                        "subtask": subtask,
                        "seed": seed,
                        "no_op_fail": noop_rate(csv_path),
                    }
                )
    return records


def plot_noop_failure():
    records = collect_noop_rates()
    out_csv = ANALYSIS_DIR / "noop_failure_rate.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "subtask", "seed", "no_op_fail"])
        writer.writeheader()
        writer.writerows(records)

    x = np.arange(len(SUBTASKS))
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for offset, method, color in [(-width / 2, "RePO", "#457b9d"), (width / 2, "APIAR", "#e63946")]:
        means, sems = [], []
        for subtask in SUBTASKS:
            vals = [r["no_op_fail"] for r in records if r["method"] == method and r["subtask"] == subtask]
            mean, sem = mean_and_sem(vals)
            means.append(mean)
            sems.append(sem)
        ax.bar(x + offset, means, width=width, yerr=sems, capsize=3, label=method, color=color, alpha=0.9)

    ax.set_ylabel("No-op failure rate")
    ax.set_xticks(x)
    ax.set_xticklabels(SUBTASKS)
    ax.set_ylim(bottom=0)
    ax.set_title("No-op failures by method and subtask")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "noop_failure_rate.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "noop_failure_rate.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    plot_headroom()
    plot_noop_failure()
    print(f"Saved figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
