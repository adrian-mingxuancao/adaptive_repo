#!/usr/bin/env python3
"""
E7a: Training dynamics figure — 4-panel layout.
  Panel A: Reward vs step (APIAR-full, RePO, ablations)
  Panel B: Loss vs step
  Panel C: Average β_guide across batch vs step (APIAR variants only)
  Panel D: Memory bank: total entries + frac_self_distill vs step
All with seed variability shading (mean ± std across 3 seeds).
"""
import json
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

OUT_DIR = Path("/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/figures")

# Run dirs with labels and seeds
ADA_OUT = Path("/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/output")
REPO_OUT = Path("/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/output")

RUNS = {
    "APIAR (full)": {
        "dirs": [ADA_OUT / "v16v17ms_v16_s42", ADA_OUT / "v16v17ms_v16_s123", ADA_OUT / "v16v17ms_v16_s456"],
        "color": "#e63946", "ls": "-", "zorder": 10,
    },
    "RePO": {
        "dirs": [REPO_OUT / "v16v17ms_repo_s42", REPO_OUT / "v16v17ms_repo_s123", REPO_OUT / "v16v17ms_repo_s456"],
        "color": "#457b9d", "ls": "-", "zorder": 9,
    },
    "Abl: β-only": {
        "dirs": [ADA_OUT / "ablation_beta_only_s42", ADA_OUT / "ablation_beta_only_s123", ADA_OUT / "ablation_beta_only_s456"],
        "color": "#f4a261", "ls": "--", "zorder": 8,
    },
    "Abl: Bank-only": {
        "dirs": [ADA_OUT / "ablation_bank_only_s42", ADA_OUT / "ablation_bank_only_s123", ADA_OUT / "ablation_bank_only_s456"],
        "color": "#2a9d8f", "ls": "--", "zorder": 7,
    },
}


def load_log_history(run_dir):
    """Load trainer_state.json log_history."""
    state_file = run_dir / "trainer_state.json"
    if not state_file.exists():
        checkpoint_states = sorted(run_dir.glob("checkpoint-*/trainer_state.json"))
        if checkpoint_states:
            state_file = checkpoint_states[-1]
    if not state_file.exists():
        return None
    with open(state_file) as f:
        state = json.load(f)
    return state.get("log_history", [])


def extract_metric(log_history, metric_key):
    """Extract (steps, values) for a given metric from log_history."""
    steps, values = [], []
    for entry in log_history:
        if metric_key in entry and "step" in entry:
            steps.append(entry["step"])
            values.append(entry[metric_key])
    return np.array(steps), np.array(values)


def get_multi_seed(run_cfg, metric_key, smooth_window=5):
    """Get mean ± std across seeds for a metric."""
    all_vals = []
    steps = None
    for d in run_cfg["dirs"]:
        hist = load_log_history(d)
        if hist is None:
            continue
        s, v = extract_metric(hist, metric_key)
        if len(v) == 0:
            continue
        all_vals.append(clean_metric_values(v, metric_key))
        if steps is None:
            steps = s
    
    if not all_vals or steps is None:
        return None, None, None
    
    # Align to shortest
    min_len = min(len(v) for v in all_vals)
    all_vals = [v[:min_len] for v in all_vals]
    steps = steps[:min_len]
    
    arr = np.array(all_vals)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    
    # Smooth without zero-padding edge artifacts.
    if smooth_window > 1 and len(mean) > smooth_window:
        mean = rolling_nanmean(mean, smooth_window)
        std = rolling_nanmean(std, smooth_window)
    
    return steps, mean, std


def clean_metric_values(values, metric_key):
    """Remove isolated numerical blow-ups before plotting aggregate curves."""
    arr = np.asarray(values, dtype=float).copy()
    arr[~np.isfinite(arr)] = np.nan
    if metric_key == "loss":
        # RePO s456 has one logged numerical blow-up at step 51
        # (loss=207.884, KL=6852.54, grad_norm=43074.5) with normal
        # neighboring steps. Treat such isolated points as logging/training
        # transients so they do not dominate the paper figure.
        for i, value in enumerate(arr):
            lo = max(0, i - 3)
            hi = min(len(arr), i + 4)
            neighbors = np.concatenate([arr[lo:i], arr[i + 1 : hi]])
            neighbors = neighbors[np.isfinite(neighbors)]
            if len(neighbors) == 0:
                continue
            local_median = np.median(neighbors)
            if value > 5.0 and value > 20.0 * max(local_median, 1e-6):
                arr[i] = np.nan
    return interpolate_nans(arr)


def interpolate_nans(values):
    arr = np.asarray(values, dtype=float)
    if not np.isnan(arr).any():
        return arr
    x = np.arange(len(arr))
    good = np.isfinite(arr)
    if good.sum() == 0:
        return arr
    arr[~good] = np.interp(x[~good], x[good], arr[good])
    return arr


def rolling_nanmean(values, window):
    arr = np.asarray(values, dtype=float)
    half = window // 2
    out = np.empty_like(arr)
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = np.nanmean(arr[lo:hi])
    return out


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("APIAR Training Dynamics (3 seeds, mean ± std)", fontsize=14, fontweight='bold')
    
    # Panel A: Reward
    ax = axes[0, 0]
    ax.set_title("(a) Reward vs Training Step", fontsize=11)
    for label, cfg in RUNS.items():
        steps, mean, std = get_multi_seed(cfg, "reward")
        if steps is None:
            continue
        ax.plot(steps, mean, label=label, color=cfg["color"], ls=cfg["ls"], lw=2, zorder=cfg["zorder"])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=cfg["color"], zorder=cfg["zorder"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    
    # Panel B: Loss
    ax = axes[0, 1]
    ax.set_title("(b) Training Loss vs Step", fontsize=11)
    for label, cfg in RUNS.items():
        steps, mean, std = get_multi_seed(cfg, "loss")
        if steps is None:
            continue
        ax.plot(steps, mean, label=label, color=cfg["color"], ls=cfg["ls"], lw=2, zorder=cfg["zorder"])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=cfg["color"], zorder=cfg["zorder"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_ylim(-0.5, 3.0)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    
    # Panel C: Beta guide
    ax = axes[1, 0]
    ax.set_title("(c) Average β_guide vs Step", fontsize=11)
    for label, cfg in RUNS.items():
        steps, mean, std = get_multi_seed(cfg, "beta_guide_mean")
        if steps is None:
            continue
        ax.plot(steps, mean, label=label, color=cfg["color"], ls=cfg["ls"], lw=2, zorder=cfg["zorder"])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=cfg["color"], zorder=cfg["zorder"])
    ax.set_xlabel("Step")
    ax.set_ylabel("β_guide (batch mean)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    
    # Panel D: Memory bank entries + frac_self_distill
    ax = axes[1, 1]
    ax.set_title("(d) Memory Bank Usage vs Step", fontsize=11)
    
    # Plot total_entries on left y-axis
    for label, cfg in RUNS.items():
        steps, mean, std = get_multi_seed(cfg, "memory_bank/total_entries")
        if steps is None:
            continue
        ax.plot(steps, mean, label=f"{label} (entries)", color=cfg["color"], ls=cfg["ls"], lw=2, zorder=cfg["zorder"])
        ax.fill_between(steps, mean - std, mean + std, alpha=0.15, color=cfg["color"], zorder=cfg["zorder"])
    
    ax.set_xlabel("Step")
    ax.set_ylabel("Memory Bank Entries", color='#333')
    ax.grid(alpha=0.3)
    
    # Plot frac_self_distill on right y-axis
    ax2 = ax.twinx()
    for label, cfg in RUNS.items():
        steps, mean, std = get_multi_seed(cfg, "frac_self_distill")
        if steps is None:
            continue
        ax2.plot(steps, mean, label=f"{label} (frac_distill)", color=cfg["color"], ls=":", lw=1.5, zorder=cfg["zorder"])
    ax2.set_ylabel("Frac Self-Distill", color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center left")
    
    plt.tight_layout()
    
    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        out_path = OUT_DIR / f"e7a_training_dynamics_4panel.{ext}"
        fig.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {out_path}")
    
    plt.close()


if __name__ == "__main__":
    main()
