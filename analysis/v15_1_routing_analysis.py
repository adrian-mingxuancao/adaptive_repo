"""
Retrospective analysis of per-example reward statistics from v15.1 retrain logs.
Extract per-completion rewards, group by G=4, compute per-example (mu, sigma, max),
and classify into the 4 routing cases (A/B/C/D).

Outputs saved to analysis/v15_1_retrain_assets/
"""
import ast
import re
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

OUT_DIR = os.path.join(os.path.dirname(__file__), "v15_1_retrain_assets")
os.makedirs(OUT_DIR, exist_ok=True)

G = 4  # num_generations

# ── Parse logs: extract per-completion rewards and step boundaries ────────────
def parse_rewards_and_steps(log_paths):
    """Parse Total reward values and step log lines from training logs."""
    all_rewards = []
    step_logs = []
    
    for path in log_paths:
        with open(path) as f:
            for line in f:
                s = line.strip()
                # Per-completion reward
                m = re.search(r"Total reward: (-?[\d.]+(?:e[+-]?\d+)?)", s)
                if m:
                    all_rewards.append(float(m.group(1)))
                # Step log (batch-level metrics)
                if s.startswith("{'loss'"):
                    try:
                        d = ast.literal_eval(s)
                        step_logs.append(d)
                    except Exception:
                        pass
    return all_rewards, step_logs

ada_logs = [
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_812647.out",
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_813476.out",
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_814309.out",
]

repo_logs = [
    "/net/scratch/caom/repo_project/logs/repo_propopt_791601.out",
    "/net/scratch/caom/repo_project/logs/repo_propopt_resume_795397.out",
]

print("Parsing AdaRePO v15.1 logs...")
ada_rewards, ada_steps = parse_rewards_and_steps(ada_logs)
print(f"  Total reward values: {len(ada_rewards)}, Step logs: {len(ada_steps)}")

print("Parsing RePO logs...")
repo_rewards, repo_steps = parse_rewards_and_steps(repo_logs)
print(f"  Total reward values: {len(repo_rewards)}, Step logs: {len(repo_steps)}")

# ── Group rewards by example (G=4 completions per example) ───────────────────
# Each step has batch_size * G completions. batch_size = per_device * num_processes = 2 * 2 = 4
# So each step has 4 examples × 4 generations = 16 rewards
# But with 2 processes, each process prints its own rewards. The reward function
# is called per-process so we see per_device_batch * G = 2*4 = 8 rewards per process per step.
# With 2 processes printing interleaved, we get ~16 per step.

# Strategy: group all rewards into chunks of G=4 to get per-example stats.
# This is approximate since multi-process output may interleave, but the reward
# function processes one batch at a time, so groups of G should be coherent.

def compute_example_stats(rewards, G=4):
    """Group rewards into examples of G completions and compute stats."""
    n = len(rewards)
    n_examples = n // G
    rewards = np.array(rewards[:n_examples * G]).reshape(-1, G)
    
    mu = rewards.mean(axis=1)
    sigma = rewards.std(axis=1)
    max_r = rewards.max(axis=1)
    min_r = rewards.min(axis=1)
    
    return pd.DataFrame({
        "mu": mu,
        "sigma": sigma,
        "max_r": max_r,
        "min_r": min_r,
        "spread": max_r - min_r,
    })

print("\nComputing per-example stats...")
df_ada = compute_example_stats(ada_rewards, G)
df_repo = compute_example_stats(repo_rewards, G)
print(f"  AdaRePO examples: {len(df_ada)}")
print(f"  RePO examples: {len(df_repo)}")

# ── Classify into routing cases ──────────────────────────────────────────────
# Use adaptive thresholds based on distribution percentiles
def classify_examples(df, sigma_thresh=None, mu_thresh=None):
    """Classify examples into 4 routing cases."""
    if sigma_thresh is None:
        sigma_thresh = df["sigma"].median()
    if mu_thresh is None:
        mu_thresh = df["mu"].median()
    
    conditions = [
        (df["sigma"] <= sigma_thresh) & (df["mu"] >= mu_thresh),    # A: stable & good
        (df["sigma"] <= sigma_thresh) & (df["mu"] < mu_thresh),     # B: stable & bad (stuck)
        (df["sigma"] > sigma_thresh) & (df["max_r"] >= mu_thresh),  # C: promising (high var, high max)
        (df["sigma"] > sigma_thresh) & (df["max_r"] < mu_thresh),   # D: noisy/hard
    ]
    labels = ["A: Stable Good", "B: Stuck", "C: Promising", "D: Noisy/Hard"]
    
    df = df.copy()
    df["case"] = "unknown"
    for cond, label in zip(conditions, labels):
        df.loc[cond, "case"] = label
    
    return df

# Use global medians as thresholds
sigma_thresh_ada = df_ada["sigma"].median()
mu_thresh_ada = df_ada["mu"].median()
sigma_thresh_repo = df_repo["sigma"].median()
mu_thresh_repo = df_repo["mu"].median()

print(f"\nAdaRePO thresholds: sigma={sigma_thresh_ada:.4f}, mu={mu_thresh_ada:.4f}")
print(f"RePO thresholds:    sigma={sigma_thresh_repo:.4f}, mu={mu_thresh_repo:.4f}")

df_ada = classify_examples(df_ada, sigma_thresh_ada, mu_thresh_ada)
df_repo = classify_examples(df_repo, sigma_thresh_repo, mu_thresh_repo)

# ── Add epoch-like progress index ────────────────────────────────────────────
# Total steps = 748, examples_per_step ≈ 4 (batch_size=4 across 2 GPUs)
# But we only see per-process rewards, so effective examples ≈ 2 per step per process
# Assign a progress fraction [0, 1] to each example
df_ada["progress"] = np.linspace(0, 1, len(df_ada))
df_repo["progress"] = np.linspace(0, 1, len(df_repo))

# Bin into training phases (quartiles)
df_ada["phase"] = pd.cut(df_ada["progress"], bins=[0, 0.25, 0.5, 0.75, 1.0],
                          labels=["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"])
df_repo["phase"] = pd.cut(df_repo["progress"], bins=[0, 0.25, 0.5, 0.75, 1.0],
                           labels=["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"])

# ── Print summary statistics ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ROUTING CASE DISTRIBUTION")
print("=" * 70)

for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    print(f"\n--- {label} ---")
    total = len(df)
    for case in ["A: Stable Good", "B: Stuck", "C: Promising", "D: Noisy/Hard"]:
        subset = df[df["case"] == case]
        pct = len(subset) / total * 100
        print(f"  {case}: {len(subset):>6} ({pct:5.1f}%) | mu={subset['mu'].mean():.4f}, "
              f"sigma={subset['sigma'].mean():.4f}, max={subset['max_r'].mean():.4f}")

print("\n" + "=" * 70)
print("CASE DISTRIBUTION BY TRAINING PHASE")
print("=" * 70)

for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    print(f"\n--- {label} ---")
    phase_case = df.groupby(["phase", "case"]).size().unstack(fill_value=0)
    phase_totals = df.groupby("phase").size()
    phase_pct = phase_case.div(phase_totals, axis=0) * 100
    print(phase_pct.round(1).to_string())

# ── Figure 1: Scatter plot (mu vs sigma) colored by case ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Per-Example Reward Statistics: μ vs σ (Routing Cases)", fontsize=14, fontweight="bold")

case_colors = {
    "A: Stable Good": "#4CAF50",
    "B: Stuck": "#F44336",
    "C: Promising": "#2196F3",
    "D: Noisy/Hard": "#FF9800",
}

for ax, (label, df) in zip(axes, [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]):
    for case, color in case_colors.items():
        subset = df[df["case"] == case]
        # Subsample for readability
        n_plot = min(2000, len(subset))
        if len(subset) > n_plot:
            subset = subset.sample(n_plot, random_state=42)
        ax.scatter(subset["mu"], subset["sigma"], c=color, alpha=0.3, s=8, label=case)
    ax.set_xlabel("μ (reward mean)")
    ax.set_ylabel("σ (reward std)")
    ax.set_title(label)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.2)
    # Draw threshold lines
    thresh_sigma = sigma_thresh_ada if "Ada" in label else sigma_thresh_repo
    thresh_mu = mu_thresh_ada if "Ada" in label else mu_thresh_repo
    ax.axhline(thresh_sigma, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.axvline(thresh_mu, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_routing_scatter.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved fig_routing_scatter.png")
plt.close()

# ── Figure 2: Case distribution over training (stacked area) ────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Routing Case Distribution Over Training", fontsize=14, fontweight="bold")

case_order = ["A: Stable Good", "B: Stuck", "C: Promising", "D: Noisy/Hard"]

for ax, (label, df) in zip(axes, [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]):
    # Bin into 20 windows
    n_bins = 20
    df_tmp = df.copy()
    df_tmp["bin"] = pd.cut(df_tmp["progress"], bins=n_bins, labels=False)
    
    fracs = []
    for b in range(n_bins):
        bin_data = df_tmp[df_tmp["bin"] == b]
        total = len(bin_data)
        if total == 0:
            fracs.append({c: 0 for c in case_order})
        else:
            fracs.append({c: len(bin_data[bin_data["case"] == c]) / total * 100 for c in case_order})
    
    fracs_df = pd.DataFrame(fracs)
    x = np.linspace(0, 4, n_bins)  # epochs
    
    colors_list = [case_colors[c] for c in case_order]
    ax.stackplot(x, *[fracs_df[c].values for c in case_order],
                 labels=case_order, colors=colors_list, alpha=0.7)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Percentage (%)")
    ax.set_title(label)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_routing_over_training.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_routing_over_training.png")
plt.close()

# ── Figure 3: Reward mu/sigma evolution over training ────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Per-Example Reward Evolution Over Training", fontsize=14, fontweight="bold")

n_bins = 30
for col_idx, (label, df) in enumerate([("AdaRePO v15.1", df_ada), ("RePO", df_repo)]):
    df_tmp = df.copy()
    df_tmp["bin"] = pd.cut(df_tmp["progress"], bins=n_bins, labels=False)
    
    binned = df_tmp.groupby("bin").agg(
        mu_mean=("mu", "mean"),
        mu_std=("mu", "std"),
        sigma_mean=("sigma", "mean"),
        sigma_std=("sigma", "std"),
        max_mean=("max_r", "mean"),
        spread_mean=("spread", "mean"),
    ).reset_index()
    
    x = np.linspace(0, 4, len(binned))
    color = "#FF5722" if "Ada" in label else "#2196F3"
    
    # mu evolution
    ax = axes[0, col_idx]
    ax.fill_between(x, binned["mu_mean"] - binned["mu_std"],
                    binned["mu_mean"] + binned["mu_std"], alpha=0.2, color=color)
    ax.plot(x, binned["mu_mean"], color=color, linewidth=2)
    ax.plot(x, binned["max_mean"], color=color, linewidth=1, linestyle="--", label="max_r")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title(f"{label}: μ and max (mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # sigma evolution
    ax = axes[1, col_idx]
    ax.fill_between(x, binned["sigma_mean"] - binned["sigma_std"],
                    binned["sigma_mean"] + binned["sigma_std"], alpha=0.2, color=color)
    ax.plot(x, binned["sigma_mean"], color=color, linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("σ (reward std)")
    ax.set_title(f"{label}: σ evolution")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_reward_evolution.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_reward_evolution.png")
plt.close()

# ── Figure 4: Compute waste analysis ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Compute Efficiency: How Much Training Is Spent on 'Already Learned' Examples?",
             fontsize=13, fontweight="bold")

for ax, (label, df) in zip(axes, [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]):
    n_bins = 20
    df_tmp = df.copy()
    df_tmp["bin"] = pd.cut(df_tmp["progress"], bins=n_bins, labels=False)
    
    x = np.linspace(0, 4, n_bins)
    
    # Case A fraction = "compute waste"
    case_a_frac = []
    # Case B fraction = "stuck, needs intervention"
    case_b_frac = []
    for b in range(n_bins):
        bin_data = df_tmp[df_tmp["bin"] == b]
        total = max(len(bin_data), 1)
        case_a_frac.append(len(bin_data[bin_data["case"] == "A: Stable Good"]) / total * 100)
        case_b_frac.append(len(bin_data[bin_data["case"] == "B: Stuck"]) / total * 100)
    
    ax.bar(x, case_a_frac, width=0.15, color="#4CAF50", alpha=0.7, label="A: Stable Good (waste)")
    ax.bar(x, case_b_frac, width=0.15, bottom=case_a_frac, color="#F44336", alpha=0.7,
           label="B: Stuck (needs help)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("% of Examples")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.set_xlim(-0.1, 4.1)
    ax.grid(True, alpha=0.2, axis="y")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_compute_waste.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_compute_waste.png")
plt.close()

# ── Save detailed stats CSV ──────────────────────────────────────────────────
summary_rows = []
for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    for phase in ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"]:
        phase_df = df[df["phase"] == phase]
        total = len(phase_df)
        for case in case_order:
            case_df = phase_df[phase_df["case"] == case]
            summary_rows.append({
                "model": label,
                "phase": phase,
                "case": case,
                "count": len(case_df),
                "pct": len(case_df) / max(total, 1) * 100,
                "mu_mean": case_df["mu"].mean() if len(case_df) > 0 else np.nan,
                "sigma_mean": case_df["sigma"].mean() if len(case_df) > 0 else np.nan,
                "max_r_mean": case_df["max_r"].mean() if len(case_df) > 0 else np.nan,
            })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(OUT_DIR, "routing_case_stats.csv"), index=False)
print(f"Saved routing_case_stats.csv")

# ── Print key insights ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("KEY INSIGHTS FOR ROUTING DESIGN")
print("=" * 70)

# Case A in late training = potential savings
for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    late = df[df["progress"] > 0.5]  # epochs 3-4
    case_a_late = len(late[late["case"] == "A: Stable Good"]) / max(len(late), 1) * 100
    case_b_late = len(late[late["case"] == "B: Stuck"]) / max(len(late), 1) * 100
    case_c_late = len(late[late["case"] == "C: Promising"]) / max(len(late), 1) * 100
    case_d_late = len(late[late["case"] == "D: Noisy/Hard"]) / max(len(late), 1) * 100
    print(f"\n{label} (epochs 3-4):")
    print(f"  A: Stable Good = {case_a_late:.1f}% ← potential compute savings")
    print(f"  B: Stuck       = {case_b_late:.1f}% ← needs teacher intervention")
    print(f"  C: Promising   = {case_c_late:.1f}% ← worth exploring")
    print(f"  D: Noisy/Hard  = {case_d_late:.1f}% ← limit budget")

# Overall sigma distribution
print(f"\nReward sigma statistics:")
for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    print(f"  {label}: median={df['sigma'].median():.4f}, p25={df['sigma'].quantile(0.25):.4f}, "
          f"p75={df['sigma'].quantile(0.75):.4f}")

# Zero-sigma examples (all G completions identical reward)
for label, df in [("AdaRePO v15.1", df_ada), ("RePO", df_repo)]:
    zero_sigma = (df["sigma"] < 0.01).sum()
    print(f"  {label}: {zero_sigma} examples ({zero_sigma/len(df)*100:.1f}%) with sigma < 0.01")

print("\nDone!")
