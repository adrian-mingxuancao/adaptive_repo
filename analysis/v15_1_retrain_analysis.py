"""
Generate analysis figures for AdaRePO v15.1 Retrain vs RePO.
Includes: training reward/advantage curves, RL diagnostics, eval comparison bar chart.
"""
import ast
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "v15_1_retrain_assets")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Parse training logs ──────────────────────────────────────────────────────
def parse_log(paths):
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                s = line.strip()
                if s.startswith("{'loss'"):
                    try:
                        records.append(ast.literal_eval(s))
                    except Exception:
                        pass
    return pd.DataFrame(records)

repo_logs = [
    "/net/scratch/caom/repo_project/logs/repo_propopt_791601.out",
    "/net/scratch/caom/repo_project/logs/repo_propopt_resume_795397.out",
]
ada_logs = [
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_812647.out",
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_813476.out",
    "/net/scratch/caom/repo_project/logs/v15_1_retrain_814309.out",
]

df_repo = parse_log(repo_logs)
df_ada = parse_log(ada_logs)

# Assign step numbers based on sequential order, then deduplicate
# Each log entry is one step; resume logs re-count from resume point
# Use epoch as a proxy: 748 steps over 4 epochs → step = round(epoch/4 * 748)
for df in [df_repo, df_ada]:
    df["step"] = (df["epoch"] / 4.0 * 748).round().astype(int)

# Deduplicate by step, keep last occurrence (from resume logs)
df_repo = df_repo.drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
df_ada = df_ada.drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)

# EMA smoothing
def ema(series, alpha=0.15):
    return series.ewm(alpha=alpha).mean()

# ── Figure 1: Reward & Advantage ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AdaRePO v15.1 Retrain vs RePO — Training Dynamics", fontsize=14, fontweight="bold")

# Reward
ax = axes[0, 0]
ax.plot(df_repo["step"], ema(df_repo["reward"]), label="RePO", color="#2196F3", linewidth=1.5)
ax.plot(df_ada["step"], ema(df_ada["reward"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.set_title("Training Reward (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

# Loss
ax = axes[0, 1]
ax.plot(df_repo["step"], ema(df_repo["loss"]), label="RePO", color="#2196F3", linewidth=1.5)
ax.plot(df_ada["step"], ema(df_ada["loss"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Loss")
ax.set_title("Training Loss (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

# Advantage mean
ax = axes[1, 0]
if "advantage/mean" in df_repo.columns:
    ax.plot(df_repo["step"], ema(df_repo["advantage/mean"]), label="RePO", color="#2196F3", linewidth=1.5)
if "advantage/mean" in df_ada.columns:
    ax.plot(df_ada["step"], ema(df_ada["advantage/mean"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Step")
ax.set_ylabel("Advantage Mean")
ax.set_title("Advantage Mean (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

# KL divergence
ax = axes[1, 1]
if "kl" in df_repo.columns:
    ax.plot(df_repo["step"], ema(df_repo["kl"]), label="RePO", color="#2196F3", linewidth=1.5)
if "kl" in df_ada.columns:
    ax.plot(df_ada["step"], ema(df_ada["kl"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("KL Divergence")
ax.set_title("KL Divergence (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_reward_advantage.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_reward_advantage.png")
plt.close()

# ── Figure 2: RL Diagnostics (beta, s_loss, completion_length) ───────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AdaRePO v15.1 Retrain — RL Diagnostics", fontsize=14, fontweight="bold")

# Beta guide (AdaRePO only)
ax = axes[0, 0]
if "beta_guide_mean" in df_ada.columns:
    ax.plot(df_ada["step"], ema(df_ada["beta_guide_mean"]), color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Beta")
ax.set_title("Dynamic Beta (AdaRePO)")
ax.grid(True, alpha=0.3)

# s_loss (guidance loss)
ax = axes[0, 1]
if "s_loss" in df_repo.columns:
    ax.plot(df_repo["step"], ema(df_repo["s_loss"]), label="RePO", color="#2196F3", linewidth=1.5)
if "s_loss" in df_ada.columns:
    ax.plot(df_ada["step"], ema(df_ada["s_loss"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Guidance Loss")
ax.set_title("Guidance Loss (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

# Completion length
ax = axes[1, 0]
if "completion_length" in df_repo.columns:
    ax.plot(df_repo["step"], ema(df_repo["completion_length"]), label="RePO", color="#2196F3", linewidth=1.5)
if "completion_length" in df_ada.columns:
    ax.plot(df_ada["step"], ema(df_ada["completion_length"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Tokens")
ax.set_title("Completion Length (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

# Grad norm
ax = axes[1, 1]
ax.plot(df_repo["step"], ema(df_repo["grad_norm"]), label="RePO", color="#2196F3", linewidth=1.5)
ax.plot(df_ada["step"], ema(df_ada["grad_norm"]), label="AdaRePO v15.1", color="#FF5722", linewidth=1.5)
ax.set_xlabel("Step")
ax.set_ylabel("Grad Norm")
ax.set_title("Gradient Norm (EMA)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_rl_diagnostics.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_rl_diagnostics.png")
plt.close()

# ── Figure 3: Eval Comparison Bar Chart ──────────────────────────────────────
eval_data = {
    "RePO ckpt-748": {
        "LogP": {"sr": 0.4434, "sim": 0.7112, "val": 0.7788},
        "MR":   {"sr": 0.5052, "sim": 0.7090, "val": 0.7692},
        "QED":  {"sr": 0.3480, "sim": 0.7217, "val": 0.7810},
    },
    "v15.1 retrain ckpt-748": {
        "LogP": {"sr": 0.4890, "sim": 0.7350, "val": 0.7936},
        "MR":   {"sr": 0.5042, "sim": 0.7128, "val": 0.7728},
        "QED":  {"sr": 0.3282, "sim": 0.7441, "val": 0.8144},
    },
}

tasks = ["LogP", "MR", "QED"]
metrics = ["SR", "Sim", "Val", "SR×Sim"]
models = list(eval_data.keys())
colors = {"RePO ckpt-748": "#2196F3", "v15.1 retrain ckpt-748": "#FF5722"}

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("AdaRePO v15.1 Retrain vs RePO — Evaluation (MolOpt Benchmark)", fontsize=14, fontweight="bold")

x = np.arange(len(tasks))
width = 0.35

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    for i, model in enumerate(models):
        vals = []
        for task in tasks:
            d = eval_data[model][task]
            if metric == "SR":
                vals.append(d["sr"])
            elif metric == "Sim":
                vals.append(d["sim"])
            elif metric == "Val":
                vals.append(d["val"])
            else:
                vals.append(d["sr"] * d["sim"])
        bars = ax.bar(x + i * width - width / 2, vals, width, label=model, color=colors[model], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_title(metric)
    ax.set_ylim(0, max(1.0, ax.get_ylim()[1] * 1.1))
    if idx == 0:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

# Add average SR×Sim annotation
for i, model in enumerate(models):
    avg = np.mean([eval_data[model][t]["sr"] * eval_data[model][t]["sim"] for t in tasks])
    axes[3].text(0.5, 0.95 - i * 0.06, f"{model}: avg={avg:.4f}",
                 transform=axes[3].transAxes, fontsize=9, color=colors[model], fontweight="bold")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig_eval_comparison.png"), dpi=150, bbox_inches="tight")
print(f"Saved fig_eval_comparison.png")
plt.close()

# ── Save summary CSV ─────────────────────────────────────────────────────────
rows = []
for model in models:
    for task in tasks:
        d = eval_data[model][task]
        rows.append({
            "model": model,
            "task": task,
            "success_rate": d["sr"],
            "similarity": d["sim"],
            "validity": d["val"],
            "sr_x_sim": d["sr"] * d["sim"],
        })
    # Average row
    avg_sr = np.mean([eval_data[model][t]["sr"] for t in tasks])
    avg_sim = np.mean([eval_data[model][t]["sim"] for t in tasks])
    avg_val = np.mean([eval_data[model][t]["val"] for t in tasks])
    avg_srsim = np.mean([eval_data[model][t]["sr"] * eval_data[model][t]["sim"] for t in tasks])
    rows.append({
        "model": model,
        "task": "Average",
        "success_rate": avg_sr,
        "similarity": avg_sim,
        "validity": avg_val,
        "sr_x_sim": avg_srsim,
    })

df_summary = pd.DataFrame(rows)
df_summary.to_csv(os.path.join(OUT_DIR, "eval_summary.csv"), index=False)
print(f"Saved eval_summary.csv")

# ── Training summary CSV ─────────────────────────────────────────────────────
train_rows = []
for step in [100, 200, 300, 400, 500, 600, 700, 748]:
    w = 10
    for label, df in [("RePO", df_repo), ("v15.1-retrain", df_ada)]:
        subset = df[(df["step"] >= step - w) & (df["step"] <= step + w)]
        if len(subset) > 0:
            train_rows.append({
                "model": label,
                "step": step,
                "reward_mean": subset["reward"].mean(),
                "loss_mean": subset["loss"].mean(),
                "kl_mean": subset["kl"].mean() if "kl" in subset.columns else None,
                "completion_length": subset["completion_length"].mean() if "completion_length" in subset.columns else None,
            })

df_train = pd.DataFrame(train_rows)
df_train.to_csv(os.path.join(OUT_DIR, "training_summary.csv"), index=False)
print(f"Saved training_summary.csv")
print("\nDone!")
