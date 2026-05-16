"""
Comprehensive analysis: AdaRePO vs RePO training comparison.
Generates multi-panel publication-quality figures.
"""
import re
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_LOG = "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/logs/repo_3B_6955714.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"
ADA_LOG  = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/ada_repo_3B_6956851.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"
OUT_DIR  = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})
COLOR_REPO = "#2171b5"
COLOR_ADA  = "#e6550d"
COLOR_BETA = "#31a354"
COLOR_KL   = "#756bb1"
ALPHA_FILL = 0.15

# ── Extract metrics ────────────────────────────────────────────────────────
def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = re.search(r"\{.*'loss'.*\}", line)
            if m:
                try:
                    d = eval(m.group(0))
                    rows.append(d)
                except:
                    pass
    return rows

print("Parsing logs...")
repo_data = parse_log(REPO_LOG)
ada_data  = parse_log(ADA_LOG)
print(f"  RePO:    {len(repo_data)} steps")
print(f"  AdaRePO: {len(ada_data)} steps")

steps = np.arange(1, len(repo_data) + 1)

def get(data, key, default=0.0):
    return np.array([d.get(key, default) for d in data])

# Core metrics
repo_reward  = get(repo_data, "reward")
ada_reward   = get(ada_data,  "reward")
repo_loss    = get(repo_data, "loss")
ada_loss     = get(ada_data,  "loss")
repo_sloss   = get(repo_data, "s_loss")
ada_sloss    = get(ada_data,  "s_loss")
ada_sloss_w  = get(ada_data,  "s_loss_weighted")
repo_kl      = get(repo_data, "kl")
ada_kl       = get(ada_data,  "kl")
repo_rstd    = get(repo_data, "reward_std")
ada_rstd     = get(ada_data,  "reward_std")
repo_clen    = get(repo_data, "completion_length")
ada_clen     = get(ada_data,  "completion_length")
repo_gnorm   = get(repo_data, "grad_norm")
ada_gnorm    = get(ada_data,  "grad_norm")
ada_beta     = get(ada_data,  "beta_guide_mean")
ada_vtop_vref= get(ada_data,  "v_top_minus_v_ref")
repo_epoch   = get(repo_data, "epoch")
ada_epoch    = get(ada_data,  "epoch")

# Smoothing helper
def ema(x, alpha=0.15):
    s = np.zeros_like(x)
    s[0] = x[0]
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Main comparison (2×2)  — reward, loss, s_loss, KL
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 1: Main comparison...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("AdaRePO vs RePO — Training Dynamics", fontsize=16, fontweight="bold", y=1.01)

# 1a: Reward
ax = axes[0, 0]
ax.plot(steps, ema(repo_reward), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_reward),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.fill_between(steps, ema(repo_reward - repo_rstd), ema(repo_reward + repo_rstd), color=COLOR_REPO, alpha=ALPHA_FILL)
ax.fill_between(steps, ema(ada_reward - ada_rstd),   ema(ada_reward + ada_rstd),   color=COLOR_ADA,  alpha=ALPHA_FILL)
ax.set_xlabel("Training Step")
ax.set_ylabel("Reward (smile_optimization)")
ax.set_title("(a) Reward Curve")
ax.legend()
ax.grid(alpha=0.3)
# Add epoch boundaries
for ep in [1, 2, 3]:
    idx = np.argmin(np.abs(repo_epoch - ep))
    ax.axvline(steps[idx], color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.text(steps[idx], ax.get_ylim()[1], f"Ep {ep}", fontsize=8, ha="center", va="bottom", color="gray")

# 1b: Total Loss
ax = axes[0, 1]
ax.plot(steps, ema(repo_loss), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_loss),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("Total Loss")
ax.set_title("(b) Total Loss")
ax.legend()
ax.grid(alpha=0.3)

# 1c: Guidance Loss (s_loss)
ax = axes[1, 0]
ax.plot(steps, ema(repo_sloss),  color=COLOR_REPO, lw=2, label="RePO s_loss (β=1)")
ax.plot(steps, ema(ada_sloss),   color=COLOR_ADA,  lw=2, label="AdaRePO s_loss (raw)", ls="--")
ax.plot(steps, ema(ada_sloss_w), color=COLOR_ADA,  lw=2.5, label="AdaRePO s_loss × β_guide")
ax.set_xlabel("Training Step")
ax.set_ylabel("Guidance Loss")
ax.set_title("(c) Answer-Level Guidance Loss")
ax.legend()
ax.grid(alpha=0.3)

# 1d: KL Divergence
ax = axes[1, 1]
ax.plot(steps, ema(repo_kl), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_kl),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("KL Divergence")
ax.set_title("(d) KL from Reference Policy")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_main_comparison.png"))
print(f"  Saved: {OUT_DIR}/fig1_main_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Dynamic Beta Analysis (2×2)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 2: Dynamic beta analysis...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dynamic β_guide Analysis — Active IL vs RL Balance", fontsize=16, fontweight="bold", y=1.01)

# 2a: Beta over time
ax = axes[0, 0]
ax.plot(steps, ada_beta, color=COLOR_BETA, lw=1, alpha=0.4, label="β_guide (raw)")
ax.plot(steps, ema(ada_beta, 0.1), color=COLOR_BETA, lw=2.5, label="β_guide (smoothed)")
ax.axhline(1.0, color=COLOR_REPO, ls="--", lw=1.5, alpha=0.7, label="RePO β=1 (fixed)")
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("Training Step")
ax.set_ylabel("β_guide")
ax.set_title("(a) Dynamic β_guide Over Training")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
# Annotate epochs
for ep in [1, 2, 3]:
    idx = np.argmin(np.abs(ada_epoch - ep))
    ax.axvline(steps[idx], color="gray", ls="--", alpha=0.4, lw=0.8)

# 2b: v_top - v_ref (reward gap)
ax = axes[0, 1]
ax.plot(steps, ada_vtop_vref, color="#d95f02", lw=1, alpha=0.4)
ax.plot(steps, ema(ada_vtop_vref, 0.1), color="#d95f02", lw=2.5, label="v_top − v_ref")
ax.axhline(0, color="gray", ls="-", lw=0.8, alpha=0.5)
ax.fill_between(steps, 0, ema(ada_vtop_vref, 0.1), where=ema(ada_vtop_vref, 0.1) > 0,
                color="#fc8d62", alpha=0.2, label="Model > Ref (RL dominates)")
ax.fill_between(steps, 0, ema(ada_vtop_vref, 0.1), where=ema(ada_vtop_vref, 0.1) <= 0,
                color="#66c2a5", alpha=0.2, label="Ref > Model (IL dominates)")
ax.set_xlabel("Training Step")
ax.set_ylabel("v_top − v_ref")
ax.set_title("(b) Reward Gap: Learner's Best vs Reference")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

# 2c: IL fraction — effective guidance weight
ax = axes[1, 0]
# IL fraction = s_loss_weighted / s_loss_raw (how much guidance is actually applied)
il_frac = np.where(ada_sloss > 0.01, ada_sloss_w / ada_sloss, 0)
rl_frac = 1 - il_frac
ax.fill_between(steps, 0, ema(il_frac, 0.1), color="#4292c6", alpha=0.6, label="IL (guidance) fraction")
ax.fill_between(steps, ema(il_frac, 0.1), 1, color="#ef6548", alpha=0.6, label="RL (exploration) fraction")
ax.set_xlabel("Training Step")
ax.set_ylabel("Fraction of Objective")
ax.set_title("(c) IL vs RL Balance (Effective Weight of Guidance)")
ax.set_ylim(0, 1)
ax.legend(loc="center right")
ax.grid(alpha=0.3)

# 2d: Beta histogram by epoch
ax = axes[1, 1]
epoch_boundaries = [0, 1, 2, 3, 4]
colors_ep = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
for i in range(len(epoch_boundaries) - 1):
    mask = (ada_epoch >= epoch_boundaries[i]) & (ada_epoch < epoch_boundaries[i + 1])
    if mask.sum() > 0:
        ax.hist(ada_beta[mask], bins=20, range=(0, 1), alpha=0.65,
                color=colors_ep[i], edgecolor="white", lw=0.5,
                label=f"Epoch {i+1} (μ={ada_beta[mask].mean():.3f})")
ax.set_xlabel("β_guide value")
ax.set_ylabel("Frequency")
ax.set_title("(d) β_guide Distribution by Epoch")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig2_beta_analysis.png"))
print(f"  Saved: {OUT_DIR}/fig2_beta_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Epoch-wise breakdown (bar charts)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 3: Epoch-wise breakdown...")
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Epoch-wise Metric Comparison", fontsize=16, fontweight="bold", y=1.02)

epoch_labels = ["Epoch 1", "Epoch 2", "Epoch 3", "Epoch 4"]
bar_width = 0.35
x_pos = np.arange(4)

def epoch_means(data_arr, epoch_arr):
    means = []
    for ep in range(4):
        mask = (epoch_arr >= ep) & (epoch_arr < ep + 1)
        if mask.sum() > 0:
            means.append(data_arr[mask].mean())
        else:
            means.append(0)
    return np.array(means)

metrics_to_plot = [
    ("reward", "Avg Reward", repo_reward, ada_reward, repo_epoch, ada_epoch),
    ("loss", "Avg Loss", repo_loss, ada_loss, repo_epoch, ada_epoch),
    ("kl", "Avg KL Divergence", repo_kl, ada_kl, repo_epoch, ada_epoch),
    ("s_loss", "Avg Guidance Loss", repo_sloss, ada_sloss_w, repo_epoch, ada_epoch),
]

for idx, (key, ylabel, repo_arr, ada_arr, repo_ep, ada_ep) in enumerate(metrics_to_plot):
    ax = axes[idx]
    repo_means = epoch_means(repo_arr, repo_ep)
    ada_means  = epoch_means(ada_arr, ada_ep)
    bars1 = ax.bar(x_pos - bar_width/2, repo_means, bar_width, color=COLOR_REPO, label="RePO", edgecolor="white")
    bars2 = ax.bar(x_pos + bar_width/2, ada_means,  bar_width, color=COLOR_ADA,  label="AdaRePO", edgecolor="white")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(epoch_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(f"({chr(97+idx)}) {ylabel}")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=7)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig3_epoch_breakdown.png"))
print(f"  Saved: {OUT_DIR}/fig3_epoch_breakdown.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 4: Stability and efficiency analysis (2×2)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 4: Stability and efficiency...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Training Stability & Efficiency Analysis", fontsize=16, fontweight="bold", y=1.01)

# 4a: Gradient norm
ax = axes[0, 0]
ax.plot(steps, ema(repo_gnorm, 0.1), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_gnorm, 0.1),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("Gradient Norm")
ax.set_title("(a) Gradient Norm (training stability)")
ax.legend()
ax.grid(alpha=0.3)
ax.set_yscale("log")

# 4b: Completion length
ax = axes[0, 1]
ax.plot(steps, ema(repo_clen), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_clen),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("Avg Completion Length (tokens)")
ax.set_title("(b) Completion Length Over Training")
ax.legend()
ax.grid(alpha=0.3)

# 4c: Reward efficiency — cumulative reward / step
ax = axes[1, 0]
repo_cumrew = np.cumsum(repo_reward) / steps
ada_cumrew  = np.cumsum(ada_reward) / steps
ax.plot(steps, repo_cumrew, color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ada_cumrew,  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("Cumulative Avg Reward")
ax.set_title("(c) Reward Efficiency (cumulative mean)")
ax.legend()
ax.grid(alpha=0.3)

# 4d: Reward std (exploration diversity)
ax = axes[1, 1]
ax.plot(steps, ema(repo_rstd), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_rstd),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_xlabel("Training Step")
ax.set_ylabel("Reward Std (within batch)")
ax.set_title("(d) Exploration Diversity (reward variance)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig4_stability_efficiency.png"))
print(f"  Saved: {OUT_DIR}/fig4_stability_efficiency.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 5: Phase portrait — beta vs reward gap (scatter)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 5: Phase portrait...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Phase Analysis of Dynamic β_guide", fontsize=16, fontweight="bold", y=1.02)

# 5a: Scatter — beta vs v_top - v_ref, colored by step
ax = axes[0]
sc = ax.scatter(ada_vtop_vref, ada_beta, c=steps, cmap="viridis", s=25, alpha=0.7, edgecolors="none")
cb = plt.colorbar(sc, ax=ax, label="Training Step")
# Overlay theoretical curve: beta = beta_max * sigmoid(-alpha * gap)
gap_range = np.linspace(ada_vtop_vref.min() - 0.1, ada_vtop_vref.max() + 0.1, 200)
beta_theory = 1.0 * 1 / (1 + np.exp(5.0 * gap_range))
ax.plot(gap_range, beta_theory, "k--", lw=2, alpha=0.7, label="Theoretical: σ(−5·gap)")
ax.set_xlabel("v_top − v_ref (reward gap)")
ax.set_ylabel("β_guide")
ax.set_title("(a) β_guide vs Reward Gap (colored by step)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)

# 5b: Beta vs reward (scatter), colored by epoch
ax = axes[1]
epoch_colors = plt.cm.Set1(np.linspace(0, 0.5, 4))
for ep_i in range(4):
    mask = (ada_epoch >= ep_i) & (ada_epoch < ep_i + 1)
    if mask.sum() > 0:
        ax.scatter(ada_reward[mask], ada_beta[mask], c=[epoch_colors[ep_i]], s=30, alpha=0.6,
                   label=f"Epoch {ep_i+1}", edgecolors="none")
ax.set_xlabel("Reward")
ax.set_ylabel("β_guide")
ax.set_title("(b) β_guide vs Reward (by epoch)")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig5_phase_portrait.png"))
print(f"  Saved: {OUT_DIR}/fig5_phase_portrait.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 6: Oracle reliance — when does the model surpass the reference?
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 6: Oracle reliance timeline...")
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle("Oracle (Reference) Reliance Over Training", fontsize=16, fontweight="bold", y=1.01)

# 6a: v_top - v_ref with zones
ax = axes[0]
gap_smooth = ema(ada_vtop_vref, 0.08)
ax.plot(steps, gap_smooth, color="#d95f02", lw=2.5)
ax.fill_between(steps, gap_smooth, 0, where=gap_smooth > 0, color="#fc8d62", alpha=0.3, label="Model outperforms ref → less guidance needed")
ax.fill_between(steps, gap_smooth, 0, where=gap_smooth <= 0, color="#66c2a5", alpha=0.3, label="Ref outperforms model → more guidance needed")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("v_top − v_ref")
ax.set_title("(a) Reward Gap: When Does the Learner Surpass the Oracle?")
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)
for ep in [1, 2, 3]:
    idx = np.argmin(np.abs(ada_epoch - ep))
    ax.axvline(steps[idx], color="gray", ls="--", alpha=0.4, lw=0.8)
    ax.text(steps[idx], ax.get_ylim()[1] * 0.95, f"Ep {ep}", fontsize=8, ha="center", color="gray")

# 6b: Rolling "oracle selection frequency" — fraction of steps where beta > 0.5
ax = axes[1]
window = 15
oracle_freq = np.convolve(ada_beta > 0.5, np.ones(window)/window, mode="same")
rl_freq = 1 - oracle_freq
ax.fill_between(steps, 0, oracle_freq, color="#4292c6", alpha=0.7, label="IL-dominant (β > 0.5)")
ax.fill_between(steps, oracle_freq, 1, color="#ef6548", alpha=0.7, label="RL-dominant (β ≤ 0.5)")
ax.set_xlabel("Training Step")
ax.set_ylabel("Rolling Frequency (window=15)")
ax.set_title("(b) Oracle Selection Frequency — IL vs RL Dominance")
ax.set_ylim(0, 1)
ax.legend(loc="center right")
ax.grid(alpha=0.3)
for ep in [1, 2, 3]:
    idx = np.argmin(np.abs(ada_epoch - ep))
    ax.axvline(steps[idx], color="gray", ls="--", alpha=0.4, lw=0.8)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_oracle_reliance.png"))
print(f"  Saved: {OUT_DIR}/fig6_oracle_reliance.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 7: Loss decomposition — RL term vs guidance term
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 7: Loss decomposition...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Loss Decomposition: RL vs Guidance Components", fontsize=16, fontweight="bold", y=1.02)

# For RePO:  total_loss = rl_loss + s_loss  (s_loss has implicit beta=1)
# For AdaRePO: total_loss = rl_loss + beta*s_loss = rl_loss + s_loss_weighted
repo_rl_loss = repo_loss - repo_sloss
ada_rl_loss  = ada_loss - ada_sloss_w

ax = axes[0]
ax.stackplot(steps, ema(np.maximum(repo_rl_loss, 0)), ema(repo_sloss),
             colors=[COLOR_REPO, "#9ecae1"], alpha=0.8,
             labels=["RL loss (reward + KL)", "Guidance loss (s_loss)"])
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("(a) RePO — Loss Decomposition (β=1 fixed)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

ax = axes[1]
ax.stackplot(steps, ema(np.maximum(ada_rl_loss, 0)), ema(ada_sloss_w),
             colors=[COLOR_ADA, "#fdae6b"], alpha=0.8,
             labels=["RL loss (reward + KL)", "Guidance loss (β·s_loss)"])
ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("(b) AdaRePO — Loss Decomposition (β dynamic)")
ax.legend(loc="upper right")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_loss_decomposition.png"))
print(f"  Saved: {OUT_DIR}/fig7_loss_decomposition.png")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 8: Summary dashboard (single page)
# ═══════════════════════════════════════════════════════════════════════════
print("Generating Figure 8: Summary dashboard...")
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("AdaRePO vs RePO — Complete Training Summary\nQwen2.5-3B-Instruct · 184 steps · 4 epochs · Polaris 4×A100",
             fontsize=15, fontweight="bold", y=0.99)

# Row 1: Reward, Loss, KL
ax = fig.add_subplot(gs[0, 0])
ax.plot(steps, ema(repo_reward), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_reward),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_title("Reward ↑"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 1])
ax.plot(steps, ema(repo_loss), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_loss),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_title("Total Loss ↓"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[0, 2])
ax.plot(steps, ema(repo_kl), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_kl),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_title("KL Divergence ↓"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 2: Beta, v_top-v_ref, IL/RL balance
ax = fig.add_subplot(gs[1, 0])
ax.plot(steps, ema(ada_beta, 0.1), color=COLOR_BETA, lw=2.5)
ax.axhline(1.0, color=COLOR_REPO, ls="--", lw=1, alpha=0.5, label="RePO β=1")
ax.set_title("β_guide (dynamic)"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 1])
ax.plot(steps, ema(ada_vtop_vref, 0.08), color="#d95f02", lw=2.5)
ax.axhline(0, color="gray", lw=1); ax.set_title("v_top − v_ref"); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[1, 2])
ax.fill_between(steps, 0, ema(il_frac, 0.1), color="#4292c6", alpha=0.6, label="IL")
ax.fill_between(steps, ema(il_frac, 0.1), 1, color="#ef6548", alpha=0.6, label="RL")
ax.set_title("IL vs RL Balance"); ax.set_ylim(0, 1); ax.legend(fontsize=8); ax.grid(alpha=0.3)

# Row 3: Guidance loss, Gradient norm, Completion length
ax = fig.add_subplot(gs[2, 0])
ax.plot(steps, ema(repo_sloss), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_sloss_w), color=COLOR_ADA, lw=2, label="AdaRePO (β·s)")
ax.set_title("Effective Guidance ↓"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 1])
ax.plot(steps, ema(repo_gnorm, 0.1), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_gnorm, 0.1),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_title("Grad Norm ↓"); ax.set_yscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

ax = fig.add_subplot(gs[2, 2])
ax.plot(steps, ema(repo_clen), color=COLOR_REPO, lw=2, label="RePO")
ax.plot(steps, ema(ada_clen),  color=COLOR_ADA,  lw=2, label="AdaRePO")
ax.set_title("Completion Length"); ax.legend(fontsize=8); ax.grid(alpha=0.3)

fig.savefig(os.path.join(OUT_DIR, "fig8_summary_dashboard.png"))
print(f"  Saved: {OUT_DIR}/fig8_summary_dashboard.png")

# ═══════════════════════════════════════════════════════════════════════════
# Print final summary stats
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("FINAL SUMMARY STATISTICS")
print("="*70)

last_n = 20
for name, data, reward, loss, sloss, kl in [
    ("RePO", repo_data, repo_reward, repo_loss, repo_sloss, repo_kl),
    ("AdaRePO", ada_data, ada_reward, ada_loss, ada_sloss_w, ada_kl),
]:
    print(f"\n{name} (last {last_n} steps):")
    print(f"  reward:  {reward[-last_n:].mean():.4f} ± {reward[-last_n:].std():.4f}")
    print(f"  loss:    {loss[-last_n:].mean():.4f} ± {loss[-last_n:].std():.4f}")
    print(f"  kl:      {kl[-last_n:].mean():.4f} ± {kl[-last_n:].std():.4f}")

print(f"\nAdaRePO-specific (last {last_n} steps):")
print(f"  beta_guide:      {ada_beta[-last_n:].mean():.4f} ± {ada_beta[-last_n:].std():.4f}")
print(f"  v_top - v_ref:   {ada_vtop_vref[-last_n:].mean():.4f} ± {ada_vtop_vref[-last_n:].std():.4f}")
print(f"  s_loss_raw:      {ada_sloss[-last_n:].mean():.4f}")
print(f"  s_loss_weighted: {ada_sloss_w[-last_n:].mean():.4f}")
print(f"  IL fraction:     {il_frac[-last_n:].mean():.4f}")

# Beta trend by epoch
print(f"\nBeta by epoch:")
for ep_i in range(4):
    mask = (ada_epoch >= ep_i) & (ada_epoch < ep_i + 1)
    if mask.sum() > 0:
        print(f"  Epoch {ep_i+1}: β={ada_beta[mask].mean():.4f} ± {ada_beta[mask].std():.4f}  "
              f"gap={ada_vtop_vref[mask].mean():.4f}  reward={ada_reward[mask].mean():.4f}")

print(f"\nIL-dominant steps (β > 0.5): {(ada_beta > 0.5).sum()}/{len(ada_beta)} ({100*(ada_beta > 0.5).mean():.1f}%)")
print(f"RL-dominant steps (β ≤ 0.5): {(ada_beta <= 0.5).sum()}/{len(ada_beta)} ({100*(ada_beta <= 0.5).mean():.1f}%)")

print(f"\n{'='*70}")
print("All figures saved to:", OUT_DIR)
print("="*70)
