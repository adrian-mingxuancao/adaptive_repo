"""
v15 + v15.1 Full Comparison Plots — presentation quality for collaborators.
Generates:
  1. Training curves (reward, reward_std, KL, s_loss, completion_length)
  2. Beta & weighted guidance loss
  3. Eval bar chart (SR, Sim, SR×Sim, Validity)
"""
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Run definitions ──────────────────────────────────────────────────────────
LOG_DIR = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs"

RUNS = [
    ("RePO baseline",       f"{LOG_DIR}/v15_v15_repo_baseline_7089030.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
    ("AdaRePO static (β=1)", f"{LOG_DIR}/v15_v15_static_7090220.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
    ("AdaRePO sigmoid (old)", f"{LOG_DIR}/v15_v15_beta_only_7090221.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
    ("AdaRePO sigmoid+PW (old)", f"{LOG_DIR}/v15_v15_active_7090222.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
    ("AdaRePO boosted",     f"{LOG_DIR}/v15_v15_1_boosted_7091218.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
    ("AdaRePO boosted+PW",  f"{LOG_DIR}/v15_v15_1_boosted_pw_7091219.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log"),
]

COLORS = [
    "#1f77b4",  # blue  — RePO
    "#2ca02c",  # green — static
    "#ff7f0e",  # orange — old sigmoid
    "#d62728",  # red   — old sigmoid+PW
    "#9467bd",  # purple — boosted
    "#8c564b",  # brown  — boosted+PW
]

EVAL_RESULTS = {
    "RePO baseline":            {"SR": 0.282, "Sim": 0.763, "Validity": 0.607},
    "AdaRePO static (β=1)":     {"SR": 0.320, "Sim": 0.763, "Validity": 0.667},
    "AdaRePO sigmoid (old)":    {"SR": 0.241, "Sim": 0.798, "Validity": 0.655},
    "AdaRePO sigmoid+PW (old)": {"SR": 0.212, "Sim": 0.833, "Validity": 0.653},
    "AdaRePO boosted":          {"SR": 0.312, "Sim": 0.761, "Validity": 0.655},
    "AdaRePO boosted+PW":       {"SR": 0.243, "Sim": 0.811, "Validity": 0.669},
}

LINESTYLES = ["-", "-", "--", "--", "-", "--"]
MARKERS = ["", "", "", "", "", ""]

# ── Log parser ───────────────────────────────────────────────────────────────
PATTERN = re.compile(r"\{[^{}]*'loss'[^{}]*\}")

def parse_log(log_file):
    keys = ["loss", "reward", "reward_std", "kl", "s_loss",
            "s_loss_weighted", "beta_guide_mean", "completion_length"]
    data = {k: [] for k in keys}
    data["step"] = []
    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            m = PATTERN.search(line)
            if m:
                try:
                    d = eval(m.group())
                    data["step"].append(len(data["step"]) + 1)
                    for k in keys:
                        data[k].append(d.get(k, None))
                except Exception:
                    pass
    return data


def smooth(vals, w=7):
    """Moving average, returns (smoothed_vals, valid_slice_start)."""
    if len(vals) < w:
        return vals, 0
    kernel = np.ones(w) / w
    s = np.convolve(vals, kernel, mode="valid")
    return s, w // 2


def plot_metric(ax, all_data, metric, title, ylabel=None, clip_top=None):
    for i, (name, data) in enumerate(all_data):
        vals = data.get(metric, [])
        steps = data["step"]
        if not vals or not any(v is not None for v in vals):
            continue
        cs = [s for s, v in zip(steps, vals) if v is not None]
        cv = np.array([v for v in vals if v is not None], dtype=float)
        if clip_top is not None:
            cv = np.clip(cv, None, clip_top)
        sv, offset = smooth(cv)
        ax.plot(cs[offset:offset+len(sv)], sv, label=name, color=COLORS[i],
                linewidth=2, linestyle=LINESTYLES[i])
        ax.plot(cs, cv, color=COLORS[i], alpha=0.12, linewidth=0.5)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Training Step", fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=8)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    out_dir = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/figures"
    os.makedirs(out_dir, exist_ok=True)

    # Parse all logs
    all_data = []
    for name, path in RUNS:
        print(f"Parsing {name} ...")
        d = parse_log(path)
        print(f"  {len(d['step'])} steps")
        all_data.append((name, d))

    # ================================================================
    # Figure 1: Training Curves (2×3)
    # ================================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("v15 Experiment — Training Curves\n"
                 "(LogP, Qwen2.5-3B, maxlen=1024, 120 steps, seed=42)",
                 fontsize=13, fontweight="bold", y=0.98)

    plot_metric(axes[0, 0], all_data, "reward", "Reward (mean)")
    plot_metric(axes[0, 1], all_data, "reward_std", "Reward Std (within group)")
    plot_metric(axes[0, 2], all_data, "kl", "KL Divergence", clip_top=5.0)
    plot_metric(axes[1, 0], all_data, "s_loss", "Guidance Loss (s_loss)")
    plot_metric(axes[1, 1], all_data, "loss", "Total Loss", clip_top=3.0)
    plot_metric(axes[1, 2], all_data, "completion_length", "Completion Length")

    # Single legend at bottom
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "v15_full_training_curves.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: v15_full_training_curves.png")

    # ================================================================
    # Figure 2: Beta & Weighted Guidance (1×2)
    # ================================================================
    fig2, (ax_b, ax_sw) = plt.subplots(1, 2, figsize=(14, 5))
    fig2.suptitle("v15 — Dynamic Beta & Weighted Guidance Loss",
                  fontsize=13, fontweight="bold")

    plot_metric(ax_b, all_data, "beta_guide_mean", "β_guide (mean)")
    ax_b.axhline(1.0, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax_b.annotate("RePO equiv (β=1)", xy=(5, 1.01), fontsize=8, color="gray")

    plot_metric(ax_sw, all_data, "s_loss_weighted", "Weighted Guidance Loss (β × s_loss)")

    handles2, labels2 = ax_b.get_legend_handles_labels()
    fig2.legend(handles2, labels2, loc="lower center", ncol=3, fontsize=9,
                frameon=True, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    fig2.savefig(os.path.join(out_dir, "v15_full_beta_guidance.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: v15_full_beta_guidance.png")

    # ================================================================
    # Figure 3: Evaluation Bar Chart
    # ================================================================
    names = list(EVAL_RESULTS.keys())
    sr_vals = [EVAL_RESULTS[n]["SR"] for n in names]
    sim_vals = [EVAL_RESULTS[n]["Sim"] for n in names]
    srsim_vals = [EVAL_RESULTS[n]["SR"] * EVAL_RESULTS[n]["Sim"] for n in names]
    val_vals = [EVAL_RESULTS[n]["Validity"] for n in names]

    short_names = [
        "RePO\nbaseline", "AdaRePO\nstatic\n(β=1)", "AdaRePO\nsigmoid\n(old)",
        "AdaRePO\nsigmoid+PW\n(old)", "AdaRePO\nboosted", "AdaRePO\nboosted+PW"
    ]

    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5.5))
    fig3.suptitle("v15 Evaluation Results — LogP (N=5000, checkpoint-120)",
                  fontsize=13, fontweight="bold")

    for ax, vals, title in [
        (axes3[0], sr_vals, "Success Rate ↑"),
        (axes3[1], sim_vals, "Similarity"),
        (axes3[2], srsim_vals, "SR × Sim ↑"),
        (axes3[3], val_vals, "Validity"),
    ]:
        bars = ax.bar(range(len(names)), vals, color=COLORS, edgecolor="white", linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(short_names, fontsize=7, ha="center")
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
        # Highlight best
        best_idx = np.argmax(vals) if "↑" in title else None
        if best_idx is not None:
            bars[best_idx].set_edgecolor("black")
            bars[best_idx].set_linewidth(2)

    plt.tight_layout()
    fig3.savefig(os.path.join(out_dir, "v15_full_eval_bars.png"), dpi=200, bbox_inches="tight")
    print(f"Saved: v15_full_eval_bars.png")

    # ================================================================
    # Print summary table
    # ================================================================
    print("\n" + "=" * 90)
    print("EVALUATION SUMMARY")
    print("=" * 90)
    print(f"{'Run':<30} {'SR':>6} {'Sim':>6} {'SR×Sim':>8} {'Validity':>8}")
    print("-" * 90)
    for n in names:
        e = EVAL_RESULTS[n]
        srsim = e["SR"] * e["Sim"]
        print(f"{n:<30} {e['SR']:>6.3f} {e['Sim']:>6.3f} {srsim:>8.3f} {e['Validity']:>8.3f}")


if __name__ == "__main__":
    main()
