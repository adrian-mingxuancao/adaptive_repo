"""
Cross-subtask analysis: compare RePO vs AdaRePO across LogP, MR, QED.
Generates per-subtask and cross-subtask comparison figures.

Usage:
  python cross_subtask_analysis.py [--subtasks LogP MR QED]

Expects log files in standard locations. Skips missing experiments gracefully.
"""
import re
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = "/lus/eagle/projects/IMPROVE_Aim1/caom"
REPO_LOG_DIR = f"{BASE_DIR}/RePO/logs"
ADA_LOG_DIR = f"{BASE_DIR}/agent_drug_discovery/adaptive_repo/logs"
OUT_DIR = f"{BASE_DIR}/agent_drug_discovery/adaptive_repo/analysis/figures_cross_subtask"
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
COLOR_ADA = "#d94801"
SUBTASK_COLORS = {"LogP": "#2ca02c", "MR": "#9467bd", "QED": "#e377c2"}

# ── Log Parsing ────────────────────────────────────────────────────────────
def parse_log(path):
    """Parse training log and extract metrics per step."""
    if not os.path.exists(path):
        return None

    data = {
        "step": [], "loss": [], "reward": [], "kl": [],
        "s_loss": [], "grad_norm": [], "completion_length": [],
        "reward_std": [], "epoch": [],
    }
    ada_data = {
        "beta_guide": [], "s_loss_weighted": [], "v_top_minus_v_ref": [],
    }
    is_ada = False
    step_counter = 0

    with open(path) as f:
        for line in f:
            m = re.search(r"'loss':\s*([\d.]+)", line)
            if not m:
                continue
            step_counter += 1
            step = step_counter
            loss = float(m.group(1))

            def _get(key, default=0.0):
                mm = re.search(rf"'{key}':\s*([\d.eE\-+]+)", line)
                return float(mm.group(1)) if mm else default

            data["step"].append(step)
            data["loss"].append(loss)
            data["reward"].append(_get("reward/smile_optimization", _get("reward", 0.0)))
            data["kl"].append(_get("kl", 0.0))
            data["s_loss"].append(_get("s_loss", 0.0))
            data["grad_norm"].append(_get("grad_norm", 0.0))
            data["completion_length"].append(_get("completion_length", 0.0))
            data["reward_std"].append(_get("reward_std", 0.0))
            data["epoch"].append(_get("epoch", 0.0))

            beta = _get("beta_guide_mean", -1.0)
            if beta >= 0:
                is_ada = True
                ada_data["beta_guide"].append(beta)
                ada_data["s_loss_weighted"].append(_get("s_loss_weighted", 0.0))
                ada_data["v_top_minus_v_ref"].append(_get("v_top_minus_v_ref", 0.0))
            else:
                ada_data["beta_guide"].append(np.nan)
                ada_data["s_loss_weighted"].append(np.nan)
                ada_data["v_top_minus_v_ref"].append(np.nan)

    if len(data["step"]) == 0:
        return None

    result = {k: np.array(v) for k, v in data.items()}
    result["is_ada"] = is_ada
    if is_ada:
        for k, v in ada_data.items():
            result[k] = np.array(v)
    return result


def ema(arr, alpha=0.15):
    out = np.empty_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


def find_log(log_dir, method, subtask):
    """Find the latest log file for a given method/subtask combo.
    Searches multiple directories since the unified launcher writes all logs to ADA_LOG_DIR."""
    prefix = f"{'ada_repo' if method == 'AdaRePO' else 'repo'}_3B_{subtask}_"

    # Search both log directories (launcher writes all to ADA_LOG_DIR)
    search_dirs = list(set([log_dir, ADA_LOG_DIR, REPO_LOG_DIR]))

    candidates = []
    for d in search_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.startswith(prefix) and f.endswith(".log"):
                    candidates.append(os.path.join(d, f))

    if not candidates:
        return None
    # Return most recently modified
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


# ── Analysis Functions ─────────────────────────────────────────────────────

def per_subtask_comparison(subtask, repo_data, ada_data):
    """Generate the standard 4-panel comparison for one subtask."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"MolOpt/{subtask} — RePO vs AdaRePO", fontsize=16, fontweight="bold")

    steps_r = repo_data["step"]
    steps_a = ada_data["step"]

    # (a) Reward
    ax = axes[0, 0]
    ax.plot(steps_r, ema(repo_data["reward"]), color=COLOR_REPO, label="RePO")
    ax.plot(steps_a, ema(ada_data["reward"]), color=COLOR_ADA, label="AdaRePO")
    ax.set_title(f"(a) Reward ({subtask})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Total Loss
    ax = axes[0, 1]
    ax.plot(steps_r, ema(repo_data["loss"]), color=COLOR_REPO, label="RePO")
    ax.plot(steps_a, ema(ada_data["loss"]), color=COLOR_ADA, label="AdaRePO")
    ax.set_title("(b) Total Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Guidance Loss
    ax = axes[1, 0]
    ax.plot(steps_r, ema(repo_data["s_loss"]), color=COLOR_REPO, label="RePO s_loss (β=1)")
    ax.plot(steps_a, ema(ada_data["s_loss"]), color=COLOR_ADA, ls="--", label="AdaRePO s_loss (raw)")
    if ada_data["is_ada"]:
        ax.plot(steps_a, ema(ada_data["s_loss_weighted"]), color=COLOR_ADA, label="AdaRePO β·s_loss")
    ax.set_title("(c) Guidance Loss")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Guidance Loss")
    ax.legend()
    ax.grid(alpha=0.3)

    # (d) KL Divergence
    ax = axes[1, 1]
    ax.plot(steps_r, ema(repo_data["kl"]), color=COLOR_REPO, label="RePO")
    ax.plot(steps_a, ema(ada_data["kl"]), color=COLOR_ADA, label="AdaRePO")
    ax.set_title("(d) KL Divergence")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("KL")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"per_subtask_{subtask}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def beta_dynamics_per_subtask(subtask, ada_data):
    """Beta dynamics plot for one subtask."""
    if not ada_data["is_ada"]:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"MolOpt/{subtask} — Dynamic β Analysis", fontsize=14, fontweight="bold")

    steps = ada_data["step"]
    beta = ada_data["beta_guide"]
    gap = ada_data["v_top_minus_v_ref"]

    # (a) Beta over time
    ax = axes[0]
    ax.plot(steps, beta, alpha=0.3, color=COLOR_ADA)
    ax.plot(steps, ema(beta), color=COLOR_ADA, lw=2, label="β_guide (smoothed)")
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5, label="RePO β=1")
    ax.set_title("(a) β_guide Over Training")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("β_guide")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Reward gap
    ax = axes[1]
    ax.plot(steps, ema(gap), color=COLOR_ADA, lw=2)
    ax.fill_between(steps, 0, ema(gap), alpha=0.2, color=COLOR_ADA)
    ax.axhline(0, ls="-", color="gray", alpha=0.5)
    ax.set_title("(b) Reward Gap (v_top − v_ref)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gap")
    ax.grid(alpha=0.3)

    # (c) Beta distribution by epoch
    ax = axes[2]
    epochs = ada_data["epoch"]
    unique_epochs = sorted(set(int(e) for e in epochs if e > 0))
    if not unique_epochs:
        unique_epochs = [1]
    for ep in unique_epochs:
        mask = (epochs >= ep) & (epochs < ep + 1)
        if mask.sum() > 0:
            ax.hist(beta[mask], bins=15, alpha=0.5, label=f"Epoch {ep} (μ={beta[mask].mean():.3f})")
    ax.set_title("(c) β Distribution by Epoch")
    ax.set_xlabel("β_guide value")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"beta_dynamics_{subtask}.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def cross_subtask_summary(results):
    """Cross-subtask comparison: how do RePO and AdaRePO compare across subtasks?"""
    subtasks = sorted(results.keys())
    n = len(subtasks)
    if n == 0:
        print("  No results to plot for cross-subtask summary")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Cross-Subtask Comparison: RePO vs AdaRePO", fontsize=16, fontweight="bold")

    last_n = 20
    metrics = ["reward", "loss", "kl", "s_loss", "grad_norm", "completion_length"]
    titles = ["Final Reward ↑", "Final Loss ↓", "KL Divergence ↓",
              "Guidance Loss", "Gradient Norm ↓", "Completion Length"]

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3][idx % 3]
        x = np.arange(n)
        w = 0.35

        repo_vals = []
        ada_vals = []
        repo_stds = []
        ada_stds = []

        for st in subtasks:
            r = results[st]
            if "repo" in r and r["repo"] is not None:
                vals = r["repo"][metric][-last_n:]
                repo_vals.append(vals.mean())
                repo_stds.append(vals.std())
            else:
                repo_vals.append(0)
                repo_stds.append(0)

            if "ada" in r and r["ada"] is not None:
                vals = r["ada"][metric][-last_n:]
                ada_vals.append(vals.mean())
                ada_stds.append(vals.std())
            else:
                ada_vals.append(0)
                ada_stds.append(0)

        bars1 = ax.bar(x - w/2, repo_vals, w, yerr=repo_stds, label="RePO",
                       color=COLOR_REPO, alpha=0.8, capsize=3)
        bars2 = ax.bar(x + w/2, ada_vals, w, yerr=ada_stds, label="AdaRePO",
                       color=COLOR_ADA, alpha=0.8, capsize=3)

        # Value labels
        for bar, val in zip(bars1, repo_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, ada_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.3f}", ha='center', va='bottom', fontsize=8)

        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(subtasks)
        ax.legend()
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "cross_subtask_summary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def beta_consistency_plot(results):
    """Show beta dynamics consistency across subtasks."""
    subtasks = [st for st in sorted(results.keys())
                if "ada" in results[st] and results[st]["ada"] is not None
                and results[st]["ada"]["is_ada"]]

    if not subtasks:
        print("  No AdaRePO results with beta data found")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Dynamic β Consistency Across Subtasks", fontsize=14, fontweight="bold")

    # (a) Beta over time for all subtasks
    ax = axes[0]
    for st in subtasks:
        d = results[st]["ada"]
        ax.plot(d["step"], ema(d["beta_guide"]), color=SUBTASK_COLORS.get(st, "gray"),
                lw=2, label=st)
    ax.axhline(1.0, ls="--", color="gray", alpha=0.5)
    ax.set_title("(a) β_guide Over Training")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("β_guide")
    ax.legend()
    ax.grid(alpha=0.3)

    # (b) Reward gap for all subtasks
    ax = axes[1]
    for st in subtasks:
        d = results[st]["ada"]
        ax.plot(d["step"], ema(d["v_top_minus_v_ref"]), color=SUBTASK_COLORS.get(st, "gray"),
                lw=2, label=st)
    ax.axhline(0, ls="-", color="gray", alpha=0.3)
    ax.set_title("(b) Reward Gap (v_top − v_ref)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Gap")
    ax.legend()
    ax.grid(alpha=0.3)

    # (c) Final beta by subtask
    ax = axes[2]
    last_n = 20
    x = np.arange(len(subtasks))
    means = [results[st]["ada"]["beta_guide"][-last_n:].mean() for st in subtasks]
    stds = [results[st]["ada"]["beta_guide"][-last_n:].std() for st in subtasks]
    colors = [SUBTASK_COLORS.get(st, "gray") for st in subtasks]
    bars = ax.bar(x, means, yerr=stds, color=colors, alpha=0.8, capsize=5)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f"{val:.3f}", ha='center', va='bottom', fontsize=10)
    ax.set_title("(c) Final β (last 20 steps)")
    ax.set_xticks(x)
    ax.set_xticklabels(subtasks)
    ax.set_ylabel("β_guide")
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "beta_consistency.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


def delta_table(results):
    """Print and save a delta table: AdaRePO - RePO for each metric and subtask."""
    subtasks = sorted(results.keys())
    last_n = 20
    metrics = ["reward", "loss", "kl", "s_loss"]
    headers = ["Subtask"] + [f"Δ{m}" for m in metrics] + ["β_final", "IL_frac"]

    rows = []
    for st in subtasks:
        r = results[st]
        repo_d = r.get("repo")
        ada_d = r.get("ada")
        if repo_d is None or ada_d is None:
            continue

        row = [st]
        for m in metrics:
            delta = ada_d[m][-last_n:].mean() - repo_d[m][-last_n:].mean()
            row.append(f"{delta:+.4f}")

        if ada_d["is_ada"]:
            beta_final = ada_d["beta_guide"][-last_n:].mean()
            il_frac = (ada_d["beta_guide"] > 0.5).sum() / len(ada_d["beta_guide"])
            row.append(f"{beta_final:.3f}")
            row.append(f"{il_frac:.1%}")
        else:
            row.append("N/A")
            row.append("N/A")

        rows.append(row)

    # Print table
    print("\n" + "=" * 80)
    print("CROSS-SUBTASK DELTA TABLE (AdaRePO − RePO, last 20 steps)")
    print("=" * 80)
    col_widths = [max(len(str(r[i])) for r in [headers] + rows) + 2 for i in range(len(headers))]
    fmt = "".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * sum(col_widths))
    for row in rows:
        print(fmt.format(*row))

    # Save as markdown
    md_path = os.path.join(OUT_DIR, "delta_table.md")
    with open(md_path, "w") as f:
        f.write("# Cross-Subtask Delta Table: AdaRePO − RePO\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    print(f"  Saved: {md_path}")

    return rows


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Cross-subtask analysis")
    parser.add_argument("--subtasks", nargs="+", default=["LogP", "MR", "QED"])
    args = parser.parse_args()

    print("Cross-Subtask Analysis: RePO vs AdaRePO")
    print("=" * 60)

    results = {}

    for subtask in args.subtasks:
        print(f"\nProcessing {subtask}...")

        # Find logs
        repo_log = find_log(REPO_LOG_DIR, "RePO", subtask)
        ada_log = find_log(ADA_LOG_DIR, "AdaRePO", subtask)

        if repo_log:
            print(f"  RePO log:    {os.path.basename(repo_log)}")
        else:
            print(f"  RePO log:    NOT FOUND")

        if ada_log:
            print(f"  AdaRePO log: {os.path.basename(ada_log)}")
        else:
            print(f"  AdaRePO log: NOT FOUND")

        repo_data = parse_log(repo_log) if repo_log else None
        ada_data = parse_log(ada_log) if ada_log else None

        if repo_data:
            print(f"  RePO:    {len(repo_data['step'])} steps")
        if ada_data:
            print(f"  AdaRePO: {len(ada_data['step'])} steps")

        results[subtask] = {"repo": repo_data, "ada": ada_data}

        # Per-subtask plots
        if repo_data is not None and ada_data is not None:
            print(f"  Generating per-subtask comparison...")
            per_subtask_comparison(subtask, repo_data, ada_data)
            beta_dynamics_per_subtask(subtask, ada_data)

    # Cross-subtask plots
    print(f"\nGenerating cross-subtask summary...")
    cross_subtask_summary(results)
    beta_consistency_plot(results)
    delta_table(results)

    print(f"\n{'='*60}")
    print(f"All figures saved to: {OUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
