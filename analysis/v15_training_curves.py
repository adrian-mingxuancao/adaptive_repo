"""
Plot v15 training curves: reward mean, reward_std, kl, s_loss, loss
from WandB local logs (offline-compatible).
"""
import json
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WANDB_BASE = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/wandb"

RUNS = {
    "v15_repo_baseline": "run-20260414_140319-dzzm847n",
    "v15_static":        "run-20260415_223544-abaz28qy",
    "v15_beta_only":     "run-20260415_224229-0takqasn",
    "v15_active":        "run-20260415_224229-sgs990gf",
}

COLORS = {
    "v15_repo_baseline": "#1f77b4",  # blue
    "v15_static":        "#2ca02c",  # green
    "v15_beta_only":     "#ff7f0e",  # orange
    "v15_active":        "#d62728",  # red
}

def load_wandb_history(run_dir):
    """Load metrics from wandb local run directory."""
    history_file = os.path.join(WANDB_BASE, run_dir, "run-" + run_dir.split("-", 1)[1].split("/")[0] + ".wandb")
    
    # Try the jsonl history file
    jsonl_dir = os.path.join(WANDB_BASE, run_dir)
    jsonl_files = glob.glob(os.path.join(jsonl_dir, "*.wandb"))
    
    if not jsonl_files:
        # Try files subdirectory
        jsonl_files = glob.glob(os.path.join(jsonl_dir, "files", "wandb-history.jsonl"))
    
    if not jsonl_files:
        # Try run-history.json
        jsonl_files = glob.glob(os.path.join(jsonl_dir, "files", "*.json"))
    
    metrics = {}
    
    # Parse the binary wandb file using the summary json as fallback
    summary_file = os.path.join(jsonl_dir, "files", "wandb-summary.json")
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"  Summary keys: {sorted(summary.keys())[:20]}...")
    
    # Try to read from the events file
    events_files = sorted(glob.glob(os.path.join(jsonl_dir, "files", "*.jsonl")))
    if not events_files:
        events_files = sorted(glob.glob(os.path.join(jsonl_dir, "logs", "*.log")))
    
    return metrics, events_files


def parse_log_for_metrics(log_file):
    """Parse the PBS log file to extract training metrics per step."""
    steps = []
    reward_means = []
    reward_stds = []
    kls = []
    s_losses = []
    losses = []
    
    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            # Look for training log lines like: {'loss': 0.123, 'reward': 0.456, ...}
            if "'loss'" in line and "'reward'" in line:
                try:
                    # Extract the dict-like string
                    start = line.index("{")
                    end = line.rindex("}") + 1
                    d = eval(line[start:end])
                    if "train_step" in d or "loss" in d:
                        steps.append(d.get("train_step", len(steps)))
                        reward_means.append(d.get("reward", None))
                        reward_stds.append(d.get("reward_std", None))
                        kls.append(d.get("kl", None))
                        s_losses.append(d.get("s_loss", None))
                        losses.append(d.get("loss", None))
                except:
                    pass
    
    return {
        "step": steps,
        "reward": reward_means,
        "reward_std": reward_stds,
        "kl": kls,
        "s_loss": s_losses,
        "loss": losses,
    }


LOG_FILES = {
    "v15_repo_baseline": "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/v15_v15_repo_baseline_7089030.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log",
    "v15_static":        "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/v15_v15_static_7090220.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log",
    "v15_beta_only":     "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/v15_v15_beta_only_7090221.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log",
    "v15_active":        "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/logs/v15_v15_active_7090222.polaris-pbs-01.hsn.cm.polaris.alcf.anl.gov.log",
}


def parse_hf_log(log_file):
    """Parse HuggingFace Trainer log lines from the PBS log."""
    import re
    data = {"step": [], "loss": [], "reward": [], "reward_std": [], "kl": [], "s_loss": [],
            "s_loss_weighted": [], "beta_guide_mean": [], "beta_guide_std": [],
            "advantage_mean": [], "advantage_std": [], "advantage_max": [],
            "completion_length": [], "learning_rate": []}
    
    pattern = re.compile(r"\{[^{}]*'loss'[^{}]*\}")
    
    with open(log_file, "r", errors="ignore") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    d = eval(m.group())
                    step = d.get("epoch", len(data["step"]))
                    data["step"].append(len(data["step"]) + 1)
                    for key in data:
                        if key == "step":
                            continue
                        data[key].append(d.get(key, None))
                except:
                    pass
    return data


def main():
    out_dir = "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/figures"
    os.makedirs(out_dir, exist_ok=True)
    
    all_data = {}
    for name, log_file in LOG_FILES.items():
        print(f"Parsing {name}...")
        data = parse_hf_log(log_file)
        print(f"  Found {len(data['step'])} steps")
        if data["step"]:
            for key in ["reward", "loss", "kl", "s_loss"]:
                vals = [v for v in data[key] if v is not None]
                if vals:
                    print(f"  {key}: mean={np.mean(vals):.4f}, last={vals[-1]:.4f}")
        all_data[name] = data
    
    # ---- Plot 1: Reward Mean ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("v15 Experiment: Training Curves (LogP, maxlen=1024, 120 steps)", fontsize=14, fontweight="bold")
    
    metrics_to_plot = [
        ("reward", "Reward (mean)", axes[0, 0]),
        ("reward_std", "Reward Std (within group)", axes[0, 1]),
        ("kl", "KL Divergence", axes[0, 2]),
        ("s_loss", "Guidance Loss (s_loss)", axes[1, 0]),
        ("loss", "Total Loss", axes[1, 1]),
        ("completion_length", "Completion Length", axes[1, 2]),
    ]
    
    for metric, title, ax in metrics_to_plot:
        for name, data in all_data.items():
            vals = data.get(metric, [])
            steps = data["step"]
            if vals and any(v is not None for v in vals):
                clean_steps = [s for s, v in zip(steps, vals) if v is not None]
                clean_vals = [v for v in vals if v is not None]
                # Smoothing with moving average (window=5)
                if len(clean_vals) > 5:
                    kernel = np.ones(5) / 5
                    smooth = np.convolve(clean_vals, kernel, mode="valid")
                    smooth_steps = clean_steps[2:-2]
                    ax.plot(smooth_steps, smooth, label=name, color=COLORS[name], linewidth=1.5)
                    ax.plot(clean_steps, clean_vals, color=COLORS[name], alpha=0.15, linewidth=0.5)
                else:
                    ax.plot(clean_steps, clean_vals, label=name, color=COLORS[name])
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "v15_training_curves.png"), dpi=150)
    print(f"\nSaved: {out_dir}/v15_training_curves.png")
    
    # ---- Plot 2: AdaRePO-specific metrics ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("v15 AdaRePO-specific Metrics", fontsize=14, fontweight="bold")
    
    ada_metrics = [
        ("beta_guide_mean", "Beta Guide (mean)", axes2[0, 0]),
        ("s_loss_weighted", "Weighted Guidance Loss (beta×s_loss)", axes2[0, 1]),
        ("advantage_mean", "Advantage Mean", axes2[1, 0]),
        ("advantage_std", "Advantage Std", axes2[1, 1]),
    ]
    
    for metric, title, ax in ada_metrics:
        for name, data in all_data.items():
            vals = data.get(metric, [])
            steps = data["step"]
            if vals and any(v is not None for v in vals):
                clean_steps = [s for s, v in zip(steps, vals) if v is not None]
                clean_vals = [v for v in vals if v is not None]
                if len(clean_vals) > 5:
                    kernel = np.ones(5) / 5
                    smooth = np.convolve(clean_vals, kernel, mode="valid")
                    smooth_steps = clean_steps[2:-2]
                    ax.plot(smooth_steps, smooth, label=name, color=COLORS[name], linewidth=1.5)
                    ax.plot(clean_steps, clean_vals, color=COLORS[name], alpha=0.15, linewidth=0.5)
                else:
                    ax.plot(clean_steps, clean_vals, label=name, color=COLORS[name])
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "v15_ada_metrics.png"), dpi=150)
    print(f"Saved: {out_dir}/v15_ada_metrics.png")
    
    # ---- Print summary table ----
    print("\n" + "="*80)
    print("SUMMARY: Last 10-step averages")
    print("="*80)
    print(f"{'Run':<25} {'Reward':>8} {'Rew_Std':>8} {'KL':>8} {'s_loss':>8} {'Loss':>8}")
    print("-"*80)
    for name, data in all_data.items():
        def last_n(vals, n=10):
            clean = [v for v in vals if v is not None]
            return np.mean(clean[-n:]) if clean else float('nan')
        print(f"{name:<25} {last_n(data['reward']):>8.4f} {last_n(data['reward_std']):>8.4f} "
              f"{last_n(data['kl']):>8.4f} {last_n(data['s_loss']):>8.4f} {last_n(data['loss']):>8.4f}")


if __name__ == "__main__":
    main()
