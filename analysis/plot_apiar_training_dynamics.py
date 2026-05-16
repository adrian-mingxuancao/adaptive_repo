"""
Plot multi-seed APIAR training dynamics directly from HuggingFace trainer_state.json
files written in each output directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo")
OUTPUT_ROOT = ROOT / "output"
FIGURE_DIR = ROOT / "analysis" / "figures"


@dataclass(frozen=True)
class RunGroup:
    label: str
    patterns: Sequence[str]
    color: str


RUN_GROUPS: Sequence[RunGroup] = (
    RunGroup(
        label="APIAR Full (v17)",
        patterns=("v16v17ms_v17_s*",),
        color="#d1495b",
    ),
    RunGroup(
        label="Beta-Only",
        patterns=("ablation_beta_only_s*",),
        color="#2e86ab",
    ),
    RunGroup(
        label="Bank-Only",
        patterns=("ablation_bank_only_s*",),
        color="#3a7d44",
    ),
    RunGroup(
        label="Hard Data Full (v18)",
        patterns=("v18ms_hard_s*",),
        color="#8d5a97",
    ),
)


METRICS = (
    ("reward", "Reward", "Mean rollout reward"),
    ("loss", "Loss", "Total RL loss"),
    ("beta_guide_mean", "Beta Guide", "Adaptive guidance strength"),
    ("memory_bank/total_entries", "Memory Bank", "Mean entries in memory bank"),
    ("frac_self_distill", "Self Distill", "Fraction self-distilled"),
    ("v_top_minus_v_ref", "Top - Ref", "Reward gap vs reference"),
    ("kl", "KL", "KL from reference"),
    ("completion_length", "Completion Len", "Generated completion length"),
)

Y_LIMITS = {
    "reward": (-0.2, 1.8),
    "loss": (0.0, 1.25),
    "beta_guide_mean": (0.0, 1.3),
    "memory_bank/total_entries": (0.0, 220.0),
    "frac_self_distill": (0.0, 0.25),
    "v_top_minus_v_ref": (-1.5, 2.5),
    "kl": (-0.02, 1.0),
    "completion_length": (50.0, 220.0),
}


def _discover_state_files(patterns: Sequence[str]) -> List[Path]:
    state_files: List[Path] = []
    for pattern in patterns:
        for run_dir in sorted(OUTPUT_ROOT.glob(pattern)):
            state_path = run_dir / "trainer_state.json"
            if state_path.exists():
                state_files.append(state_path)
    return state_files


def _load_log_history(path: Path) -> List[dict]:
    with path.open() as f:
        payload = json.load(f)
    history = payload.get("log_history", [])
    return [entry for entry in history if isinstance(entry, dict) and "step" in entry]


def _extract_series(history: Iterable[dict], key: str) -> Dict[int, float]:
    series: Dict[int, float] = {}
    for entry in history:
        step = entry.get("step")
        value = entry.get(key)
        if isinstance(step, (int, float)) and isinstance(value, (int, float)):
            series[int(step)] = float(value)
    return series


def _aggregate_seed_series(state_files: Sequence[Path], key: str) -> Dict[str, np.ndarray]:
    per_seed = []
    for path in state_files:
        history = _load_log_history(path)
        values_by_step = _extract_series(history, key)
        if values_by_step:
            per_seed.append(values_by_step)
    if not per_seed:
        return {}

    all_steps = sorted({step for series in per_seed for step in series})
    matrix = np.full((len(per_seed), len(all_steps)), np.nan, dtype=float)
    for i, series in enumerate(per_seed):
        for j, step in enumerate(all_steps):
            if step in series:
                matrix[i, j] = series[step]

    return {
        "steps": np.asarray(all_steps, dtype=float),
        "values": matrix,
        "mean": np.nanmean(matrix, axis=0),
        "std": np.nanstd(matrix, axis=0),
        "n": np.sum(~np.isnan(matrix), axis=0),
    }


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=float) / window
    valid = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="same")
    counts = np.convolve(~np.isnan(values), np.ones(window, dtype=float), mode="same")
    counts = np.where(counts == 0, 1.0, counts)
    return valid / counts


def _plot_metric(ax: plt.Axes, metric_key: str, title: str, ylabel: str) -> None:
    for group in RUN_GROUPS:
        state_files = _discover_state_files(group.patterns)
        agg = _aggregate_seed_series(state_files, metric_key)
        if not agg:
            continue

        steps = agg["steps"]
        mean = _smooth(agg["mean"])
        std = agg["std"]
        for seed_values in agg["values"]:
            if np.all(np.isnan(seed_values)):
                continue
            ax.plot(
                steps,
                _smooth(seed_values),
                color=group.color,
                alpha=0.18,
                linewidth=0.8,
            )
        ax.plot(
            steps,
            mean,
            color=group.color,
            linewidth=2.2,
            label=f"{group.label} (n={len(state_files)})",
        )
        ax.fill_between(
            steps,
            mean - std,
            mean + std,
            color=group.color,
            alpha=0.12,
        )

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if metric_key in Y_LIMITS:
        ax.set_ylim(*Y_LIMITS[metric_key])


def _write_summary() -> Path:
    summary_path = FIGURE_DIR / "apiar_training_summary.txt"
    lines = []
    header = (
        f"{'Run Group':<22} {'Metric':<18} {'Start':>10} {'Last10Med':>10} "
        f"{'Last10Avg':>10} {'PeakSmooth':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    summary_metrics = ("reward", "loss", "beta_guide_mean", "memory_bank/total_entries", "frac_self_distill")
    for group in RUN_GROUPS:
        state_files = _discover_state_files(group.patterns)
        for metric in summary_metrics:
            agg = _aggregate_seed_series(state_files, metric)
            if not agg:
                continue
            mean = agg["mean"]
            clean = mean[~np.isnan(mean)]
            if clean.size == 0:
                continue
            smooth = _smooth(clean)
            last10 = clean[-10:] if clean.size >= 10 else clean
            lines.append(
                f"{group.label:<22} {metric:<18} {clean[0]:>10.4f} {np.nanmedian(last10):>10.4f} "
                f"{np.nanmean(last10):>10.4f} {np.nanmax(smooth):>10.4f}"
            )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n")
    return summary_path


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("APIAR RL Training Dynamics — Main Signals", fontsize=15, fontweight="bold")
    main_metrics = (
        ("reward", "Reward", "Mean rollout reward"),
        ("loss", "Loss", "Total RL loss"),
        ("beta_guide_mean", "Beta Guide", "Adaptive guidance strength"),
        ("memory_bank/total_entries", "Memory Bank", "Mean bank entries"),
    )
    for ax, (metric, title, ylabel) in zip(axes.flat, main_metrics):
        _plot_metric(ax, metric, title, ylabel)
    axes[0, 0].legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGURE_DIR / "apiar_training_dynamics_main.png", dpi=180)
    fig.savefig(FIGURE_DIR / "apiar_training_dynamics_main.pdf")
    plt.close(fig)

    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle("APIAR RL Training Dynamics — Mechanism Signals", fontsize=15, fontweight="bold")
    mechanism_metrics = (
        ("frac_self_distill", "Self Distillation", "Fraction self-distilled"),
        ("v_top_minus_v_ref", "Top - Reference", "Reward gap"),
        ("kl", "KL", "KL from reference"),
        ("completion_length", "Completion Length", "Generated tokens"),
    )
    for ax, (metric, title, ylabel) in zip(axes2.flat, mechanism_metrics):
        _plot_metric(ax, metric, title, ylabel)
    axes2[0, 0].legend(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(FIGURE_DIR / "apiar_training_dynamics_mechanism.png", dpi=180)
    fig2.savefig(FIGURE_DIR / "apiar_training_dynamics_mechanism.pdf")
    plt.close(fig2)

    summary_path = _write_summary()
    print(f"Saved figures to: {FIGURE_DIR}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
