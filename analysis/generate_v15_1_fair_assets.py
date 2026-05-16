"""Generate AdaRePO v15.1 boosted vs RePO DSI fair-comparison training assets."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRATCH_ROOT = Path("/net/scratch/caom/repo_project")
OUT_DIR = Path("/home/caom/rl-agent/agent_drug_discovery/adaptive_repo/analysis/v15_1_fair_assets")

RUNS = {
    "RePO DSI": [
        SCRATCH_ROOT / "logs" / "repo_propopt_791601.out",
        SCRATCH_ROOT / "logs" / "repo_propopt_resume_795397.out",
    ],
    "v15.1 Boosted": [
        SCRATCH_ROOT / "logs" / "v15_1_boosted_812287.out",
    ],
    "v15.1 Retrain": [
        SCRATCH_ROOT / "outputs" / "ada_repo_dsi_v15_1_retrain" / "trainer_state.json",
    ],
    "v16": [
        SCRATCH_ROOT / "outputs" / "ada_repo_dsi_v16" / "trainer_state.json",
    ],
}

COLORS = {
    "RePO DSI": "#1f4e79",
    "v15.1 Boosted": "#b5525c",
    "v15.1 Retrain": "#2a9d8f",
    "v16": "#8e5ea2",
}

EVAL_RUNS = {
    "RePO": SCRATCH_ROOT / "outputs" / "eval_ckpt748" / "predictions" / "checkpoint-748" / "open_generation" / "MolOpt",
    "v15.1 Retrain": SCRATCH_ROOT / "outputs" / "eval_v15_1_retrain" / "predictions" / "checkpoint-748" / "open_generation" / "MolOpt",
    "v16": SCRATCH_ROOT / "outputs" / "eval_v16" / "predictions" / "checkpoint-748" / "open_generation" / "MolOpt",
}

EVAL_COLORS = {
    "RePO": "#1f4e79",
    "v15.1 Retrain": "#2a9d8f",
    "v16": "#8e5ea2",
}

TASKS = ["LogP", "MR", "QED"]

BG = "#fbfaf7"
AXIS = "#222222"
GRID = "#d6d6d6"
NOTE_BG = "#f3ead2"
NOTE_EDGE = "#d9c9a3"

MAIN_METRICS = [
    ("reward", "Reward Mean", None),
    ("reward_std", "Reward Std", None),
    ("advantage/pos_frac", "Advantage Positive Fraction", None),
    ("advantage/std", "Advantage Std", None),
]

RL_METRICS = [
    ("advantage/max", "Advantage Max", None),
    ("advantage/min", "Advantage Min", None),
    ("s_loss", "Guidance Loss", None),
    ("kl", "KL (clipped at 2.0)", 2.0),
]

METRIC_ALIASES = {
    "advantage/mean": ["advantage_mean"],
    "advantage/std": ["advantage_std"],
    "advantage/min": ["advantage_min"],
    "advantage/max": ["advantage_max"],
    "advantage/abs_mean": ["advantage_abs_mean"],
    "advantage/pos_frac": ["frac_positive_advantage"],
}


def get_fonts() -> tuple[ImageFont.FreeTypeFont | ImageFont.ImageFont, ...]:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        try:
            return (
                ImageFont.truetype(path, 28),
                ImageFont.truetype(path, 20),
                ImageFont.truetype(path, 16),
                ImageFont.truetype(path, 14),
            )
        except OSError:
            continue
    default = ImageFont.load_default()
    return (default, default, default, default)


TITLE_FONT, LABEL_FONT, TICK_FONT, SMALL_FONT = get_fonts()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: str | tuple[int, int, int],
) -> None:
    width, height = text_size(draw, text, font)
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, font=font, fill=fill)


def parse_log_dicts(paths: Iterable[Path]) -> list[dict[str, float]]:
    entries: list[dict[str, float]] = []
    safe_globals = {"nan": float("nan"), "inf": float("inf"), "__builtins__": {}}
    for path in paths:
        if not path.exists():
            continue
        if path.name == "trainer_state.json":
            payload = json.loads(path.read_text())
            for row in payload.get("log_history", []):
                if isinstance(row, dict) and "epoch" in row and "reward" in row:
                    entries.append(row)
            continue
        with path.open(errors="ignore") as handle:
            for line in handle:
                line = line.strip()
                if not (line.startswith("{") and "'epoch'" in line and "'reward'" in line):
                    continue
                try:
                    payload = eval(line, safe_globals)
                except Exception:
                    continue
                if isinstance(payload, dict) and "epoch" in payload:
                    entries.append(payload)
    return aggregate_by_epoch(entries)


def read_eval_row(path: Path) -> dict[str, float]:
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    sr = float(row["success_rate"])
    sim = float(row["similarity"])
    validity = float(row["validity"])
    return {
        "sr": sr,
        "similarity": sim,
        "validity": validity,
        "sr_x_sim": sr * sim,
    }


def collect_eval_rows() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for method, eval_dir in EVAL_RUNS.items():
        row: dict[str, float | str] = {"method": method}
        vals = []
        for task in TASKS:
            metrics = read_eval_row(eval_dir / f"{task}_summary.csv")
            for key, value in metrics.items():
                row[f"{task.lower()}_{key}"] = value
            vals.append(metrics["sr_x_sim"])
        row["avg_sr_x_sim"] = sum(vals) / len(vals)
        rows.append(row)
    return rows


def aggregate_by_epoch(entries: list[dict[str, float]]) -> list[dict[str, float]]:
    grouped: dict[float, list[dict[str, float]]] = defaultdict(list)
    for entry in entries:
        grouped[round(float(entry["epoch"]), 4)].append(entry)

    rows: list[dict[str, float]] = []
    for epoch in sorted(grouped):
        merged: dict[str, float] = {"epoch": epoch}
        keys = sorted({key for item in grouped[epoch] for key in item})
        for key in keys:
            values = []
            for item in grouped[epoch]:
                value = item.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                merged[key] = float(np.mean(values))
        rows.append(merged)
    return rows


def ema(values: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    if values.size == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    smoothed.fill(np.nan)
    last = float("nan")
    for idx, value in enumerate(values):
        if math.isfinite(float(value)):
            if math.isfinite(last):
                last = alpha * float(value) + (1.0 - alpha) * last
            else:
                last = float(value)
        if math.isfinite(last):
            smoothed[idx] = last
    return smoothed


def nice_limits(values: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([arr[np.isfinite(arr)] for arr in values if arr.size > 0])
    if finite.size == 0:
        return (0.0, 1.0)
    low = float(np.min(finite))
    high = float(np.max(finite))
    if math.isclose(low, high):
        pad = 0.1 if math.isclose(low, 0.0) else abs(low) * 0.15
        return (low - pad, high + pad)
    pad = (high - low) * 0.10
    return (low - pad, high + pad)


def map_point(
    x: float,
    y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    box: tuple[float, float, float, float],
) -> tuple[float, float]:
    left, top, right, bottom = box
    px = left + (x - x_min) / (x_max - x_min) * (right - left)
    py = bottom - (y - y_min) / (y_max - y_min) * (bottom - top)
    return (px, py)


def prepare_series(metric_specs: list[tuple[str, str, float | None]]) -> dict[str, dict[str, np.ndarray]]:
    prepared: dict[str, dict[str, np.ndarray]] = {}
    for method, paths in RUNS.items():
        rows = parse_log_dicts(paths)
        prepared[method] = {"epoch": np.array([row["epoch"] for row in rows], dtype=float)}
        for key, _, clip in metric_specs:
            values = np.array([get_metric(row, key) for row in rows], dtype=float)
            if clip is not None:
                values = np.minimum(values, clip)
            prepared[method][key] = ema(values)
    return prepared


def get_metric(row: dict[str, float], metric: str) -> float:
    value = row.get(metric)
    if value is not None and math.isfinite(float(value)):
        return float(value)
    for alias in METRIC_ALIASES.get(metric, []):
        value = row.get(alias)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return float("nan")


def draw_metric_grid(
    filename: str,
    title: str,
    metric_specs: list[tuple[str, str, float | None]],
    note_lines: list[str],
) -> None:
    prepared = prepare_series(metric_specs)
    max_epoch = max(float(np.nanmax(series["epoch"])) for series in prepared.values() if series["epoch"].size)
    width, height = 1500, 900
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw_centered_text(draw, (width / 2, 30), title, TITLE_FONT, AXIS)

    legend_xs = [95, 395]
    legend_y0 = 62
    for idx, method in enumerate(RUNS):
        legend_x = legend_xs[idx % 2]
        legend_y = legend_y0 + 30 * (idx // 2)
        color = COLORS[method]
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 26, legend_y + 16), radius=4, fill=color)
        draw.text((legend_x + 36, legend_y - 5), method, font=LABEL_FONT, fill=AXIS)

    note_box = (width - 480, 52, width - 55, 122)
    draw.rounded_rectangle(note_box, radius=12, fill=NOTE_BG, outline=NOTE_EDGE, width=2)
    y = note_box[1] + 12
    for line in note_lines:
        draw.text((note_box[0] + 14, y), line, font=SMALL_FONT, fill=AXIS)
        y += 20

    panel_w, panel_h = 640, 310
    lefts = [85, 780]
    tops = [150, 510]

    for idx, (key, label, _) in enumerate(metric_specs):
        left = lefts[idx % 2]
        top = tops[idx // 2]
        panel_box = (left, top, left + panel_w, top + panel_h)
        plot_box = (left + 65, top + 45, left + panel_w - 28, top + panel_h - 52)
        draw.rounded_rectangle(panel_box, radius=14, outline="#d2d2d2", width=2, fill="#ffffff")
        draw_centered_text(draw, (left + panel_w / 2, top + 22), label, LABEL_FONT, AXIS)

        y_min, y_max = nice_limits([prepared[m][key] for m in prepared])
        x_ticks = np.linspace(0.0, max_epoch, 5)
        y_ticks = np.linspace(y_min, y_max, 5)

        for tick in y_ticks:
            y_px = map_point(0.0, float(tick), 0.0, max_epoch, y_min, y_max, plot_box)[1]
            draw.line((plot_box[0], y_px, plot_box[2], y_px), fill=GRID, width=1)
            draw.text((left + 8, y_px - 8), f"{tick:.2f}", font=TICK_FONT, fill="#555555")

        for tick in x_ticks:
            x_px = map_point(float(tick), y_min, 0.0, max_epoch, y_min, y_max, plot_box)[0]
            draw.line((x_px, plot_box[1], x_px, plot_box[3]), fill=GRID, width=1)
            tw, _ = text_size(draw, f"{tick:.1f}", TICK_FONT)
            draw.text((x_px - tw / 2, plot_box[3] + 10), f"{tick:.1f}", font=TICK_FONT, fill="#555555")

        draw.line((plot_box[0], plot_box[3], plot_box[2], plot_box[3]), fill=AXIS, width=2)
        draw.line((plot_box[0], plot_box[1], plot_box[0], plot_box[3]), fill=AXIS, width=2)
        draw_centered_text(draw, ((plot_box[0] + plot_box[2]) / 2, top + panel_h - 16), "Epoch", SMALL_FONT, AXIS)

        for method in RUNS:
            epochs = prepared[method]["epoch"]
            values = prepared[method][key]
            valid = np.isfinite(epochs) & np.isfinite(values)
            points = [
                map_point(float(x), float(val), 0.0, max_epoch, y_min, y_max, plot_box)
                for x, val in zip(epochs[valid], values[valid])
            ]
            if len(points) >= 2:
                draw.line(points, fill=COLORS[method], width=4)

    image.save(OUT_DIR / filename)


def write_summary(prepared_rows: dict[str, list[dict[str, float]]]) -> None:
    metrics = [
        "reward",
        "reward_std",
        "advantage/abs_mean",
        "advantage/std",
        "advantage/pos_frac",
        "advantage/zero_frac",
        "kl",
        "s_loss",
        "beta_guide_mean",
    ]
    path = OUT_DIR / "training_summary.csv"
    with path.open("w", newline="") as handle:
        fieldnames = ["method", "metric", "first", "mid", "last", "min", "max"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, rows in prepared_rows.items():
            for metric in metrics:
                values = [get_metric(row, metric) for row in rows]
                values = [value for value in values if math.isfinite(value)]
                if not values:
                    continue
                writer.writerow(
                    {
                        "method": method,
                        "metric": metric,
                        "first": values[0],
                        "mid": values[len(values) // 2],
                        "last": values[-1],
                        "min": min(values),
                        "max": max(values),
                    }
                )


def write_eval_table(rows: list[dict[str, float | str]]) -> None:
    csv_path = OUT_DIR / "table_v15_1_v16_eval_comparison.csv"
    md_path = OUT_DIR / "table_v15_1_v16_eval_comparison.md"
    fieldnames = ["method"]
    for task in TASKS:
        prefix = task.lower()
        fieldnames.extend(
            [
                f"{prefix}_sr",
                f"{prefix}_similarity",
                f"{prefix}_sr_x_sim",
                f"{prefix}_validity",
            ]
        )
    fieldnames.append("avg_sr_x_sim")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    headers = [
        "Method",
        "LogP SR",
        "LogP Sim",
        "LogP SRxSim",
        "LogP Val",
        "MR SR",
        "MR Sim",
        "MR SRxSim",
        "MR Val",
        "QED SR",
        "QED Sim",
        "QED SRxSim",
        "QED Val",
        "Avg SRxSim",
    ]
    with md_path.open("w") as handle:
        handle.write("# RePO vs v15.1 Retrain vs v16 - Eval Comparison\n\n")
        handle.write("Sources are the existing `*_summary.csv` files under scratch eval directories.\n\n")
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for row in rows:
            handle.write(
                "| "
                + " | ".join(
                    [
                        str(row["method"]),
                        f"{row['logp_sr']:.4f}",
                        f"{row['logp_similarity']:.4f}",
                        f"{row['logp_sr_x_sim']:.4f}",
                        f"{row['logp_validity']:.4f}",
                        f"{row['mr_sr']:.4f}",
                        f"{row['mr_similarity']:.4f}",
                        f"{row['mr_sr_x_sim']:.4f}",
                        f"{row['mr_validity']:.4f}",
                        f"{row['qed_sr']:.4f}",
                        f"{row['qed_similarity']:.4f}",
                        f"{row['qed_sr_x_sim']:.4f}",
                        f"{row['qed_validity']:.4f}",
                        f"{row['avg_sr_x_sim']:.4f}",
                    ]
                )
                + " |\n"
            )


def plot_eval_srxsim(rows: list[dict[str, float | str]]) -> None:
    width, height = 1280, 720
    image = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw_centered_text(draw, (width / 2, 30), "RePO vs v15.1 Retrain vs v16 - Eval", TITLE_FONT, AXIS)

    plot_box = (110, 120, width - 60, height - 120)
    draw.rounded_rectangle((70, 88, width - 35, height - 55), radius=14, outline="#d2d2d2", width=2, fill="#ffffff")

    categories = TASKS + ["Avg"]
    values_by_method: dict[str, list[float]] = {}
    for row in rows:
        method = str(row["method"])
        values_by_method[method] = [
            float(row["logp_sr_x_sim"]),
            float(row["mr_sr_x_sim"]),
            float(row["qed_sr_x_sim"]),
            float(row["avg_sr_x_sim"]),
        ]

    y_max = max(max(vals) for vals in values_by_method.values()) * 1.18
    y_ticks = np.linspace(0.0, y_max, 5)
    x_centers = np.linspace(plot_box[0] + 130, plot_box[2] - 80, len(categories))

    for tick in y_ticks:
        y_px = map_point(0.0, float(tick), 0.0, 1.0, 0.0, y_max, plot_box)[1]
        draw.line((plot_box[0], y_px, plot_box[2], y_px), fill=GRID, width=1)
        draw.text((35, y_px - 8), f"{tick:.2f}", font=TICK_FONT, fill="#555555")

    draw.line((plot_box[0], plot_box[3], plot_box[2], plot_box[3]), fill=AXIS, width=2)
    draw.line((plot_box[0], plot_box[1], plot_box[0], plot_box[3]), fill=AXIS, width=2)
    draw.text((35, plot_box[1] - 24), "SRxSim", font=LABEL_FONT, fill=AXIS)

    method_order = [str(row["method"]) for row in rows]
    group_width = 132
    bar_width = 30
    offsets = [-bar_width - 8, 0, bar_width + 8]

    legend_x = 120
    for method in method_order:
        color = EVAL_COLORS[method]
        draw.rounded_rectangle((legend_x, 64, legend_x + 24, 78), radius=4, fill=color)
        draw.text((legend_x + 32, 59), method, font=SMALL_FONT, fill=AXIS)
        legend_x += 240

    for cat_idx, category in enumerate(categories):
        center = x_centers[cat_idx]
        for method_idx, method in enumerate(method_order):
            value = values_by_method[method][cat_idx]
            left = center + offsets[method_idx] - bar_width / 2
            top = map_point(0.0, value, 0.0, 1.0, 0.0, y_max, plot_box)[1]
            draw.rounded_rectangle((left, top, left + bar_width, plot_box[3]), radius=8, fill=EVAL_COLORS[method])
            tw, th = text_size(draw, f"{value:.3f}", SMALL_FONT)
            draw.text((left + bar_width / 2 - tw / 2, top - th - 6), f"{value:.3f}", font=SMALL_FONT, fill=AXIS)

        tw, th = text_size(draw, category, LABEL_FONT)
        draw.text((center - tw / 2, plot_box[3] + 16), category, font=LABEL_FONT, fill=AXIS)

    note_box = (width - 360, 120, width - 82, 214)
    draw.rounded_rectangle(note_box, radius=12, fill=NOTE_BG, outline=NOTE_EDGE, width=2)
    lines = [
        f"Avg SRxSim: RePO {values_by_method['RePO'][-1]:.3f}",
        f"v15.1 {values_by_method['v15.1 Retrain'][-1]:.3f}",
        f"v16 {values_by_method['v16'][-1]:.3f}",
    ]
    y = note_box[1] + 14
    for line in lines:
        draw.text((note_box[0] + 14, y), line, font=LABEL_FONT, fill=AXIS)
        y += 24

    image.save(OUT_DIR / "fig_v15_1_v16_eval_srxsim.png")


def write_manifest() -> None:
    with (OUT_DIR / "README.md").open("w") as handle:
        handle.write("# AdaRePO v15.1/v16 vs RePO - DSI Fair Comparison\n\n")
        handle.write("Generated from local scratch training logs. Curves aggregate duplicate log rows by epoch and apply EMA smoothing for presentation readability.\n\n")
        handle.write("## Sources\n\n")
        for method, paths in RUNS.items():
            joined = ", ".join(f"`{path}`" for path in paths)
            handle.write(f"- {method}: {joined}\n")
        handle.write("\n## Eval Sources\n\n")
        for method, path in EVAL_RUNS.items():
            handle.write(f"- {method}: `{path}`\n")
        handle.write("\n## Generated Files\n\n")
        handle.write("- `fig_v15_1_fair_reward_advantage.png`\n")
        handle.write("- `fig_v15_1_fair_rl_diagnostics.png`\n")
        handle.write("- `fig_v15_1_v16_eval_srxsim.png`\n")
        handle.write("- `training_summary.csv`\n")
        handle.write("- `table_v15_1_v16_eval_comparison.csv`\n")
        handle.write("- `table_v15_1_v16_eval_comparison.md`\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {method: parse_log_dicts(paths) for method, paths in RUNS.items()}
    eval_rows = collect_eval_rows()
    write_summary(rows)
    write_eval_table(eval_rows)
    draw_metric_grid(
        filename="fig_v15_1_fair_reward_advantage.png",
        title="AdaRePO v15.1/v16 vs RePO - DSI Fair Training",
        metric_specs=MAIN_METRICS,
        note_lines=[
            "Shared RL metrics: reward + advantage",
            "Not an eval plot; use for training behavior",
        ],
    )
    draw_metric_grid(
        filename="fig_v15_1_fair_rl_diagnostics.png",
        title="AdaRePO v15.1/v16 vs RePO - RL Diagnostics",
        metric_specs=RL_METRICS,
        note_lines=[
            "Advantage balance, guidance loss, and KL stability",
            "KL is clipped at 2.0 to avoid rare spike domination",
        ],
    )
    plot_eval_srxsim(eval_rows)
    write_manifest()


if __name__ == "__main__":
    main()
