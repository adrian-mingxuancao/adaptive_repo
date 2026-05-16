#!/usr/bin/env python3
"""Lightweight repository sanity checks for AdaRePO/APIAR.

This script intentionally avoids importing the training stack (torch, trl,
vLLM, x_r1). It is safe to run on a login node or a clean CPU-only checkout.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "ada_repo.py",
    "ada_repo_config.py",
    "ada_repo_trainer.py",
    "dynamic_beta.py",
    "memory_bank.py",
    "experience_buffer.py",
    "RESULTS_SUMMARY.md",
    "results/aggregated_results.csv",
    "results/table1_main_srxsim.csv",
    "configs/DSI_v16_adaptive_curriculum.yaml",
    "configs/v18_Polaris_hard.yaml",
]

REQUIRED_DIRS = [
    "analysis",
    "configs",
    "docs",
    "results",
    "scripts",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def check_required_paths(errors: list[str]) -> None:
    for item in REQUIRED_FILES:
        path = ROOT / item
        if not path.is_file():
            errors.append(f"missing required file: {item}")
    for item in REQUIRED_DIRS:
        path = ROOT / item
        if not path.is_dir():
            errors.append(f"missing required directory: {item}")


def check_no_generated_junk(errors: list[str]) -> None:
    junk_patterns = ["*.pyc", "*.pyo", "*.log", "slurm-*.out"]
    for pattern in junk_patterns:
        for path in ROOT.rglob(pattern):
            if ".git" not in path.parts:
                errors.append(f"generated file should not be tracked/present: {rel(path)}")
    for path in ROOT.rglob("__pycache__"):
        if ".git" not in path.parts:
            errors.append(f"cache directory should not be present: {rel(path)}")


def check_python_syntax(errors: list[str]) -> None:
    for path in sorted(ROOT.rglob("*.py")):
        if ".git" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"syntax error in {rel(path)}:{exc.lineno}: {exc.msg}")


def check_csv_files(errors: list[str]) -> None:
    for path in sorted((ROOT / "results").glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if not rows:
            errors.append(f"empty CSV: {rel(path)}")
            continue
        if not rows[0]:
            errors.append(f"CSV has empty header: {rel(path)}")


def check_yaml_shape(errors: list[str]) -> None:
    yaml_files = sorted((ROOT / "configs").glob("*.yaml"))
    if not yaml_files:
        errors.append("no YAML configs found under configs/")
        return
    required_any = {"model_name_or_path", "reward_funcs", "output_dir"}
    for path in yaml_files:
        text = path.read_text(encoding="utf-8")
        keys = {
            match.group(1)
            for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*):", text, flags=re.MULTILINE)
        }
        if not (keys & required_any):
            errors.append(f"config has no expected top-level training keys: {rel(path)}")


def check_standalone_notes(errors: list[str], warnings: list[str]) -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8") if (ROOT / "README.md").exists() else ""
    required_phrases = [
        "RePO",
        "PYTHONPATH",
        "data/OpenMolIns",
        "scripts/sanity_check.py",
        "results/table1_main_srxsim.csv",
    ]
    for phrase in required_phrases:
        if phrase not in readme:
            errors.append(f"README missing reproducibility note: {phrase}")

    hardcoded = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_dir() or ".git" in path.parts:
            continue
        if path.suffix not in {".py", ".sh", ".pbs", ".md", ".yaml"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in ["/lus/eagle/", "/net/scratch", "/home/caom/"]):
            hardcoded.append(rel(path))
    if hardcoded:
        warnings.append(
            "cluster/local absolute paths remain in historical scripts/configs; "
            "README documents which variables to update before rerunning."
        )


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    check_required_paths(errors)
    check_no_generated_junk(errors)
    check_python_syntax(errors)
    check_csv_files(errors)
    check_yaml_shape(errors)
    check_standalone_notes(errors, warnings)

    print(f"Sanity check root: {ROOT}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: required files, syntax, CSVs, config shape, and README notes passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
