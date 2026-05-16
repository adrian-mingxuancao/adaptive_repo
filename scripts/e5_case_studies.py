#!/usr/bin/env python3
"""
E5: Qualitative case studies — find examples where APIAR's memory bank
was activated and produced better results than RePO.

Picks 6 representative test examples (2 per subtask) where:
  1. APIAR succeeded and RePO failed, OR
  2. Both succeeded but APIAR has higher similarity

Renders source molecule, RePO output, APIAR output with property deltas.
"""
import csv
import os
import sys
from pathlib import Path
from collections import defaultdict

try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, Descriptors
    from PIL import Image, ImageDraw, ImageFont
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    print("WARNING: RDKit not available. Will output SMILES text only.", file=sys.stderr)

PRED_DIRS = [
    "/lus/eagle/projects/IMPROVE_Aim1/caom/RePO/predictions",
    "/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/evaluation_results",
]

# Use seed 42 for case study (single representative seed)
APIAR_FRAGS = ("v16v17ms_v16_s42", "v16_s42checkpoint-120")
REPO_FRAGS = ("v16v17ms_repo_s42", "repo_s42checkpoint-120")

SUBTASKS = ["LogP", "MR", "QED"]
OUT_DIR = Path("/lus/eagle/projects/IMPROVE_Aim1/caom/agent_drug_discovery/adaptive_repo/analysis/e5_case_studies")


def find_detailed_csv(fragments, subtask):
    for base_dir in PRED_DIRS:
        for frag in fragments:
            candidates = [
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv",
                Path(base_dir) / frag / "open_generation" / "MolOpt" / f"{subtask}.csv",
            ]
            for subdir in Path(base_dir).iterdir() if Path(base_dir).exists() else []:
                if subdir.is_dir():
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}_detailed_results.csv")
                    candidates.append(subdir / frag / "open_generation" / "MolOpt" / f"{subtask}.csv")
            for c in candidates:
                if c.exists():
                    return c
    return None


def load_results(csv_path):
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def find_case_studies(subtask, n=2):
    """Find n best case study examples for a subtask."""
    a_path = find_detailed_csv(APIAR_FRAGS, subtask)
    r_path = find_detailed_csv(REPO_FRAGS, subtask)
    
    if a_path is None or r_path is None:
        print(f"  WARNING: missing data for {subtask}")
        return []
    
    a_rows = load_results(a_path)
    r_rows = load_results(r_path)
    
    candidates = []
    for i, (a, r) in enumerate(zip(a_rows, r_rows)):
        a_succ = int(float(a.get("success", 0)))
        r_succ = int(float(r.get("success", 0)))
        a_sim = float(a.get("similarity", 0))
        r_sim = float(r.get("similarity", 0))
        a_valid = a.get("validity", "False") == "True"
        r_valid = r.get("validity", "False") == "True"
        
        # Case 1: APIAR succeeds, RePO fails
        if a_succ == 1 and r_succ == 0 and a_sim > 0.5:
            score = a_sim  # prefer higher similarity cases
            candidates.append((score, "apiar_wins", i, a, r))
        # Case 2: Both succeed, APIAR has higher sim (better quality)
        elif a_succ == 1 and r_succ == 1 and a_sim > r_sim + 0.05:
            score = a_sim - r_sim
            candidates.append((score, "both_succeed_apiar_better", i, a, r))
    
    # Sort by score descending, pick top n
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:n]


def render_molecule_grid(cases, subtask, out_path):
    """Render a grid with source, RePO, and APIAR plus evaluation metadata."""
    if not HAS_RDKIT:
        return

    cell_w, cell_h = 430, 350
    mol_h = 235
    margin = 16
    canvas = Image.new("RGB", (cell_w * 3, cell_h * len(cases)), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for row_num, (_, case_type, idx, a_row, r_row) in enumerate(cases):
        src_smi = a_row.get("original_molecule", "")
        a_smi = a_row.get("generated_molecule", "")
        r_smi = r_row.get("generated_molecule", "")

        cells = [
            ("Source", src_smi, a_row, None),
            ("RePO", r_smi, r_row, r_row),
            ("APIAR", a_smi, a_row, a_row),
        ]
        for col_num, (name, smi, prop_row, eval_row) in enumerate(cells):
            x0 = col_num * cell_w
            y0 = row_num * cell_h
            draw.rectangle([x0, y0, x0 + cell_w, y0 + cell_h], fill="white")

            mol = Chem.MolFromSmiles(smi) if smi else None
            if mol is not None:
                mol_img = Draw.MolToImage(mol, size=(cell_w - 2 * margin, mol_h))
                canvas.paste(mol_img, (x0 + margin, y0 + 8))
            else:
                draw.text((x0 + margin, y0 + 95), "Invalid SMILES", fill="#9a3412", font=font)

            lines = build_legend_lines(name, idx, prop_row, eval_row)
            text_y = y0 + mol_h + 10
            for line in lines:
                draw.text((x0 + margin, text_y), line, fill="#111111", font=small_font)
                text_y += 19

    canvas.save(str(out_path))
    print(f"  Saved molecule grid: {out_path}")


def build_legend_lines(name, idx, prop_row, eval_row):
    prop_text = format_property(prop_row, source=(eval_row is None))
    if eval_row is None:
        return [f"{name} idx={idx}", prop_text]

    sim = float(eval_row.get("similarity", 0) or 0)
    success = int(float(eval_row.get("success", 0) or 0))
    valid = eval_row.get("validity", "False") == "True"
    marker = "OK" if success else "FAIL"
    valid_text = "valid" if valid else "invalid"
    return [f"{name} | {marker} | {valid_text}", f"sim={sim:.2f} | {prop_text}"]


def format_property(row, source=False):
    prop_names = [
        ("logP", "LogP"),
        ("MR", "MR"),
        ("qed", "QED"),
        ("QED", "QED"),
    ]
    for key, label in prop_names:
        value_key = f"original_{key}" if source else f"generated_{key}"
        if value_key in row and row[value_key] not in ("", None):
            value = float(row[value_key])
            if source:
                return f"{label}={value:.3f}"
            change_key = f"{key}_change"
            if change_key in row and row[change_key] not in ("", None):
                change = float(row[change_key])
                return f"{label}={value:.3f} (d={change:+.3f})"
            return f"{label}={value:.3f}"
    return "property=N/A"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_cases = {}
    
    print("E5: Finding qualitative case studies...")
    for subtask in SUBTASKS:
        cases = find_case_studies(subtask, n=2)
        all_cases[subtask] = cases
        
        print(f"\n  {subtask}: found {len(cases)} case studies")
        for score, case_type, idx, a_row, r_row in cases:
            src = a_row.get("original_molecule", "")[:50]
            a_smi = a_row.get("generated_molecule", "")[:50]
            r_smi = r_row.get("generated_molecule", "")[:50]
            a_sim = float(a_row.get("similarity", 0))
            r_sim = float(r_row.get("similarity", 0))
            a_succ = int(float(a_row.get("success", 0)))
            r_succ = int(float(r_row.get("success", 0)))
            print(f"    [{case_type}] idx={idx}: APIAR(sim={a_sim:.3f},SR={a_succ}) vs RePO(sim={r_sim:.3f},SR={r_succ})")
            print(f"      src:   {src}")
            print(f"      apiar: {a_smi}")
            print(f"      repo:  {r_smi}")
    
    # Save text summary
    summary_path = OUT_DIR / "case_studies_summary.txt"
    with open(summary_path, "w") as f:
        for subtask, cases in all_cases.items():
            f.write(f"\n{'='*60}\n{subtask}\n{'='*60}\n")
            for score, case_type, idx, a_row, r_row in cases:
                f.write(f"\nCase: {case_type} (index={idx})\n")
                f.write(f"  Source:     {a_row.get('original_molecule', '')}\n")
                f.write(f"  APIAR gen:  {a_row.get('generated_molecule', '')}\n")
                f.write(f"  RePO gen:   {r_row.get('generated_molecule', '')}\n")
                f.write(f"  Instruction: {a_row.get('instruction', '')}\n")
                f.write(f"  APIAR: valid={a_row.get('validity')}, success={a_row.get('success')}, sim={a_row.get('similarity')}\n")
                f.write(f"  RePO:  valid={r_row.get('validity')}, success={r_row.get('success')}, sim={r_row.get('similarity')}\n")
                
                # Property-specific info
                for k in a_row:
                    if k.startswith("original_") or k.endswith("_change") or k.startswith("generated_"):
                        if k not in ("original_molecule", "generated_molecule"):
                            f.write(f"  APIAR {k}={a_row[k]}\n")
                            f.write(f"  RePO  {k}={r_row[k]}\n")
    print(f"\n  Summary saved: {summary_path}")
    
    # Render molecules if RDKit available
    if HAS_RDKIT:
        for subtask, cases in all_cases.items():
            if cases:
                render_molecule_grid(cases, subtask, OUT_DIR / f"case_studies_{subtask}.png")
        
        # Combined figure
        all_flat = []
        for subtask in SUBTASKS:
            all_flat.extend(all_cases.get(subtask, []))
        if all_flat:
            render_molecule_grid(all_flat, "all", OUT_DIR / "case_studies_all.png")


if __name__ == "__main__":
    main()
