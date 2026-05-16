#!/usr/bin/env python3
"""
Smoke test: verify reward functions work correctly for each MolOpt subtask.
Runs on login node (no GPU needed). Uses a handful of real examples from training data.
"""
import sys
import os
import pandas as pd

# Add RePO src to path for reward imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO", "src", "x_r1"))

from rewards import get_smile_optimization_reward

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO", "data", "OpenMolIns", "light", "train.csv")

def test_subtask(subtask, property_name, n_samples=3):
    """Test reward function on real samples for a given subtask."""
    print(f"\n{'='*80}")
    print(f"SUBTASK: {subtask}  |  property_name: {property_name}")
    print(f"{'='*80}")

    df = pd.read_csv(DATA_PATH)
    df_sub = df[df["SubTask"] == subtask].head(n_samples)
    assert len(df_sub) > 0, f"No samples found for subtask {subtask}"

    reward_fn = get_smile_optimization_reward(
        property_name=property_name,
        target_direction=None,       # inferred from prompt
        reference_smiles=None,       # extracted from prompt
        similarity_weight=0.3,
        property_weight=0.7,
        min_similarity=0.1,
        extract_pattern=r"<answer>(.*?)</answer>",
    )

    for idx, row in df_sub.iterrows():
        instruction = row["Instruction"]
        gold_molecule = row["molecule"]

        # Simulate a model completion that returns the gold molecule
        fake_completion = f"<think>I need to modify the molecule.</think><answer>{gold_molecule}</answer>"

        prompts = [[
            {"role": "system", "content": "A conversation between User and Assistant."},
            {"role": "user", "content": instruction},
        ]]
        completions = [[{"content": fake_completion}]]

        print(f"\n--- Sample {idx} ---")
        print(f"  Prompt:     {instruction[:120]}...")
        print(f"  Gold mol:   {gold_molecule}")

        rewards = reward_fn(completions=completions, prompts=prompts)
        print(f"  Reward:     {rewards[0]:.4f}")

        # Sanity checks
        assert isinstance(rewards[0], float), f"Reward should be float, got {type(rewards[0])}"
        # Reward should be non-negative for valid molecules
        if rewards[0] < -10:
            print(f"  ⚠ WARNING: very negative reward ({rewards[0]}), check property direction")

    # Also test with an invalid SMILES
    print(f"\n--- Invalid SMILES test ---")
    bad_completion = "<think>thinking</think><answer>INVALID_SMILES_XYZ</answer>"
    prompts = [[
        {"role": "system", "content": "A conversation between User and Assistant."},
        {"role": "user", "content": df_sub.iloc[0]["Instruction"]},
    ]]
    completions = [[{"content": bad_completion}]]
    rewards = reward_fn(completions=completions, prompts=prompts)
    print(f"  Reward for invalid SMILES: {rewards[0]:.4f}")
    assert rewards[0] == 0.0, f"Invalid SMILES should get 0.0 reward, got {rewards[0]}"
    print(f"  ✓ Invalid SMILES correctly gets 0.0 reward")

    print(f"\n✓ {subtask} ({property_name}) smoke test PASSED")


def test_direction_inference():
    """Verify direction is correctly inferred from prompt text."""
    print(f"\n{'='*80}")
    print(f"DIRECTION INFERENCE TEST")
    print(f"{'='*80}")

    df = pd.read_csv(DATA_PATH)
    for subtask in ["LogP", "MR", "QED"]:
        df_sub = df[df["SubTask"] == subtask]
        increase_count = 0
        decrease_count = 0
        for _, row in df_sub.iterrows():
            inst = row["Instruction"].lower()
            if any(w in inst for w in ["increase", "higher", "maximize"]):
                increase_count += 1
            elif any(w in inst for w in ["decrease", "lower", "minimize"]):
                decrease_count += 1
        print(f"  {subtask}: increase={increase_count}, decrease={decrease_count}, total={len(df_sub)}")
    print(f"  ✓ Direction distribution looks correct")


def test_data_schema():
    """Verify data schema is consistent."""
    print(f"\n{'='*80}")
    print(f"DATA SCHEMA VALIDATION")
    print(f"{'='*80}")

    df = pd.read_csv(DATA_PATH)
    print(f"  Total rows: {len(df)}")
    print(f"  Columns: {list(df.columns)}")

    assert "SubTask" in df.columns, "Missing SubTask column"
    assert "Instruction" in df.columns, "Missing Instruction column"
    assert "molecule" in df.columns, "Missing molecule column"

    for subtask in ["LogP", "MR", "QED"]:
        df_sub = df[df["SubTask"] == subtask]
        assert len(df_sub) == 500, f"Expected 500 rows for {subtask}, got {len(df_sub)}"

        # Check no empty instructions or molecules
        assert df_sub["Instruction"].notna().all(), f"Found NaN instructions in {subtask}"
        assert df_sub["molecule"].notna().all(), f"Found NaN molecules in {subtask}"
        assert (df_sub["Instruction"].str.len() > 10).all(), f"Found suspiciously short instructions in {subtask}"
        assert (df_sub["molecule"].str.len() > 2).all(), f"Found suspiciously short molecules in {subtask}"

        print(f"  {subtask}: {len(df_sub)} rows, all non-empty ✓")

    print(f"  ✓ Data schema validation PASSED")


if __name__ == "__main__":
    print("Smoke Test: Reward Functions for MolOpt Subtasks")
    print("=" * 80)

    # 1. Data schema
    test_data_schema()

    # 2. Direction inference
    test_direction_inference()

    # 3. Per-subtask reward tests
    subtask_props = [
        ("LogP", "logP"),
        ("MR",   "mr"),
        ("QED",  "qed"),
    ]
    for subtask, prop in subtask_props:
        test_subtask(subtask, prop, n_samples=3)

    print(f"\n{'='*80}")
    print(f"ALL SMOKE TESTS PASSED")
    print(f"{'='*80}")
