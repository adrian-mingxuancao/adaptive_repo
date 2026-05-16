"""
AdaRePO Entry Point — Adaptive Reference-guided Policy Optimization.

Mirrors the structure of RePO/src/x_r1/repo.py but uses AdaRePOTrainer
with dynamic beta and optional self-distillation.

Usage:
    bash scripts/run_RL_training.sh \
      --gpus 0,1,2 \
      --num_processes 2 \
      --entry agent_drug_discovery/adaptive_repo/ada_repo.py \
      --config agent_drug_discovery/adaptive_repo/configs/ada_repo_3B_config.yaml \
      --output_dir ./output/ada_repo_run
"""
import json
import logging
import os
import sys

import datasets
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

# Add this directory to sys.path so sibling modules are importable as a script
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

# Add RePO source to path for shared utilities
REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO", "src")
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from x_r1.rewards import (
    accuracy_reward,
    format_reward,
    get_cosine_scaled_reward,
    get_molecular_structure_reward,
    get_repetition_penalty_reward,
    get_smile_optimization_reward,
    get_smile_similarity_reward,
    get_smile_validity_reward,
    len_reward,
    reasoning_steps_reward,
)
from x_r1.utils.callbacks import get_callbacks

from ada_repo_config import AdaRePOConfig
from ada_repo_trainer import AdaRePOTrainer


logger = logging.getLogger(__name__)


class CleanOptimizerStatesCallback(transformers.TrainerCallback):
    """Proactively clean DeepSpeed ZeRO-3 optimizer states before new saves.

    Strategy: on_step_end (before save), clean ALL existing optimizer states
    so the upcoming checkpoint write won't cause disk overflow. The newly
    saved checkpoint will be the only one with optimizer states, enabling
    resume. Disk budget: 1×full_ckpt(~41G) + N×light_ckpts(~6G each)."""
    import shutil as _shutil

    def _clean_all_optim_states(self, output_dir, reason=""):
        import glob
        for ckpt_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
            for gs_dir in glob.glob(os.path.join(ckpt_dir, "global_step*")):
                if os.path.isdir(gs_dir):
                    self._shutil.rmtree(gs_dir, ignore_errors=True)
                    logger.info(f"Cleaned optimizer states ({reason}): {gs_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        # If trainer is about to save, pre-clean old optimizer states
        if control.should_save:
            self._clean_all_optim_states(args.output_dir, reason="pre-save")
        return control

# System prompt (identical to RePO)
SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def init_wandb_training(training_args):
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project


from dataclasses import dataclass, field


@dataclass
class AdaRePOScriptArguments(ScriptArguments):
    variant: str = field(
        default="default",
        metadata={"help": "Entry variant (default|mumo)"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions."},
    )
    cosine_min_value_wrong: float = field(default=0.0)
    cosine_max_value_wrong: float = field(default=-0.5)
    cosine_min_value_correct: float = field(default=0.5)
    cosine_max_value_correct: float = field(default=1.0)
    cosine_max_len: int = field(default=1000)
    repetition_n_grams: int = field(default=3)
    repetition_max_penalty: float = field(default=-1.0)
    length_min_tokens: int = field(default=100)
    length_target_tokens: int = field(default=500)
    length_max_reward: float = field(default=0.8)
    length_reward_curve: str = field(default="linear")
    data_scale: str = field(default="light")
    train_data_file: str = field(default="train.csv")
    subtask_selection: list[str] = field(
        default_factory=lambda: ["LogP", "MR", "QED"],
    )
    property_name: str = field(
        default="logP",
        metadata={"help": "Molecular property for smile_optimization reward. Options: logP, qed, mr, tpsa"},
    )
    similarity_weight: float = field(
        default=0.3,
        metadata={"help": "Weight for similarity component in smile_optimization reward"},
    )
    property_weight: float = field(
        default=0.7,
        metadata={"help": "Weight for property improvement component in smile_optimization reward"},
    )
    min_similarity: float = field(
        default=0.1,
        metadata={"help": "Minimum similarity threshold for a valid reward"},
    )


def main(script_args, training_args, model_args, variant: str = "default"):
    variant = (variant or "default").strip().lower()
    set_seed(training_args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}, "
        f"16-bits: {training_args.fp16}"
    )

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # ---- Dataset ----
    dataset = None
    _molopt_subtasks = {"LogP", "MR", "QED"}
    _struct_subtasks = {"AddComponent", "SubComponent", "DelComponent"}

    if isinstance(script_args.subtask_selection, list) and set(script_args.subtask_selection).issubset(_struct_subtasks):
        # Structural optimization dataset (JSON)
        struct_json = "data/structural_opt_light.json"
        _REPO_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO")
        for base in [_REPO_BASE, "."]:
            full_path = os.path.join(base, struct_json)
            if os.path.exists(full_path):
                struct_json = full_path
                break
        data = json.load(open(struct_json, "r"))
        df = pd.DataFrame(data)
        # Filter to selected structure subtasks if not all three
        selected = set(script_args.subtask_selection)
        if selected != _struct_subtasks:
            def _classify_struct(row):
                has_added = bool(row.get('added_group', ''))
                has_removed = bool(row.get('removed_group', ''))
                if has_added and not has_removed:
                    return 'AddComponent'
                elif has_removed and not has_added:
                    return 'DelComponent'
                else:
                    return 'SubComponent'
            df['_subtask'] = df.apply(_classify_struct, axis=1)
            df = df[df['_subtask'].isin(selected)]
            df = df.drop(columns=['_subtask'])
            logger.info(f"Filtered structural dataset to {selected}: {len(df)} examples")
        ds = Dataset.from_pandas(df)
        dataset = DatasetDict({"train": ds})
        dataset = dataset.rename_column("instruction", "problem")
        dataset = dataset.rename_column("output", "solution")
        logger.info(f"Loaded structural dataset: {len(dataset['train'])} examples from {struct_json}")
    elif isinstance(script_args.subtask_selection, list) and set(script_args.subtask_selection).issubset(_molopt_subtasks):
        train_file = getattr(script_args, 'train_data_file', 'train.csv')
        dataset_path = f"data/OpenMolIns/{script_args.data_scale}/{train_file}"
        # Try relative to RePO dir first
        _REPO_BASE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO")
        for base in [_REPO_BASE, "."]:
            full_path = os.path.join(base, dataset_path)
            if os.path.exists(full_path):
                dataset_path = full_path
                break

        df = pd.read_csv(dataset_path)
        if isinstance(script_args.subtask_selection, list):
            df = df[df["SubTask"].isin(script_args.subtask_selection)]
        train_dataset = Dataset.from_pandas(df)
        dataset = DatasetDict({"train": train_dataset})
        dataset = dataset.rename_column("Instruction", "problem")
        dataset = dataset.rename_column("molecule", "solution")
        logger.info(f"Loaded OpenMolIns dataset: {len(dataset['train'])} examples")
    else:
        dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    system_prompt = SYSTEM_PROMPT

    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": example["problem"]},
            ],
            "solution": example["solution"] if "solution" in example else None,
        }

    dataset = dataset.map(make_conversation)

    # ---- Reward functions ----
    REWARD_FUNCS_REGISTRY = {
        "accuracy": lambda prompts, completions, solutions=None, **kw: accuracy_reward(
            completions=completions, solution=solutions
        )
        if solutions is not None
        else [0.0] * len(completions),
        "format": lambda prompts, completions, **kw: format_reward(completions=completions),
        "reasoning_steps": lambda prompts, completions, **kw: reasoning_steps_reward(completions=completions),
        "cosine": lambda prompts, completions, solutions=None, **kw: get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        )(completions=completions, solution=solutions if solutions is not None else [""] * len(completions)),
        "repetition_penalty": lambda prompts, completions, **kw: get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams, max_penalty=script_args.repetition_max_penalty
        )(completions=completions),
        "length": lambda prompts, completions, solutions=None, **kw: len_reward(
            prompts=prompts,
            completions=completions,
            solutions=solutions,
            min_tokens=script_args.length_min_tokens,
            target_tokens=script_args.length_target_tokens,
            max_reward=script_args.length_max_reward,
            reward_curve=script_args.length_reward_curve,
        ),
        "smile_validity": lambda prompts, completions, **kw: get_smile_validity_reward(
            extract_pattern=r"<answer>(.*?)</answer>", validity_weight=1.0
        )(completions=completions),
        "smile_similarity": lambda prompts, completions, **kw: get_smile_similarity_reward(
            extract_pattern=r"<answer>(.*?)</answer>"
        )(completions=completions),
        "smile_optimization": lambda prompts, completions, **kw: get_smile_optimization_reward(
            property_name=script_args.property_name,
            target_direction=None,
            reference_smiles=None,
            similarity_weight=script_args.similarity_weight,
            property_weight=script_args.property_weight,
            min_similarity=script_args.min_similarity,
            extract_pattern=r"<answer>(.*?)</answer>",
        )(completions=completions, prompts=prompts),
        "structure_optimization": lambda prompts, completions, **kw: get_molecular_structure_reward(
            extract_pattern=r"<answer>(.*?)</answer>"
        )(completions=[[{"role": "assistant", "content": c}] if isinstance(c, str) else c for c in completions], prompts=prompts, **kw),
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    reward_func_names = script_args.reward_funcs

    if hasattr(training_args, "reward_weights") and isinstance(training_args.reward_weights, dict):
        reward_weights = [training_args.reward_weights[func] for func in script_args.reward_funcs]
        training_args.reward_weights = reward_weights

    # ---- Model kwargs ----
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    training_args.gradient_checkpointing = False
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    training_args.model_init_kwargs = model_kwargs

    # ---- Initialize AdaRePO Trainer ----
    logger.info("*** Initializing AdaRePO Trainer ***")
    logger.info(
        f"  beta_guide_mode: {training_args.beta_guide_mode}, "
        f"beta_guide_max: {getattr(training_args, 'beta_guide_max', 1.5)}, "
        f"beta_guide_min: {getattr(training_args, 'beta_guide_min', 0.3)}, "
        f"beta_guide_alpha: {getattr(training_args, 'beta_guide_alpha', 3.0)}"
    )
    logger.info(
        f"  use_memory_bank: {training_args.use_memory_bank}, "
        f"memory_bank_size: {training_args.memory_bank_size}"
    )

    trainer = AdaRePOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        reward_func_names=reward_func_names,
        args=training_args,
        variant=variant,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args) + [CleanOptimizerStatesCallback()],
    )

    # ---- Train ----
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # ---- Save ----
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    kwargs = {"dataset_name": script_args.dataset_name, "tags": ["AdaRePO"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


def run(variant: str = "default"):
    parser = TrlParser((AdaRePOScriptArguments, AdaRePOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if variant == "default":
        variant = getattr(script_args, "variant", "default")
    main(script_args, training_args, model_args, variant=variant)


if __name__ == "__main__":
    run("default")
