"""
AdaRePO Trainer — Adaptive Reference-guided Policy Optimization.

Subclasses the RePO XGRPOTrainer to add:
  1. Dynamic beta_guide for the answer-level guidance loss
  2. Memory bank for self-distillation (active reference selection)
  3. (Optional) Confidence-aware beta via ensemble reward predictors

The key change is in compute_loss:
    Original RePO:   loss = L_RL + s_loss               (beta_guide = 1.0 implicit)
    AdaRePO:         loss = L_RL + beta_guide * s_loss   (beta_guide adaptive)
"""
import hashlib
import logging
import os
import re
import sys
from collections import defaultdict
from typing import Any, Optional, Union

import torch
import torch.nn as nn

try:
    from .dynamic_beta import DynamicBetaController
    from .memory_bank import MoleculeMemoryBank
    from .experience_buffer import ExperienceBuffer
except ImportError:
    from dynamic_beta import DynamicBetaController
    from memory_bank import MoleculeMemoryBank
    from experience_buffer import ExperienceBuffer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parent class lives in the RePO repo; add its path for import.
# ---------------------------------------------------------------------------
REPO_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "RePO", "src")
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

from x_r1.x_repo_trainer import XGRPOTrainer, replace_smile, RewardFunc, _disable_peft_adapters

from accelerate.utils import gather, gather_object, broadcast_object_list
from transformers import PreTrainedModel, PreTrainedTokenizerBase, Trainer, TrainerCallback
from trl.data_utils import is_conversational, maybe_apply_chat_template, apply_chat_template
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import pad, selective_log_softmax
from accelerate.utils.other import is_compiled_module
from accelerate.utils import is_peft_model
from datasets import Dataset, IterableDataset

try:
    from rdkit import Chem
except ImportError:
    Chem = None


def _extract_answer_smiles(text: str) -> Optional[str]:
    """Extract SMILES from <answer>...</answer> tags."""
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m and Chem is not None:
        smi = m.group(1).strip()
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                return smi
        except Exception:
            pass
    return None


def _eval_reward_on_completions(
    reward_funcs, reward_processing_classes, reward_weights,
    prompts_text, completions, inputs, device,
):
    """
    Evaluate the full reward pipeline on a list of completions.
    Returns a (B,) tensor of weighted rewards.
    """
    B = len(completions)
    rewards = torch.zeros(B, device=device)
    for i, (reward_func, rpc) in enumerate(zip(reward_funcs, reward_processing_classes)):
        if isinstance(reward_func, nn.Module):
            continue  # skip model-based reward for v_ref (too expensive)
        keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
        if "solution" in reward_kwargs:
            solutions_for_reward = reward_kwargs.pop("solution")
            if "molecule" in reward_kwargs:
                original_mol = reward_kwargs.pop("molecule")
                removed_group = reward_kwargs.pop("removed_group")
                added_group = reward_kwargs.pop("added_group")
                output = reward_func(
                    prompts=prompts_text, completions=completions,
                    target_mol=solutions_for_reward, original_mol=original_mol,
                    removed_group=removed_group, added_group=added_group,
                    **reward_kwargs,
                )
            else:
                output = reward_func(
                    prompts=prompts_text, completions=completions,
                    solution=solutions_for_reward, **reward_kwargs,
                )
        else:
            output = reward_func(prompts=prompts_text, completions=completions, **reward_kwargs)
        w = reward_weights[i].item() if reward_weights is not None else 1.0
        rewards += w * torch.tensor(output, dtype=torch.float32, device=device)
    return rewards


def _append_advantage_metrics(metrics_store, advantages, prefix="advantage"):
    if advantages is None or not isinstance(advantages, torch.Tensor) or advantages.numel() == 0:
        return

    flat = advantages.detach().float().reshape(-1)
    metrics_store[f"{prefix}/mean"].append(flat.mean().item())
    metrics_store[f"{prefix}/std"].append(flat.std(unbiased=False).item())
    metrics_store[f"{prefix}/min"].append(flat.min().item())
    metrics_store[f"{prefix}/max"].append(flat.max().item())
    metrics_store[f"{prefix}/abs_mean"].append(flat.abs().mean().item())
    metrics_store[f"{prefix}/p10"].append(torch.quantile(flat, 0.10).item())
    metrics_store[f"{prefix}/p50"].append(torch.quantile(flat, 0.50).item())
    metrics_store[f"{prefix}/p90"].append(torch.quantile(flat, 0.90).item())
    metrics_store[f"{prefix}/pos_frac"].append((flat > 0).float().mean().item())
    metrics_store[f"{prefix}/neg_frac"].append((flat < 0).float().mean().item())
    metrics_store[f"{prefix}/zero_frac"].append((flat.abs() < 1e-8).float().mean().item())


class AdaRePOTrainer(XGRPOTrainer):
    """
    Adaptive RePO trainer with dynamic beta and optional self-distillation.

    Additional args (via AdaRePOConfig):
        beta_guide_max, beta_guide_alpha, beta_guide_top_k_frac,
        beta_guide_mode, beta_guide_softmax_tau,
        use_memory_bank, memory_bank_size, promotion_margin,
        promotion_sim_min, memory_bank_max_age
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes=None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers=(None, None),
        peft_config=None,
        reward_func_names: Optional[list[str]] = None,
        variant: str = "default",
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            reward_func_names=reward_func_names,
            variant=variant,
        )

        # --- Dynamic Beta Controller ---
        self.beta_controller = DynamicBetaController(
            mode=getattr(args, "beta_guide_mode", "sigmoid_gap"),
            beta_max=getattr(args, "beta_guide_max", 1.5),
            beta_min=getattr(args, "beta_guide_min", 0.3),
            alpha=getattr(args, "beta_guide_alpha", 3.0),
            top_k_frac=getattr(args, "beta_guide_top_k_frac", 0.33),
            softmax_tau=getattr(args, "beta_guide_softmax_tau", 1.0),
            confidence_threshold=getattr(args, "confidence_threshold", 0.5),
        )

        # --- Memory Bank (Self-Distillation) ---
        self.use_memory_bank = getattr(args, "use_memory_bank", False)
        if self.use_memory_bank:
            self.memory_bank = MoleculeMemoryBank(
                max_size_per_query=getattr(args, "memory_bank_size", 5),
                promotion_margin=getattr(args, "promotion_margin", 0.1),
                similarity_min=getattr(args, "promotion_sim_min", 0.3),
                max_age=getattr(args, "memory_bank_max_age", 1000),
            )
        else:
            self.memory_bank = None

        # --- Priority Learning (curriculum-style sample weighting) ---
        self.use_priority_weighting = getattr(args, "use_priority_weighting", False)
        self.priority_variance_scale = getattr(args, "priority_variance_scale", 2.0)
        self.priority_frontier_center = getattr(args, "priority_frontier_center", 0.3)
        self.priority_frontier_width = getattr(args, "priority_frontier_width", 0.3)
        self.priority_min_weight = getattr(args, "priority_min_weight", 0.2)

        # --- Experience Buffer (Stable Example Replay) ---
        self.use_experience_buffer = getattr(args, "use_experience_buffer", False)
        if self.use_experience_buffer:
            self.experience_buffer = ExperienceBuffer(
                max_size=getattr(args, "exp_buffer_max_size", 256),
                max_age=getattr(args, "exp_buffer_max_age", 100),
                max_replay_per_batch=getattr(args, "exp_buffer_max_replay_per_batch", 2),
                sigma_threshold=getattr(args, "exp_buffer_sigma_threshold", 0.05),
                mu_threshold=getattr(args, "exp_buffer_mu_threshold", 0.4),
            )
        else:
            self.experience_buffer = None

        # --- Adaptive Temperature (Anti-Collapse) ---
        self.use_adaptive_temperature = getattr(args, "use_adaptive_temperature", False)
        self.adaptive_temp_base = getattr(args, "adaptive_temp_base", 0.9)
        self.adaptive_temp_high = getattr(args, "adaptive_temp_high", 1.3)
        self.adaptive_temp_sigma_threshold = getattr(args, "adaptive_temp_sigma_threshold", 0.05)
        self.adaptive_temp_ema_alpha = getattr(args, "adaptive_temp_ema_alpha", 0.3)
        self._prompt_sigma_ema: dict[str, float] = {}  # prompt_hash -> EMA sigma

        # Import SamplingParams if adaptive temperature is used with vLLM
        if self.use_adaptive_temperature and getattr(args, "use_vllm", False):
            try:
                from vllm import SamplingParams as _SP
                self._SamplingParams = _SP
            except ImportError:
                self._SamplingParams = None
                logger.warning("vLLM not available; adaptive temperature disabled.")
                self.use_adaptive_temperature = False

        logger.info(
            f"AdaRePO initialized: mode={self.beta_controller.mode}, "
            f"beta_max={self.beta_controller.beta_max}, "
            f"alpha={self.beta_controller.alpha}, "
            f"memory_bank={self.use_memory_bank}, "
            f"priority_weighting={self.use_priority_weighting}, "
            f"adaptive_temp={self.use_adaptive_temperature}, "
            f"experience_buffer={self.use_experience_buffer}"
        )

    # ------------------------------------------------------------------
    # Override _prepare_inputs: identical to parent but we intercept raw
    # rewards and add beta_guide + memory bank logic after reward computation.
    # ------------------------------------------------------------------

    def _prepare_inputs(self, inputs):
        # Dict inputs (e.g. from Trainer._prepare_inputs) pass through
        if isinstance(inputs, dict) and "input_ids" in inputs:
            return Trainer._prepare_inputs(self, inputs=inputs)

        device = self.accelerator.device

        # ---------- Prompt encoding (same as parent) ----------
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(ex, self.processing_class)["prompt"] for ex in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, inputs=prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # ---------- Generation (with optional adaptive temperature) ----------
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                if self.use_adaptive_temperature and self._SamplingParams is not None:
                    # Per-prompt temperature: high T for collapsed prompts
                    per_prompt_params = []
                    n_high_temp = 0
                    for pt in all_prompts_text:
                        ph = hashlib.md5(pt.encode()).hexdigest()
                        ema_sigma = self._prompt_sigma_ema.get(ph, None)
                        if ema_sigma is not None and ema_sigma < self.adaptive_temp_sigma_threshold:
                            temp = self.adaptive_temp_high
                            n_high_temp += 1
                        else:
                            temp = self.adaptive_temp_base
                        per_prompt_params.append(self._SamplingParams(
                            temperature=temp,
                            max_tokens=self.max_completion_length,
                        ))
                    outputs = self.llm.generate(
                        all_prompts_text, sampling_params=per_prompt_params, use_tqdm=False
                    )
                else:
                    outputs = self.llm.generate(
                        all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False
                    )
                    n_high_temp = 0
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
                n_high_temp = 0
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            prompt_ids = prompt_ids.to(device)
            prompt_mask = prompt_mask.to(device)
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # EOS masking
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # ---------- Ref log-probs (same as parent) ----------
        logits_to_keep = completion_ids.size(1)
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with _disable_peft_adapters(self.accelerator.unwrap_model(self.model)):
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # ---------- Decode completions ----------
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text

        solutions = [str(example.get("solution", "")) for example in inputs]

        # ---------- Reward computation (same as parent) ----------
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = Trainer._prepare_inputs(self, inputs=reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                if "solution" in reward_kwargs:
                    solutions_for_reward = reward_kwargs.pop("solution")
                    if "molecule" in reward_kwargs:
                        original_mol = reward_kwargs.pop("molecule")
                        removed_group = reward_kwargs.pop("removed_group")
                        added_group = reward_kwargs.pop("added_group")
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions,
                            target_mol=solutions_for_reward, original_mol=original_mol,
                            removed_group=removed_group, added_group=added_group,
                            **reward_kwargs,
                        )
                    else:
                        output_reward_func = reward_func(
                            prompts=prompts, completions=completions,
                            solution=solutions_for_reward, **reward_kwargs,
                        )
                else:
                    output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Local raw rewards (before gather)
        raw_rewards_local = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        rewards_per_func = gather(rewards_per_func)
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # GRPO group-normalized advantages
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        advantage_stats = advantages

        # ============================================================
        # AdaRePO EXTENSION: Update EMA sigma tracker (for adaptive temp)
        # Uses global rewards to compute per-query sigma, then updates EMA.
        # ============================================================
        if self.use_adaptive_temperature:
            G = self.num_generations
            grouped_for_ema = rewards.view(-1, G)  # (num_queries, G)
            sigma_per_query = grouped_for_ema.std(dim=1)  # (num_queries,)
            # Reconstruct all_prompts_text on non-main processes
            all_pt = gather_object(prompts_text)
            # all_pt has B_global prompts, but rewards has B_global*G entries
            # Each prompt appears G times consecutively
            n_queries = sigma_per_query.shape[0]
            # Deduplicate: all_pt has B_global entries, sigma_per_query has B_global entries
            alpha = self.adaptive_temp_ema_alpha
            n_collapsed = 0
            for qi in range(n_queries):
                ph = hashlib.md5(all_pt[qi].encode()).hexdigest()
                new_sigma = sigma_per_query[qi].item()
                if ph in self._prompt_sigma_ema:
                    self._prompt_sigma_ema[ph] = alpha * new_sigma + (1 - alpha) * self._prompt_sigma_ema[ph]
                else:
                    self._prompt_sigma_ema[ph] = new_sigma
                if self._prompt_sigma_ema[ph] < self.adaptive_temp_sigma_threshold:
                    n_collapsed += 1
            self._metrics["adaptive_temp/n_collapsed"].append(n_collapsed)
            self._metrics["adaptive_temp/frac_collapsed"].append(n_collapsed / max(n_queries, 1))
            self._metrics["adaptive_temp/ema_sigma_mean"].append(
                sum(self._prompt_sigma_ema.values()) / max(len(self._prompt_sigma_ema), 1)
            )

        # ============================================================
        # AdaRePO EXTENSION: compute v_ref and dynamic beta
        # Beta must be computed on GLOBAL rewards (multiple of G),
        # then sliced to local — same pattern as advantages.
        # ============================================================

        # Compute v_ref locally: evaluate reward on reference molecule
        ref_completions = [
            f"<think>Reference molecule.</think><answer>{sol}</answer>"
            for sol in solutions
        ]
        v_ref_local = _eval_reward_on_completions(
            self.reward_funcs, self.reward_processing_classes,
            self.reward_weights, prompts_text, ref_completions,
            inputs, device,
        )

        # Gather v_ref across ranks so it aligns with global rewards
        v_ref_global = gather(v_ref_local)  # (total_samples,)

        # Compute dynamic beta on GLOBAL rewards and v_ref (both multiples of G)
        beta_guide_global = self.beta_controller.compute(
            rewards=rewards,       # global gathered rewards, shape (N*G,)
            v_ref=v_ref_global,    # global gathered v_ref, shape (N*G,)
            num_generations=self.num_generations,
        )

        # Slice everything to local
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        beta_guide = beta_guide_global[process_slice]
        raw_rewards_local_gathered = rewards[process_slice]

        B_local = advantages.shape[0]

        # ============================================================
        # AdaRePO EXTENSION: Priority Learning (sample weighting)
        # Weight each prompt group by informativeness:
        #   - variance component: high reward variance → uncertain → learn more
        #   - frontier component: reward mean near frontier_center → learning edge
        # Priority weights multiply the advantages so the optimizer
        # focuses gradient on the most informative prompts.
        # ============================================================
        if self.use_priority_weighting:
            G = self.num_generations
            # Use global rewards for computing per-query stats
            grouped = rewards.view(-1, G)  # (num_queries, G)
            query_mean = grouped.mean(dim=1)   # (num_queries,)
            query_std = grouped.std(dim=1)     # (num_queries,)

            # Variance component: sigmoid(scale * std) → [0.5, 1.0]
            w_var = torch.sigmoid(self.priority_variance_scale * query_std)

            # Frontier component: Gaussian centered at frontier_center
            w_frontier = torch.exp(
                -0.5 * ((query_mean - self.priority_frontier_center) / (self.priority_frontier_width + 1e-6)) ** 2
            )

            # Combined priority: product, then clamp to [min_weight, 1.0]
            priority_per_query = (w_var * w_frontier)
            # Normalize to mean=1 within batch so total gradient magnitude is preserved
            priority_per_query = priority_per_query / (priority_per_query.mean() + 1e-8)
            priority_per_query = priority_per_query.clamp(min=self.priority_min_weight)

            # Expand to per-sample (each query has G samples)
            priority_per_sample = priority_per_query.repeat_interleave(G, dim=0)

            # Slice to local process
            priority_local = priority_per_sample[process_slice]

            # Apply to advantages
            advantages = advantages * priority_local

            self._metrics["priority_weight_mean"].append(priority_per_query.mean().item())
            self._metrics["priority_weight_std"].append(priority_per_query.std().item())

        # Log v_top - v_ref for monitoring (use global tensors)
        G = self.num_generations
        grouped_raw = rewards.view(-1, G)
        v_ref_per_query = v_ref_global.view(-1, G)[:, 0]
        k = max(1, int(G * self.beta_controller.top_k_frac))
        if grouped_raw.shape[0] > 0 and grouped_raw.shape[1] >= k:
            v_top = grouped_raw.topk(k, dim=1).values.mean(dim=1)
            gap = (v_top - v_ref_per_query).mean().item()
        else:
            gap = 0.0

        # ============================================================
        # AdaRePO EXTENSION: memory bank (self-distillation)
        # ============================================================

        active_solutions = list(solutions)
        frac_self_distill = 0.0

        if self.use_memory_bank and self.memory_bank is not None:
            current_step = self.state.global_step
            n_self_distill = 0

            for idx in range(B_local):
                query = prompts_text[idx]
                ref_smi = solutions[idx]
                v_ref_val = v_ref_local[idx].item() if idx < v_ref_local.shape[0] else 0.0

                gen_smi = _extract_answer_smiles(completions_text[idx])
                if gen_smi is not None:
                    gen_reward = raw_rewards_local_gathered[idx].item()
                    self.memory_bank.try_promote(
                        query=query, smiles=gen_smi, reward=gen_reward,
                        ref_smiles=ref_smi, ref_reward=v_ref_val,
                        step=current_step,
                    )

                best_smi, _ = self.memory_bank.get_best_reference(
                    query=query, ref_smiles=ref_smi,
                    ref_reward=v_ref_val, current_step=current_step,
                )
                active_solutions[idx] = best_smi
                if best_smi != ref_smi:
                    n_self_distill += 1

            frac_self_distill = n_self_distill / max(B_local, 1)

        # ---------- Build solution_ids / solution_mask (RePO section) ----------
        completions_text_for_sol = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        solutions_text = [replace_smile(text, x) for text, x in zip(completions_text_for_sol, active_solutions)]

        solution_positions = []
        for text, sol in zip(solutions_text, active_solutions):
            start_char = text.find(sol)
            if start_char == -1:
                solution_positions.append((0, 0))
            else:
                solution_positions.append((start_char, start_char + len(sol)))

        solution_inputs = self.processing_class(
            solutions_text, return_tensors="pt", padding=True,
            padding_side="right", add_special_tokens=False,
            return_offsets_mapping=True,
        )
        offset_mappings = solution_inputs.pop("offset_mapping")
        batch_size, seq_len = offset_mappings.shape[:2]
        solution_mask = torch.zeros((batch_size, seq_len), dtype=torch.long)

        for i in range(batch_size):
            start_char, end_char = solution_positions[i]
            for j in range(seq_len):
                start, end = offset_mappings[i][j].tolist()
                if (start, end) != (0, 0) and start < end_char and end > start_char:
                    solution_mask[i][j] = 1

        solution_inputs["attention_mask"] = solution_mask
        solution_inputs = Trainer._prepare_inputs(self, inputs=solution_inputs)
        solution_ids = solution_inputs["input_ids"]
        solution_mask = solution_inputs["attention_mask"]

        # ---------- Log metrics ----------
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if self.reward_func_names is not None and i < len(self.reward_func_names):
                reward_func_name = self.reward_func_names[i]
            elif isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = getattr(reward_func, "__name__", f"reward_func_{i}")
                if reward_func_name == "<lambda>":
                    reward_func_name = f"reward_func_{i}"
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        _append_advantage_metrics(self._metrics, advantage_stats)
        self._metrics["beta_guide_mean"].append(beta_guide.mean().item())
        self._metrics["beta_guide_std"].append(beta_guide.std().item())
        self._metrics["v_top_minus_v_ref"].append(gap)

        if self.use_memory_bank:
            self._metrics["frac_self_distill"].append(frac_self_distill)
            self._metrics["memory_bank/total_entries"].append(
                self.memory_bank.total_size if self.memory_bank else 0
            )

        # ============================================================
        # AdaRePO EXTENSION: Experience Buffer — promote stable examples
        # Uses LOCAL tensors only (each process maintains its own buffer).
        # B_local = per_device_batch * G completions on this process.
        # ============================================================
        if self.use_experience_buffer and self.experience_buffer is not None:
            G = self.num_generations
            current_step = self.state.global_step
            # Compute per-query stats from LOCAL rewards
            n_local_queries = B_local // G
            local_rewards = raw_rewards_local_gathered  # (B_local,) — local slice of global rewards
            grouped_local = local_rewards.view(n_local_queries, G)
            mu_local = grouped_local.mean(dim=1)
            sigma_local = grouped_local.std(dim=1)
            n_promoted = 0
            for qi in range(n_local_queries):
                s_idx = qi * G
                e_idx = (qi + 1) * G
                promoted = self.experience_buffer.try_add(
                    prompt_text=prompts_text[qi],
                    solution=active_solutions[qi],
                    completion_ids=completion_ids[s_idx:e_idx],
                    completion_mask=completion_mask[s_idx:e_idx],
                    reward_mu=mu_local[qi].item(),
                    reward_sigma=sigma_local[qi].item(),
                    advantages=advantages[s_idx:e_idx],
                    beta_guide=beta_guide[s_idx:e_idx],
                    step=current_step,
                )
                if promoted:
                    n_promoted += 1
            buf_stats = self.experience_buffer.stats()
            for k, v in buf_stats.items():
                self._metrics[k].append(v)
            self._metrics["exp_buffer/n_promoted"].append(n_promoted)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "solution_ids": solution_ids,
            "solution_mask": solution_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "beta_guide": beta_guide,
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to apply dynamic beta_guide to the guidance loss.

        Original RePO:   loss = L_RL + s_loss    (batch-average s_loss)
        AdaRePO:         loss = L_RL + mean_i(beta_guide_i * s_loss_i)  (per-sample)
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Guidance loss (answer-level SFT on m_star)
        solution_ids, solution_mask = inputs["solution_ids"], inputs["solution_mask"]
        _disable_ref = getattr(self.args, "disable_reference_guidance", False)
        if not _disable_ref:
            input_ids_sol = torch.cat([prompt_ids, solution_ids], dim=1)
            attention_mask_sol = torch.cat([prompt_mask, solution_mask], dim=1)
            logits_to_keep_sol = solution_ids.size(1)
            s_per_token_logps = self._get_per_token_logps(
                model, input_ids_sol, attention_mask_sol, logits_to_keep_sol
            )
            # Per-sample guidance loss (Bug #2 fix: avoid batch-average before weighting)
            per_sample_mask_sum = solution_mask.sum(dim=1).clamp(min=1e-4)  # (B,)
            per_sample_s_loss = -(s_per_token_logps * solution_mask).sum(dim=1) / per_sample_mask_sum  # (B,)
            s_loss_mean = per_sample_s_loss.mean()  # for logging only
        else:
            per_sample_s_loss = torch.zeros(prompt_ids.size(0), device=prompt_ids.device)
            s_loss_mean = torch.tensor(0.0, device=prompt_ids.device)

        # KL divergence
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # RL loss (GRPO surrogate)
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss_rl = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Per-sample beta-weighted guidance loss (Bug #1 fix: per-prompt beta, not batch mean)
        beta_guide = inputs["beta_guide"]  # (B,) — per-sample, identical within prompt groups
        weighted_s_loss = (beta_guide * per_sample_s_loss).mean()
        loss = loss_rl + weighted_s_loss

        # Metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        self._metrics["s_loss"].append(self.accelerator.gather_for_metrics(s_loss_mean).mean().item())
        self._metrics["s_loss_weighted"].append(
            self.accelerator.gather_for_metrics(weighted_s_loss).mean().item()
        )

        return loss
