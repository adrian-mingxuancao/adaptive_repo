"""
Experience Buffer for AdaRePO — Stable Example Replay.

Caches training-ready tensors for "stable & good" (Case A) examples,
allowing them to be replayed without re-running vLLM generation.
This saves ~50s/step of inference compute for each cached example
that replaces a fresh generation slot.

Design:
  - Each buffer entry stores: prompt_ids, prompt_mask, completion_ids,
    completion_mask, solution_ids, solution_mask, advantages, beta_guide,
    plus metadata (prompt_hash, reward_mu, reward_sigma, step, age).
  - Entries are promoted when: sigma < sigma_threshold AND mu > mu_threshold.
  - Entries are evicted when: age > max_age steps.
  - At sampling time, trainer calls `try_replace_batch()` which returns
    indices of batch slots that can be filled from buffer, plus the
    cached tensors.

Note: ref_per_token_logps are NOT cached (they'd go stale as model updates).
Instead, only completion_ids are cached, and ref_logps are recomputed each
step. This means we save vLLM generation but still pay the ref model forward
pass — a good trade-off since ref forward is ~5s vs generation ~50s.
"""
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class BufferEntry:
    """A single cached experience."""
    prompt_hash: str
    prompt_text: str
    solution: str           # gold or memory-bank-promoted SMILES
    completion_ids: torch.Tensor   # (G, seq_len) on CPU
    completion_mask: torch.Tensor  # (G, seq_len) on CPU
    reward_mu: float
    reward_sigma: float
    advantages: torch.Tensor       # (G,) on CPU
    beta_guide: torch.Tensor       # (G,) on CPU
    step: int               # step when this entry was created/updated
    n_replays: int = 0      # how many times this entry has been replayed


class ExperienceBuffer:
    """
    Fixed-capacity buffer of stable, high-reward training examples.

    Promotion criterion (from trainer):
        EMA_sigma(prompt) < sigma_threshold  AND  reward_mu > mu_threshold

    Replacement policy:
        When buffer is full, replace the entry with lowest reward_mu.

    Replay policy:
        Each step, up to `max_replay_per_batch` examples from the buffer
        can replace fresh generation slots. Selection prioritizes entries
        with fewer replays (fairness) and higher reward (quality).
    """

    def __init__(
        self,
        max_size: int = 256,
        max_age: int = 100,
        max_replay_per_batch: int = 2,
        sigma_threshold: float = 0.05,
        mu_threshold: float = 0.4,
    ):
        self.max_size = max_size
        self.max_age = max_age
        self.max_replay_per_batch = max_replay_per_batch
        self.sigma_threshold = sigma_threshold
        self.mu_threshold = mu_threshold
        self._buffer: Dict[str, BufferEntry] = OrderedDict()

    # ------------------------------------------------------------------
    # Promotion: add or update entries
    # ------------------------------------------------------------------

    def try_add(
        self,
        prompt_text: str,
        solution: str,
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        reward_mu: float,
        reward_sigma: float,
        advantages: torch.Tensor,
        beta_guide: torch.Tensor,
        step: int,
    ) -> bool:
        """
        Try to add/update an entry. Returns True if added/updated.
        Only adds if sigma < threshold AND mu > threshold.
        """
        if reward_sigma >= self.sigma_threshold:
            return False
        if reward_mu < self.mu_threshold:
            return False

        ph = hashlib.md5(prompt_text.encode()).hexdigest()

        entry = BufferEntry(
            prompt_hash=ph,
            prompt_text=prompt_text,
            solution=solution,
            completion_ids=completion_ids.detach().cpu(),
            completion_mask=completion_mask.detach().cpu(),
            reward_mu=reward_mu,
            reward_sigma=reward_sigma,
            advantages=advantages.detach().cpu(),
            beta_guide=beta_guide.detach().cpu(),
            step=step,
            n_replays=0,
        )

        if ph in self._buffer:
            # Update existing entry
            self._buffer[ph] = entry
            return True

        if len(self._buffer) < self.max_size:
            self._buffer[ph] = entry
            return True

        # Buffer full — replace lowest-mu entry
        worst_ph = min(self._buffer, key=lambda k: self._buffer[k].reward_mu)
        if reward_mu > self._buffer[worst_ph].reward_mu:
            del self._buffer[worst_ph]
            self._buffer[ph] = entry
            return True

        return False

    # ------------------------------------------------------------------
    # Eviction: remove stale entries
    # ------------------------------------------------------------------

    def evict_stale(self, current_step: int):
        """Remove entries older than max_age."""
        if self.max_age <= 0:
            return
        to_remove = [
            ph for ph, e in self._buffer.items()
            if (current_step - e.step) > self.max_age
        ]
        for ph in to_remove:
            del self._buffer[ph]

    # ------------------------------------------------------------------
    # Replay: select entries for batch replacement
    # ------------------------------------------------------------------

    def sample_for_replay(
        self,
        current_prompts: List[str],
        current_step: int,
        device: torch.device,
    ) -> Tuple[List[int], List[BufferEntry]]:
        """
        Select buffer entries to replay in the current batch.

        Returns:
            replace_indices: list of batch indices to replace
            entries: corresponding BufferEntry objects

        Strategy: for each prompt in current batch, check if it's in buffer.
        If yes, mark it for replacement. Cap at max_replay_per_batch.
        """
        self.evict_stale(current_step)

        if not self._buffer:
            return [], []

        replace_indices = []
        entries = []

        for idx, pt in enumerate(current_prompts):
            if len(replace_indices) >= self.max_replay_per_batch:
                break
            ph = hashlib.md5(pt.encode()).hexdigest()
            if ph in self._buffer:
                entry = self._buffer[ph]
                # Only replay if entry is still "fresh enough"
                if (current_step - entry.step) <= self.max_age:
                    replace_indices.append(idx)
                    entries.append(entry)
                    entry.n_replays += 1

        return replace_indices, entries

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._buffer)

    def stats(self) -> dict:
        if not self._buffer:
            return {
                "exp_buffer/size": 0,
                "exp_buffer/avg_mu": 0.0,
                "exp_buffer/avg_sigma": 0.0,
                "exp_buffer/avg_replays": 0.0,
            }
        entries = list(self._buffer.values())
        return {
            "exp_buffer/size": len(entries),
            "exp_buffer/avg_mu": sum(e.reward_mu for e in entries) / len(entries),
            "exp_buffer/avg_sigma": sum(e.reward_sigma for e in entries) / len(entries),
            "exp_buffer/avg_replays": sum(e.n_replays for e in entries) / len(entries),
            "exp_buffer/max_replays": max(e.n_replays for e in entries),
        }
