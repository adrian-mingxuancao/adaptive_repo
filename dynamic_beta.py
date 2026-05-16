"""
Dynamic Beta Controller for AdaRePO.

Computes the adaptive coefficient beta_guide(q) for the answer-level
reference guidance loss, based on the gap between sampled output rewards
and the reference molecule reward.

Implements four modes:
  - sigmoid_gap:    beta = beta_min + (beta_max - beta_min) * sigmoid(-alpha * (v_top - v_ref))
                    When student << teacher: beta → beta_max (strong IL, can be >1)
                    When student >> teacher: beta → beta_min (transition to RL)
  - sample_sigmoid: beta_i = beta_min + (beta_max-beta_min) * sigmoid(-alpha*(v_i-v_ref))   [per-sample]
  - rank:           beta = beta_max * (1 - frac(samples >= v_ref))
  - softmax_gap:    beta = beta_max * softmax_weight(v_ref among all)
  - confidence:     beta = beta_max * sigmoid(-alpha * (LCB_learner - UCB_ref))
  - fixed:          beta = beta_max  (constant, recovers original RePO)
"""
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class DynamicBetaController:
    """Stateless controller that computes beta_guide given rewards."""

    def __init__(
        self,
        mode: str = "sigmoid_gap",
        beta_max: float = 1.5,
        beta_min: float = 0.3,
        alpha: float = 3.0,
        top_k_frac: float = 0.33,
        softmax_tau: float = 1.0,
        confidence_threshold: float = 0.5,
    ):
        self.mode = mode
        self.beta_max = beta_max
        self.beta_min = beta_min
        self.alpha = alpha
        self.top_k_frac = top_k_frac
        self.softmax_tau = softmax_tau
        self.confidence_threshold = confidence_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        rewards: torch.Tensor,
        v_ref: torch.Tensor,
        num_generations: int,
        mu_learner: Optional[torch.Tensor] = None,
        sigma_learner: Optional[torch.Tensor] = None,
        mu_ref: Optional[torch.Tensor] = None,
        sigma_ref: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute beta_guide for each sample (or each query group).

        Args:
            rewards:  (B*G,) flat tensor of per-sample rewards.
            v_ref:    (B*G,) tensor of reference rewards, repeated per generation.
                      OR (B,) tensor; will be expanded.
            num_generations: G, number of completions per query.
            mu_learner, sigma_learner: ensemble predictions for learner (B,).
            mu_ref, sigma_ref: ensemble predictions for reference (B,).

        Returns:
            beta_guide: (B*G,) tensor of per-sample beta values.
        """
        device = rewards.device
        B_times_G = rewards.shape[0]
        B = B_times_G // num_generations
        G = num_generations

        # Ensure v_ref is (B*G,)
        if v_ref.shape[0] == B:
            v_ref = v_ref.repeat_interleave(G)

        if self.mode == "fixed":
            return torch.full((B_times_G,), self.beta_max, device=device)

        elif self.mode == "sigmoid_gap":
            return self._sigmoid_gap(rewards, v_ref, B, G, device)

        elif self.mode == "sample_sigmoid":
            return self._sample_sigmoid(rewards, v_ref, device)

        elif self.mode == "rank":
            return self._rank(rewards, v_ref, B, G, device)

        elif self.mode == "softmax_gap":
            return self._softmax_gap(rewards, v_ref, B, G, device)

        elif self.mode == "confidence":
            return self._confidence(
                rewards, v_ref, B, G, device,
                mu_learner, sigma_learner, mu_ref, sigma_ref,
            )

        else:
            raise ValueError(f"Unknown beta_guide mode: {self.mode}")

    # ------------------------------------------------------------------
    # Mode implementations
    # ------------------------------------------------------------------

    def _sigmoid_gap(
        self, rewards: torch.Tensor, v_ref: torch.Tensor, B: int, G: int, device
    ) -> torch.Tensor:
        """
        Query-level boosted sigmoid-gap beta.
        beta(q) = beta_min + (beta_max - beta_min) * sigmoid( -alpha * (v_top(q) - v_ref(q)) )

        When student << teacher (gap << 0): beta → beta_max (strong IL boost, can be >1)
        When student ≈  teacher (gap ≈  0): beta = (beta_max + beta_min) / 2
        When student >> teacher (gap >> 0): beta → beta_min (transition to RL)
        """
        grouped = rewards.view(B, G)  # (B, G)
        k = max(1, int(G * self.top_k_frac))
        top_k_vals, _ = grouped.topk(k, dim=1)  # (B, k)
        v_top = top_k_vals.mean(dim=1)  # (B,)

        v_ref_per_query = v_ref.view(B, G)[:, 0]  # (B,)
        gap = v_top - v_ref_per_query  # (B,)

        beta_q = self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(-self.alpha * gap)  # (B,)
        return beta_q.repeat_interleave(G)  # (B*G,)

    def _sample_sigmoid(
        self, rewards: torch.Tensor, v_ref: torch.Tensor, device
    ) -> torch.Tensor:
        """
        Per-sample boosted sigmoid beta.
        beta_i = beta_min + (beta_max - beta_min) * sigmoid( -alpha * (v_i - v_ref) )
        """
        gap = rewards - v_ref
        return self.beta_min + (self.beta_max - self.beta_min) * torch.sigmoid(-self.alpha * gap)

    def _rank(
        self, rewards: torch.Tensor, v_ref: torch.Tensor, B: int, G: int, device
    ) -> torch.Tensor:
        """
        Rank-based beta (scale-invariant).
        beta(q) = beta_max * (1 - frac(samples >= v_ref))
        """
        grouped = rewards.view(B, G)
        v_ref_per_query = v_ref.view(B, G)[:, 0].unsqueeze(1)  # (B, 1)
        frac_better = (grouped >= v_ref_per_query).float().mean(dim=1)  # (B,)
        beta_q = self.beta_max * (1.0 - frac_better)
        return beta_q.repeat_interleave(G)

    def _softmax_gap(
        self, rewards: torch.Tensor, v_ref: torch.Tensor, B: int, G: int, device
    ) -> torch.Tensor:
        """
        Softmax-gap beta.
        w_ref = exp(v_ref/tau) / (exp(v_ref/tau) + sum_i exp(v_i/tau))
        beta(q) = beta_max * w_ref
        """
        grouped = rewards.view(B, G)  # (B, G)
        v_ref_per_query = v_ref.view(B, G)[:, 0].unsqueeze(1)  # (B, 1)

        tau = max(self.softmax_tau, 1e-6)
        all_vals = torch.cat([v_ref_per_query, grouped], dim=1)  # (B, 1+G)
        logits = all_vals / tau
        weights = torch.softmax(logits, dim=1)  # (B, 1+G)
        w_ref = weights[:, 0]  # (B,)  weight of the reference
        beta_q = self.beta_max * w_ref
        return beta_q.repeat_interleave(G)

    def _confidence(
        self,
        rewards: torch.Tensor,
        v_ref: torch.Tensor,
        B: int,
        G: int,
        device,
        mu_learner: Optional[torch.Tensor],
        sigma_learner: Optional[torch.Tensor],
        mu_ref: Optional[torch.Tensor],
        sigma_ref: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Confidence-aware beta using UCB/LCB (RPI-style).
        UCB_ref = mu_ref + sigma_ref
        LCB_learner = mu_learner - sigma_learner
        beta = beta_max * sigmoid( -alpha * (LCB_learner - UCB_ref) )
        If sigma_ref > threshold, set beta = 0 (unreliable reference).
        """
        if mu_learner is None or mu_ref is None:
            logger.warning(
                "Confidence mode requested but ensemble predictions not provided. "
                "Falling back to sigmoid_gap."
            )
            return self._sigmoid_gap(rewards, v_ref, B, G, device)

        # Ensure shapes are (B,)
        if mu_learner.dim() == 0:
            mu_learner = mu_learner.unsqueeze(0).expand(B)
        if sigma_learner is None:
            sigma_learner = torch.zeros(B, device=device)
        if sigma_ref is None:
            sigma_ref = torch.zeros(B, device=device)

        ucb_ref = mu_ref + sigma_ref  # (B,)
        lcb_learner = mu_learner - sigma_learner  # (B,)

        gap = lcb_learner - ucb_ref
        beta_q = self.beta_max * torch.sigmoid(-self.alpha * gap)

        # Threshold: if reference uncertainty too high, don't trust it
        unreliable = sigma_ref > self.confidence_threshold
        beta_q = beta_q.masked_fill(unreliable, 0.0)

        return beta_q.repeat_interleave(G)
