"""
AdaRePO Configuration
Extends the RePO GRPOConfig with hyperparameters for dynamic beta,
memory bank (self-distillation), and confidence-aware ensemble.
"""
from dataclasses import dataclass, field
from typing import Optional

import trl


@dataclass
class AdaRePOConfig(trl.GRPOConfig):
    """
    Configuration for Adaptive Reference-guided Policy Optimization.

    Inherits all GRPO/RePO parameters and adds:
    - Dynamic beta controller parameters
    - Memory bank (self-distillation) parameters
    - Confidence-aware ensemble parameters
    """

    # --- Callbacks / logging (carried from RePO's GRPOConfig) ---
    benchmarks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "Benchmarks to run after training."}
    )
    callbacks: list[str] = field(
        default_factory=lambda: [], metadata={"help": "Callbacks to run during training."}
    )
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "Optional system prompt for benchmarking."}
    )
    hub_model_revision: Optional[str] = field(default="main")
    overwrite_hub_revision: bool = field(default=False)
    push_to_hub_revision: bool = field(default=False)
    wandb_entity: Optional[str] = field(default=None)
    wandb_project: Optional[str] = field(default=None)

    # ============================================================
    # Dynamic Beta Controller
    # ============================================================
    beta_guide_max: float = field(
        default=1.0,
        metadata={"help": "Maximum guidance strength (scales L_guidance)."},
    )
    beta_guide_min: float = field(
        default=0.0,
        metadata={"help": "Minimum guidance strength (floor for beta in sigmoid modes)."},
    )
    beta_guide_alpha: float = field(
        default=5.0,
        metadata={
            "help": "Temperature for sigmoid in dynamic beta. "
            "Higher = sharper transition from guidance to RL."
        },
    )
    beta_guide_top_k_frac: float = field(
        default=0.33,
        metadata={"help": "Fraction of G samples to use for top-k mean in v_top."},
    )
    beta_guide_mode: str = field(
        default="sigmoid_gap",
        metadata={
            "help": "Dynamic beta mode: sigmoid_gap | sample_sigmoid | rank | softmax_gap | confidence | fixed"
        },
    )
    beta_guide_softmax_tau: float = field(
        default=1.0,
        metadata={"help": "Temperature for softmax-gap beta variant."},
    )

    # ============================================================
    # Memory Bank (Self-Distillation)
    # ============================================================
    use_memory_bank: bool = field(
        default=False,
        metadata={"help": "Enable per-query memory bank for self-distillation."},
    )
    memory_bank_size: int = field(
        default=5,
        metadata={"help": "Max entries stored per query."},
    )
    promotion_margin: float = field(
        default=0.1,
        metadata={"help": "delta: generated mol must beat ref by this margin to be promoted."},
    )
    promotion_sim_min: float = field(
        default=0.3,
        metadata={"help": "Minimum Tanimoto similarity to reference for promotion."},
    )
    memory_bank_max_age: int = field(
        default=1000,
        metadata={"help": "Entries older than this many steps are evicted."},
    )

    # ============================================================
    # Priority Learning (Curriculum-style sample weighting)
    # ============================================================
    use_priority_weighting: bool = field(
        default=False,
        metadata={"help": "Weight advantages by prompt informativeness (reward variance + frontier)."},
    )
    priority_variance_scale: float = field(
        default=2.0,
        metadata={"help": "Scale for reward-variance contribution to priority weight."},
    )
    priority_frontier_center: float = field(
        default=0.3,
        metadata={"help": "Reward mean value considered 'learning frontier' (peaks priority)."},
    )
    priority_frontier_width: float = field(
        default=0.3,
        metadata={"help": "Width of the Gaussian frontier window around frontier_center."},
    )
    priority_min_weight: float = field(
        default=0.2,
        metadata={"help": "Minimum priority weight (floor so no sample is fully ignored)."},
    )

    # ============================================================
    # Experience Buffer (Stable Example Replay)
    # ============================================================
    use_experience_buffer: bool = field(
        default=False,
        metadata={"help": "Cache stable-good examples and replay them instead of re-generating."},
    )
    exp_buffer_max_size: int = field(
        default=256,
        metadata={"help": "Max entries in experience buffer."},
    )
    exp_buffer_max_age: int = field(
        default=100,
        metadata={"help": "Max age (steps) before buffer entry is evicted."},
    )
    exp_buffer_max_replay_per_batch: int = field(
        default=2,
        metadata={"help": "Max buffer entries replayed per batch (per process)."},
    )
    exp_buffer_sigma_threshold: float = field(
        default=0.05,
        metadata={"help": "EMA sigma below this qualifies for buffer promotion."},
    )
    exp_buffer_mu_threshold: float = field(
        default=0.4,
        metadata={"help": "Reward mean above this qualifies for buffer promotion."},
    )

    # ============================================================
    # Adaptive Temperature (Anti-Collapse)
    # ============================================================
    use_adaptive_temperature: bool = field(
        default=False,
        metadata={"help": "Enable per-prompt adaptive temperature based on EMA reward sigma."},
    )
    adaptive_temp_base: float = field(
        default=0.9,
        metadata={"help": "Base temperature for generation (normal prompts)."},
    )
    adaptive_temp_high: float = field(
        default=1.3,
        metadata={"help": "High temperature for collapsed prompts (low sigma)."},
    )
    adaptive_temp_sigma_threshold: float = field(
        default=0.05,
        metadata={"help": "EMA sigma below this triggers high temperature."},
    )
    adaptive_temp_ema_alpha: float = field(
        default=0.3,
        metadata={"help": "EMA smoothing factor for per-prompt sigma tracking (higher = faster update)."},
    )

    # ============================================================
    # Confidence-Aware Ensemble (Phase 3)
    # ============================================================
    use_ensemble: bool = field(
        default=False,
        metadata={"help": "Enable ensemble reward predictor for confidence-aware beta."},
    )
    ensemble_size: int = field(
        default=5,
        metadata={"help": "Number of models in the reward predictor ensemble."},
    )
    confidence_threshold: float = field(
        default=0.5,
        metadata={"help": "Gamma_s: if sigma_ref > this, fall back to RL only (set beta=0)."},
    )
    ensemble_update_freq: int = field(
        default=10,
        metadata={"help": "Update ensemble every N training steps."},
    )
    ensemble_hidden_dim: int = field(
        default=128,
        metadata={"help": "Hidden dimension for ensemble MLP reward predictors."},
    )
    ensemble_fp_bits: int = field(
        default=1024,
        metadata={"help": "Morgan fingerprint bit length for ensemble input."},
    )
