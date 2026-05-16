# Adaptive Reference-guided Policy Optimization (AdaRePO)
## Bridging RePO and RPI for Molecular Optimization LLMs

---

## A. High-Level Synthesis of the Two Papers

**RePO** (Reference-guided Policy Optimization) addresses a critical gap in LLM-based molecular optimization: supervised fine-tuning (SFT) on reference molecules collapses reasoning chains, while pure RLVR (e.g., GRPO) provides sparse rewards under similarity constraints because the model lacks effective exploration early in training. RePO's solution is a hybrid objective that combines GRPO-style reinforcement learning with answer-level reference guidance. At each update, the model samples candidate molecules with reasoning trajectories, computes GRPO advantages for the RL term, and simultaneously trains on the reference molecule answer using the model's own reasoning chain as context -- but only on the answer tokens (the SMILES in `<answer>...</answer>`), not the reasoning tokens. This preserves the model's reasoning ability while grounding outputs to known-good molecular edits. The RL term drives exploration of novel molecules; the guidance term mitigates reward sparsity by anchoring to references.

**RPI** (Robust Policy Improvement) addresses the complementary problem of *when* to imitate vs. self-improve. Given a set of black-box oracles (possibly suboptimal), RPI defines a max+ baseline `f+(s) = max_k V^k(s)` over the extended oracle set (oracles + learner). It introduces RAPS (Robust Active Policy Selection), which uses UCB of oracle value estimates vs. LCB of the learner's value estimate to decide per-state whether to imitate an oracle or self-improve. The key insight: use UCB for oracles (optimistic about their potential) and LCB for the learner (conservative about own performance) so the learner only stops imitating when it is *confidently* better. As training progresses, the learner enters the extended oracle set and eventually becomes the dominant policy, at which point RPI reduces to pure RL.

**The bridge**: RePO's fixed beta for the answer-level guidance term is the molecular-optimization analogue of RPI's static oracle-following. By making beta dynamic -- a function of sampled output quality relative to the reference -- we import RPI's adaptive IL/RL blending into the RePO framework without requiring a separate critic network or per-state value estimation, since molecular optimization rewards are computed per-completion rather than per-timestep.

---

## B. Why Dynamic Beta Is the Right Bridge

In RePO, the total loss is:

```
L = L_GRPO + L_guidance    where
L_GRPO  = -( exp(log_pi - log_pi.detach()) * A_i - beta_KL * KL(pi || pi_ref) )   (over completion tokens)
L_guidance = -( log_pi(m_ref | q, t_i) )   (over answer tokens only, using model's own reasoning t_i)
```

The guidance term `L_guidance` (called `s_loss` in the code) is added with an **implicit weight of 1.0** -- it has no explicit coefficient in the current implementation. This means the reference molecule always contributes equally to the gradient regardless of whether the model is already generating better molecules than the reference.

This is precisely the failure mode RPI identifies: naively imitating the oracle even when the learner has surpassed it. In RPI's terms, RePO's current design corresponds to `f+(s) = V^oracle(s)` always, never considering `V^learner(s)`.

**Dynamic beta is the minimal intervention** that imports RPI's active switching logic:
- When sampled molecules are worse than `m_ref` --> high beta --> strong guidance (imitation regime)
- When sampled molecules match `m_ref` --> moderate beta --> balanced exploration/exploitation
- When sampled molecules surpass `m_ref` --> low/zero beta --> pure RL (self-improvement regime)

This is principled because:
1. It preserves RePO's answer-level guidance structure (no architectural changes)
2. It naturally implements RPI's "learner enters the oracle set" -- when the learner surpasses the reference, guidance fades
3. It's grounded in the actual reward signal, not a heuristic schedule

---

## C. First-Pass Objective with Dynamic Beta

### C.1 Notation (from RePO)

- **q**: query/prompt (instruction to modify a molecule)
- **o_i = [t_i ; m_i]**: the i-th sampled output, consisting of reasoning trace t_i and answer molecule m_i
- **m_ref**: reference molecule from the training data (the "solution")
- **v(o_i) = r(o_i)**: the reward for sample i (e.g., `smile_optimization` reward combining property improvement + similarity)
- **v(m_ref)**: the reward that the reference molecule would receive under the same reward function
- **G**: number of generations per query (num_generations)

### C.2 RePO Objective (from source code, exactly)

```
L_RePO = L_RL + L_guidance

L_RL = -1/|C| * sum_{tokens in completion} [ (exp(log_pi - log_pi.detach()) * A_i - beta_KL * KL_token) * mask ]

L_guidance = -1/|S| * sum_{tokens in answer} [ log_pi(token | q, t_i) * answer_mask ]
```

where:
- `A_i = (r_i - mean(r)) / (std(r) + eps)` is the GRPO group-normalized advantage
- `KL_token = exp(log_pi_ref - log_pi) - (log_pi_ref - log_pi) - 1` (reverse KL)
- `answer_mask` selects only the SMILES tokens within `<answer>...</answer>`, keeping the reasoning trace t_i as context but not training on it
- beta_KL is `args.beta` (the KL penalty coefficient, typically 0.04)
- The guidance loss `s_loss` is added with **implicit weight 1.0**

### C.3 Dynamic Beta Formulation

We introduce `beta_guide(q)` as the coefficient for `L_guidance`, replacing the implicit 1.0:

```
L_AdaRePO = L_RL + beta_guide(q) * L_guidance
```

#### Version 1: Query-Level Sigmoid-Gap Beta

```
beta_guide(q) = beta_max * sigmoid( -alpha * (v_top(q) - v_ref(q)) )
```

where:
- `v_top(q) = mean of top-k rewards among {v(o_1), ..., v(o_G)}` (k = max(1, G//3))
- `v_ref(q)` = reward of the reference molecule (computed once per query)
- `alpha > 0` is a temperature parameter controlling transition sharpness
- `beta_max` is the maximum guidance strength (default: 1.0, matching current RePO)

**Behavior**:
- When `v_top >> v_ref` (learner surpasses reference): sigmoid -> 0, beta -> 0 (pure RL)
- When `v_top << v_ref` (learner much worse): sigmoid -> 1, beta -> beta_max (strong guidance)
- When `v_top ~ v_ref`: beta ~ beta_max/2 (balanced)

#### Version 2: Sample-Level Beta

```
beta_i(q) = beta_max * sigmoid( -alpha * (v(o_i) - v_ref(q)) )
```

Each sample gets its own guidance weight. Bad samples get strong reference signal; good samples get weak reference signal. This is closer to RPI's per-state switching.

#### Version 3: Rank-Based Beta

```
beta_guide(q) = beta_max * (1 - rank_fraction(v_ref, {v(o_i)}))
```

where `rank_fraction` = fraction of samples that have reward >= v_ref. If 80% of samples beat the reference, beta = 0.2 * beta_max.

#### Version 4: Softmax-Gap Beta

```
w_ref = exp(v_ref / tau) / (exp(v_ref / tau) + sum_i exp(v(o_i) / tau))
beta_guide(q) = beta_max * w_ref
```

The reference competes with all samples in a softmax. As samples improve, the reference's softmax weight shrinks.

### C.4 Recommended First Experiment

**Version 1 (Query-Level Sigmoid-Gap)** is the best first experiment because:
1. Single scalar per query -- simplest to implement, debug, and log
2. Sigmoid provides smooth, bounded output [0, beta_max]
3. Only one new hyperparameter (alpha) beyond beta_max
4. Directly interpretable: logs beta_guide(q) over training to visualize the IL-to-RL transition
5. Top-k mean is robust to outlier samples

Specifically: `alpha=5.0, beta_max=1.0, k=max(1, G//3)`.

---

## D. Confidence-Aware Extension (RPI-Style)

### D.1 Motivation

The first-pass beta uses point estimates of reward. But molecular rewards are noisy -- a single sample might get a high reward by luck (invalid SMILES that happens to parse, a trivially similar molecule). RPI's key insight is to use **uncertainty** in value estimates to make the IL/RL switch robust.

### D.2 Ensemble Value Estimator

Maintain an ensemble of M lightweight reward predictors `{f_1, ..., f_M}` (e.g., small MLPs on Morgan fingerprint of the generated SMILES + prompt embedding):

```
mu_learner(q) = (1/M) * sum_j f_j(q, o_best)     # mean predicted reward for learner's best sample
sigma_learner(q) = std_j(f_j(q, o_best))           # uncertainty

mu_ref(q) = (1/M) * sum_j f_j(q, m_ref)
sigma_ref(q) = std_j(f_j(q, m_ref))
```

These are trained on the actual rewards observed during training (cheap: just log (query, molecule, reward) tuples and periodically update the ensemble).

### D.3 Confidence-Aware Beta (UCB/LCB)

Following RPI's RAPS logic:
- **UCB for the reference** (optimistic about oracle): `V_ref_UCB = mu_ref + sigma_ref`
- **LCB for the learner** (conservative about self): `V_learner_LCB = mu_learner - sigma_learner`

```
beta_conf(q) = beta_max * sigmoid( -alpha * (V_learner_LCB(q) - V_ref_UCB(q)) )
```

**Interpretation**: The learner must be *confidently* better than the reference (even accounting for uncertainty) before guidance fades. This is exactly RPI's philosophy: "use LCB for the learner, UCB for the oracle."

### D.4 Confidence Threshold (from RPI's Gamma_s)

Following RPI Remark 6.3, add a confidence threshold Gamma:

```
if sigma_ref(q) > Gamma:
    # Reference value too uncertain -- fall back to RL only
    beta_conf(q) = 0
```

This prevents the system from trusting unreliable reference value estimates (analogous to RPI's `Gamma_s = 0.5` threshold controlling oracle reliability).

### D.5 Connection to RPI

| RPI Concept | AdaRePO Analogue |
|------------|-----------------|
| Oracle set {pi^k} | Reference molecule m_ref |
| Learner policy pi_n | Current model's sampled molecules {o_i} |
| State s | Query q |
| V^k(s) (oracle value) | v(m_ref) or mu_ref(q) |
| V^{K+1}(s) (learner value) | v_top(q) or mu_learner(q) |
| UCB/LCB policy selection | beta_conf(q) via UCB(ref) vs LCB(learner) |
| f+(s) = max(V^oracle, V^learner) | Implicit in beta -> 0 when learner dominates |
| A^{GAE+} advantage | GRPO advantage A_i (already group-normalized) |
| Gamma_s threshold | Confidence threshold Gamma |

---

## E. Self-Distillation / Active-Reference Extension

### E.1 Memory Bank

Maintain a per-query memory bank `B(q)` of high-value generated molecules:

```python
B = {}  # key: query_hash, value: list of (molecule_smiles, reward, step) tuples
MAX_BANK_SIZE_PER_QUERY = 5
```

### E.2 Promotion Criterion

A generated molecule `m_i` from sample `o_i` is promoted to pseudo-reference if:

```
v(o_i) > v(m_ref) + delta   AND   sim(m_i, m_ref) > sim_min   AND   is_valid(m_i)
```

where:
- `delta >= 0` is a promotion margin (default: 0.1) ensuring the generated molecule is meaningfully better
- `sim_min` is a minimum structural similarity threshold (e.g., 0.3) preventing degenerate solutions
- `is_valid` checks RDKit SMILES validity

### E.3 Active Reference Selection

At each training step, for query q, the guidance target is:

```
m_star(q) = argmax_{m in {m_ref} union B(q)} v(m)
```

If the memory bank contains a molecule better than the original reference, the guidance loss trains toward that molecule instead:

```
L_guidance = -beta_guide(q) * (1/|S|) * sum_{answer tokens} log_pi(m_star | q, t_i)
```

### E.4 Decay and Freshness

To prevent stale references:
- Molecules older than `K` training steps are removed from the bank
- Each promotion replaces the lowest-reward entry if the bank is full
- The reward for bank entries can be periodically re-evaluated (optional, expensive)

### E.5 Connection to "Learner Becomes the Improved Oracle"

This directly mirrors RPI's Extended Oracle Set (Definition 4.1):

```
Pi^E_n = Pi^o union {pi_n}    (RPI: learner joins the oracle set)
```

In our case:
```
References(q, step_n) = {m_ref} union B_n(q)    (AdaRePO: generated molecules join the reference set)
```

When B(q) contains molecules much better than m_ref, the dynamic beta will be small (because the learner is already good), and the guidance target shifts from the original reference to the learner's own best outputs. This is **self-distillation**: the model teaches itself from its own best generations.

The progression:
1. **Early training**: B(q) empty, m_star = m_ref, beta high --> strong reference guidance (IL regime)
2. **Mid training**: B(q) has some entries, m_star may shift, beta moderate --> blended regime
3. **Late training**: B(q) has high-quality molecules, m_star = best generated, beta low --> self-distillation replaces external reference

---

## F. Pseudocode for the Full Training Loop

```python
# === AdaRePO Training Loop ===

# Initialize
model = load_pretrained(model_path)
ref_model = create_reference_model(model)  # for KL computation
memory_bank = {}                            # query_hash -> [(smiles, reward, step)]
ensemble = [MLP() for _ in range(M)]        # optional: reward predictor ensemble

for step in range(total_steps):
    batch = sample_batch(dataset)  # each item has (query q, reference m_ref)

    for each query q in batch:
        # 1. Generate G completions
        outputs = [model.generate(q) for _ in range(G)]  # o_i = [t_i; m_i]

        # 2. Compute rewards
        rewards = [reward_fn(q, o_i) for o_i in outputs]  # v(o_1), ..., v(o_G)
        v_ref = reward_fn(q, m_ref)                         # v(m_ref)

        # 3. GRPO advantages (group-normalized)
        mean_r, std_r = mean(rewards), std(rewards)
        advantages = [(r - mean_r) / (std_r + eps) for r in rewards]

        # 4. Compute dynamic beta
        v_top = mean(top_k(rewards, k=max(1, G//3)))

        # --- First pass: sigmoid-gap ---
        beta_guide = beta_max * sigmoid(-alpha * (v_top - v_ref))

        # --- OR confidence-aware version ---
        # mu_l, sigma_l = ensemble_predict(q, best_molecule)
        # mu_r, sigma_r = ensemble_predict(q, m_ref)
        # V_learner_LCB = mu_l - sigma_l
        # V_ref_UCB = mu_r + sigma_r
        # beta_guide = beta_max * sigmoid(-alpha * (V_learner_LCB - V_ref_UCB))
        # if sigma_r > Gamma: beta_guide = 0  # unreliable reference

        # 5. Select active reference (self-distillation)
        candidates = [m_ref] + [m for (m, r, s) in memory_bank.get(hash(q), [])]
        m_star = argmax(candidates, key=lambda m: reward_fn(q, m))

        # 6. Update memory bank
        for o_i, r_i in zip(outputs, rewards):
            m_i = extract_answer(o_i)
            if r_i > v_ref + delta and sim(m_i, m_ref) > sim_min and is_valid(m_i):
                memory_bank_add(hash(q), (m_i, r_i, step))

        # 7. Compute losses
        # L_RL: GRPO surrogate loss with KL penalty (over all completion tokens)
        L_RL = grpo_loss(model, ref_model, outputs, advantages, beta_kl)

        # L_guidance: answer-level SFT on m_star (answer tokens only, reasoning as context)
        for each o_i:
            t_i = extract_reasoning(o_i)
            context = [q, t_i]  # keep model's own reasoning
            L_guidance_i = -log_prob(model, m_star | context) * answer_mask
        L_guidance = mean(L_guidance_i)

        # 8. Total loss
        L = L_RL + beta_guide * L_guidance

    # 9. Gradient step
    L.backward()
    optimizer.step()

    # 10. (Optional) Update ensemble on observed (query, molecule, reward) tuples
    # ensemble.update(training_buffer)

    # 11. Log metrics
    log("beta_guide_mean", mean(all beta_guide this step))
    log("memory_bank_size", sum(len(v) for v in memory_bank.values()))
    log("frac_self_distill", fraction of queries where m_star != m_ref)
```

---

## G. Ablation Plan

| # | Experiment | What Changes | What It Tests |
|---|-----------|-------------|--------------|
| 0 | **Baseline RePO** | Nothing (beta_guide=1.0 fixed) | Reproduction baseline |
| 1 | **Dynamic beta (sigmoid-gap)** | beta_guide = sigmoid(-alpha*(v_top - v_ref)) | Core hypothesis: adaptive guidance improves over fixed |
| 2 | **Sample-level beta** | beta_i per sample instead of per query | Is per-sample adaptation better? |
| 3 | **Rank-based beta** | beta = 1 - frac(samples > ref) | Robustness to reward scale |
| 4 | **Alpha sweep** | alpha in {1, 3, 5, 10, 20} | Sensitivity of transition sharpness |
| 5 | **beta_max sweep** | beta_max in {0.1, 0.5, 1.0, 2.0} | Optimal guidance ceiling |
| 6 | **Self-distillation** | Add memory bank + active reference | Does self-teaching help? |
| 7 | **Confidence-aware** | Add ensemble + UCB/LCB | Does uncertainty help switching? |
| 8 | **No guidance (GRPO only)** | beta_guide = 0 always | Verify guidance helps at all |
| 9 | **Decay beta_max** | beta_max *= 0.99 per epoch | Is explicit annealing needed or does adaptive suffice? |

**Recommended order**: 0 -> 8 -> 1 -> 4 -> 5 -> 3 -> 6 -> 2 -> 7 -> 9

---

## H. Failure Modes and What to Monitor

### H.1 Beta Collapses to Zero Too Early
**Symptom**: beta_guide drops to ~0 within first epoch; model generates high-reward but invalid/trivially-similar molecules.
**Monitor**: `beta_guide_mean` over training; SMILES validity rate; diversity of generated molecules.
**Fix**: Increase alpha (slower transition), add validity gating to v_top computation.

### H.2 Beta Never Decreases
**Symptom**: beta_guide stays near beta_max; model essentially doing SFT, not exploring.
**Monitor**: `v_top - v_ref` distribution; should trend positive over training.
**Fix**: Decrease alpha, check that reward function is well-calibrated (v_ref should be achievable).

### H.3 Memory Bank Poisoning
**Symptom**: Memory bank fills with molecules that got lucky high rewards but are actually poor.
**Monitor**: Re-evaluate bank entries periodically; track `frac_self_distill` vs actual reward improvement.
**Fix**: Increase promotion margin delta; add validity re-checking; limit bank entries by recency.

### H.4 Reward Scale Mismatch
**Symptom**: v_ref and v(o_i) on very different scales, making sigmoid always saturated.
**Monitor**: Histogram of `v_top - v_ref` values.
**Fix**: Use rank-based beta (Version 3) which is scale-invariant; or normalize rewards before computing gap.

### H.5 KL Divergence Explosion
**Symptom**: KL increases rapidly when beta_guide drops and the RL term dominates.
**Monitor**: `kl` metric from trainer.
**Fix**: Ensure beta_KL (the KL penalty, separate from beta_guide) is not too small; consider increasing it when beta_guide decreases.

### H.6 Ensemble Overconfidence (Confidence-Aware Version)
**Symptom**: sigma_learner and sigma_ref both small despite high actual variance.
**Monitor**: Calibration plots of ensemble predictions vs actual rewards.
**Fix**: Use proper ensembling with dropout; add epistemic uncertainty regularization.

### Key Metrics Dashboard
- `beta_guide_mean`, `beta_guide_std` (per step)
- `v_top_minus_v_ref` (per step, should trend positive)
- `frac_self_distill` (fraction of queries using memory bank target)
- `memory_bank_total_size`
- `reward_mean`, `reward_std` (standard)
- `kl`, `s_loss` (standard)
- `smiles_validity_rate` (critical for molecular optimization)
- `tanimoto_similarity_mean` (ensure diversity)

---

## I. Minimal Implementation Plan

### I.1 What to Modify in RePO Codebase

The implementation requires changes in **3 files** and **1 new module**:

#### New Files (in `agent_drug_discovery/adaptive_repo/`)

1. **`__init__.py`** - Package init
2. **`dynamic_beta.py`** - All beta computation logic (sigmoid-gap, rank-based, softmax-gap, confidence-aware)
3. **`memory_bank.py`** - Per-query memory bank for self-distillation
4. **`ada_repo_trainer.py`** - Subclass of `XGRPOTrainer` that overrides `_prepare_inputs` and `compute_loss`
5. **`ada_repo_config.py`** - Config dataclass extending `GRPOConfig` with new hyperparameters
6. **`ada_repo.py`** - Entry point script (like `repo.py` but using `AdaRePoTrainer`)
7. **`METHOD_DESIGN.md`** - This document

#### Specific Code Changes

**In `ada_repo_trainer.py` (subclass of XGRPOTrainer)**:

1. **`_prepare_inputs`**: After computing rewards, also compute `v_ref` for each query and store `beta_guide` in the returned dict.

2. **`compute_loss`**: Replace `loss = loss + s_loss` with `loss = loss + beta_guide * s_loss` where `beta_guide` is the dynamic coefficient.

3. **Memory bank update**: After reward computation, check promotion criteria and update bank.

4. **Active reference selection**: Before constructing `solution_ids`/`solution_mask`, select `m_star` from `{m_ref} union B(q)`.

#### No Changes Required To
- Reward functions (`rewards.py`) -- reward computation is independent
- Data loading (`repo.py` dataset logic) -- reused as-is
- vLLM generation -- unchanged
- DeepSpeed/training infrastructure -- unchanged

### I.2 Implementation Phases

**Phase 1 (MVP - 1 day)**: Dynamic beta only
- Implement `dynamic_beta.py` with sigmoid-gap
- Subclass trainer with `beta_guide` in compute_loss
- Run experiment 0 vs 1

**Phase 2 (+ 1 day)**: Self-distillation
- Implement `memory_bank.py`
- Add active reference selection to trainer
- Run experiment 6

**Phase 3 (+ 2 days)**: Confidence-aware
- Implement ensemble reward predictor
- Add UCB/LCB logic
- Run experiment 7

### I.3 Compute Requirements (Polaris)

Same as current RePO training:
- 1 node, 4x A100 40GB
- GPUs 0-2: DeepSpeed ZeRO-3, GPU 3: vLLM
- Memory bank is CPU-only (dict), negligible overhead
- Dynamic beta computation: ~10 lines of torch ops, negligible
- Ensemble (Phase 3): M=5 small MLPs, <100MB total, runs on CPU

### I.4 Hyperparameter Defaults

```yaml
# Dynamic beta
beta_guide_max: 1.0          # Maximum guidance strength
beta_guide_alpha: 5.0        # Sigmoid temperature
beta_guide_top_k_frac: 0.33  # Fraction of samples for top-k mean
beta_guide_mode: "sigmoid_gap"  # sigmoid_gap | rank | softmax_gap | confidence

# Self-distillation
use_memory_bank: false
memory_bank_size: 5          # Max entries per query
promotion_margin: 0.1        # delta: how much better than ref to promote
promotion_sim_min: 0.3       # Minimum similarity for promotion
memory_bank_max_age: 1000    # Steps before entry expires

# Confidence-aware (Phase 3)
use_ensemble: false
ensemble_size: 5
confidence_threshold: 0.5    # Gamma_s from RPI
ensemble_update_freq: 10     # Update ensemble every N steps
```
