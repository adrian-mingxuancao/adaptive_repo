# AdaRePO: Adaptive Reference-guided Policy Optimization with Priority-Weighted Learning

## 1. Background: RePO Objective

RePO (Reference-guided Policy Optimization) extends GRPO by adding a guidance loss that steers the policy toward a known reference molecule $m^*$:

$$\mathcal{L}_{\text{RePO}} = \mathcal{L}_{\text{RL}} + \mathcal{L}_{\text{guide}}$$

where $\mathcal{L}_{\text{RL}}$ is the GRPO surrogate loss with KL penalty, and $\mathcal{L}_{\text{guide}}$ is a supervised cross-entropy loss on the reference answer tokens.

**Problem:** The fixed guidance weight treats all training stages and prompts equally — early in training the model needs strong guidance, but later it should rely more on its own exploration via RL.

---

## 2. AdaRePO Extension 1: Dynamic Beta Guidance

AdaRePO introduces an adaptive coefficient $\beta$ that modulates guidance strength based on how well the policy already performs relative to the reference:

$$\mathcal{L}_{\text{AdaRePO}} = \mathcal{L}_{\text{RL}} + \beta \cdot \mathcal{L}_{\text{guide}}$$

The dynamic beta is computed per-batch using the **sigmoid-gap schedule**:

$$\beta = \beta_{\max} \cdot \sigma\left(-\alpha \cdot (v_{\text{top}} - v_{\text{ref}})\right)$$

where:
- $v_{\text{top}} = \frac{1}{k}\sum_{i=1}^{k} r_{(i)}$ — mean reward of top-$k$ generations per prompt ($k = \lfloor 0.33 \cdot G \rfloor$)
- $v_{\text{ref}} = R(m^*)$ — reward of the reference molecule
- $\sigma(\cdot)$ — sigmoid function
- $\alpha$ — controls transition sharpness (default: 5.0)
- $\beta_{\max}$ — maximum guidance strength (default: 1.0)

**Intuition:**
| Condition | $\beta$ value | Behavior |
|-----------|--------------|----------|
| $v_{\text{top}} \gg v_{\text{ref}}$ (policy beats reference) | $\beta \to 0$ | Pure RL — model explores freely |
| $v_{\text{top}} \approx v_{\text{ref}}$ (policy matches reference) | $\beta \approx 0.5$ | Balanced RL + guidance |
| $v_{\text{top}} \ll v_{\text{ref}}$ (policy struggles) | $\beta \to \beta_{\max}$ | Strong guidance from reference |

---

## 3. AdaRePO Extension 2: Priority-Weighted Learning

Standard GRPO treats all prompts equally. However, not all prompts are equally informative for learning. AdaRePO assigns a **priority weight** $w^{(q)}$ to each prompt $q$ that multiplies the advantages, focusing gradient on the most informative samples.

### 3.1 Variance Signal

High reward variance across generations indicates the model is **uncertain** — sometimes generating good molecules, sometimes bad. These prompts have the highest learning potential:

$$w_{\text{var}}^{(q)} = \sigma\left(\lambda \cdot \text{std}\{r_1^{(q)}, \ldots, r_G^{(q)}\}\right)$$

where $\lambda = 2.0$ is the variance scale and $G$ is the number of generations per prompt.

### 3.2 Frontier Signal

Prompts near the **learning frontier** (intermediate mean reward) are more valuable than trivially easy ($\bar{r} \approx 1$) or impossibly hard ($\bar{r} \approx 0$) ones. We use a Gaussian window centered on the frontier:

$$w_{\text{front}}^{(q)} = \exp\left(-\frac{(\bar{r}^{(q)} - \mu_c)^2}{2\delta^2}\right)$$

where $\bar{r}^{(q)}$ is the mean reward for prompt $q$, $\mu_c = 0.3$ is the frontier center, and $\delta = 0.3$ is the width.

### 3.3 Combined Priority Weight

$$\tilde{w}^{(q)} = w_{\text{var}}^{(q)} \cdot w_{\text{front}}^{(q)}$$

Normalize to preserve gradient magnitude and apply a floor:

$$w^{(q)} = \max\left(w_{\min},\; \frac{\tilde{w}^{(q)}}{\frac{1}{B}\sum_{q=1}^{B}\tilde{w}^{(q)}}\right)$$

where $w_{\min} = 0.2$ prevents any sample from being fully ignored, and $B$ is the batch size.

### 3.4 Weighted Advantages

The priority weights multiply the GRPO advantages:

$$\hat{A}_i^{(q)} = w^{(q)} \cdot A_i^{(q)}$$

---

## 4. Complete AdaRePO Objective

Combining dynamic beta and priority weighting:

$$\mathcal{L}_{\text{AdaRePO}} = \underbrace{-\frac{1}{|\mathcal{B}|}\sum_{q,i} \hat{A}_i^{(q)} \cdot \frac{\pi_\theta(y_i \mid x)}{\pi_{\text{old}}(y_i \mid x)}}_{\text{Priority-weighted RL loss}} + \underbrace{\beta \cdot \mathcal{L}_{\text{guide}}(m^*)}_{\text{Adaptive guidance}}$$

---

## 5. Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Beta max | $\beta_{\max}$ | 1.0 | Maximum guidance strength |
| Sigmoid sharpness | $\alpha$ | 5.0 | Transition speed for dynamic beta |
| Top-k fraction | $k/G$ | 0.33 | Fraction of generations for $v_{\text{top}}$ |
| Variance scale | $\lambda$ | 2.0 | Sensitivity to reward variance |
| Frontier center | $\mu_c$ | 0.3 | Reward value at the learning frontier |
| Frontier width | $\delta$ | 0.3 | Gaussian width of the frontier window |
| Min weight | $w_{\min}$ | 0.2 | Floor on priority weights |
| Num generations | $G$ | 3 | Generations per prompt (3 train + 1 vLLM GPU) |
| Learning rate | lr | 5e-6 | With cosine schedule + 15% warmup |
| Gradient accumulation | — | 16 | Effective batch size = 96 |
| Training epochs | — | 4 | ~60 steps per epoch |

---

## 6. QED Experiment Results

**Setup:** Qwen2.5-3B-Instruct, 4 epochs on OpenMolIns QED (light), 4× A40, DeepSpeed ZeRO-3

### Training Dynamics

| Metric | Start (Step 1) | End (Step 60) | Change |
|--------|---------------|---------------|--------|
| Total Loss | 0.621 | 0.139 | ↓78% |
| Guidance Loss | 1.757 | 0.509 | ↓71% |
| Reward | 0.168 | 0.175 | +4% |
| KL Divergence | 0.000 | 0.254 | moderate |
| $\beta_{\text{guide}}$ | 0.452 | 0.499 | stable ~0.49 |
| Priority Weight Std | 0.142 | 0.120 | active weighting |

### Evaluation (TOMG-Bench, N=5000)

| Metric | AdaRePO QED |
|--------|-------------|
| **Validity** | 68.3% |
| **Success Rate** | 9.5% |
| **Similarity** | 0.901 |
| Valid molecules | 3413 / 5000 |
| Successful optimizations | 475 / 5000 |

### Key Observations

1. **Loss convergence:** Total loss dropped 78%, guidance loss dropped 71% — model learned effectively
2. **No mode collapse:** KL stayed moderate (peaked 0.69, settled ~0.25)
3. **Dynamic β worked:** Stabilized around 0.49, indicating balanced RL + guidance regime. The model matched reference quality ($v_{\text{top}} \approx v_{\text{ref}}$)
4. **Priority weighting active:** Weight std ~0.13 throughout training — different prompts received meaningfully different gradient focus
5. **High structural preservation:** Similarity 0.901 — model makes minimal edits
6. **Room for improvement:** 9.5% success rate suggests longer training, larger dataset, or multi-property objectives could help
