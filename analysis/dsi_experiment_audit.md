# DSI Cluster Experiment Audit — April 26, 2026

## 1. Experiment Inventory

All DSI runs: single-seed (42), Qwen2.5-3B-Instruct, joint LogP+MR+QED (1500 samples, OpenMolIns light), 4 epochs (748 steps), 2 training GPUs + 1 vLLM GPU (A100 80GB).

### Training Runs

| Version | Config | Key Features | Steps | Status | Reward Weights |
|---------|--------|-------------|-------|--------|---------------|
| **RePO (vanilla)** | DSI_3B_PropertyOpt.yaml | Fixed β=1.0, no adaptive mechanisms | 748 | ✅ Complete | 0.3/0.7 ⚠️ |
| **v15.1 Boosted** | DSI_v15_1_LogP_boosted.yaml | sigmoid_gap β, no memory bank | 700 | ⚠️ Incomplete (700/748) | 0.5/0.5 |
| **v15.1 Retrain** | DSI_v15_1_LogP_boosted.yaml | Same config as Boosted, fresh run | 748 | ✅ Complete | 0.5/0.5 |
| **v16** | DSI_v16_adaptive_curriculum.yaml | sigmoid_gap β + adaptive temp + memory bank self-distillation | 748 | ✅ Complete | 0.5/0.5 |
| **v17** | DSI_v17_per_sample_beta.yaml | Same as v16 + per-sample β fix (Bug #1/#2) | 700 (resume pending) | ⏳ Job 820859 | 0.5/0.5 |

### Key Config Differences

| Feature | RePO | v15.1 | v16 | v17 |
|---------|------|-------|-----|-----|
| Dynamic β (sigmoid_gap) | ✗ | ✓ | ✓ | ✓ |
| Adaptive temperature | ✗ | ✗ | ✓ | ✓ |
| Memory bank self-distillation | ✗ | ✗ | ✓ | ✓ |
| Per-sample β weighting | ✗ | ✗ | ✗ (Bug: batch-mean) | ✓ (fixed) |
| Reward weights (sim/prop) | **0.3/0.7** | 0.5/0.5 | 0.5/0.5 | 0.5/0.5 |

⚠️ **Confound**: RePO baseline uses 0.3/0.7 reward weights while all AdaRePO variants use 0.5/0.5. This means AdaRePO vs RePO comparisons conflate the method change with the reward weight change.

---

## 2. Evaluation Results (MolOpt Benchmark, TOMG-Bench)

All evals on 5000 test molecules per subtask.

### Success Rate (↑ better)

| Method | LogP | MR | QED | **Avg** |
|--------|------|-----|-----|---------|
| RePO | 0.443 | 0.505 | **0.348** | 0.432 |
| v15.1 Boosted | 0.452 | 0.454 | 0.264 | 0.390 |
| v15.1 Retrain | 0.489 | 0.504 | 0.328 | 0.440 |
| **v16** | **0.669** | **0.574** | 0.330 | **0.524** |
| v17 | — | — | — | pending |

### Similarity (↑ better, structural preservation)

| Method | LogP | MR | QED | **Avg** |
|--------|------|-----|-----|---------|
| RePO | 0.711 | 0.709 | **0.722** | 0.714 |
| v15.1 Boosted | 0.721 | 0.724 | 0.725 | 0.723 |
| v15.1 Retrain | **0.735** | **0.713** | 0.744 | **0.731** |
| v16 | 0.609 | 0.600 | 0.635 | 0.615 |

### Validity (↑ better)

| Method | LogP | MR | QED | **Avg** |
|--------|------|-----|-----|---------|
| RePO | 0.779 | 0.769 | 0.781 | 0.776 |
| v15.1 Boosted | 0.710 | 0.690 | 0.734 | 0.711 |
| v15.1 Retrain | 0.794 | 0.773 | **0.814** | 0.794 |
| v16 | **0.788** | **0.751** | **0.815** | **0.785** |

### SR × Similarity (composite metric, ↑ better)

| Method | LogP | MR | QED | **Avg** |
|--------|------|-----|-----|---------|
| RePO | 0.315 | 0.358 | 0.251 | 0.308 |
| v15.1 Boosted | 0.326 | 0.329 | 0.191 | 0.282 |
| v15.1 Retrain | 0.359 | 0.359 | 0.244 | 0.321 |
| **v16** | **0.408** | **0.345** | 0.210 | **0.321** |

---

## 3. Training Reward Comparison (per-epoch mean)

| Epoch | RePO | v15.1 Retrain | v16 | v17 (partial) |
|-------|------|---------------|-----|---------------|
| 0 | 0.358 | 0.356 | 0.360 | 0.330 |
| 1 | 0.390 | 0.389 | 0.357 | 0.434 |
| 2 | 0.428 | 0.446 | 0.563 | 0.760 |
| 3 | 0.465 | 0.495 | 0.833 | 0.932 |
| Last-50 | 0.472 | 0.517 | 0.987 | 1.173 |

---

## 4. Known Issues & Confounds

### Critical
1. **Reward weight mismatch**: RePO uses 0.3/0.7 (sim/prop), all others use 0.5/0.5. Need a RePO baseline with 0.5/0.5 for fair comparison, OR re-run AdaRePO with 0.3/0.7.
2. **Single seed**: All DSI runs use seed=42 only. No error bars. Polaris runs have 3 seeds but are shorter (120 steps) and different scale.
3. **v15.1 Boosted incomplete**: Only reached step 700/748, eval was done on checkpoint-700 (not final).

### Design Limitations (audited in previous session)
4. **Bug #1 (v16)**: `beta_guide.mean()` collapsed per-prompt β to batch scalar — per-prompt adaptation was lost in loss computation. Fixed in v17.
5. **Bug #2 (v16, inherited from RePO)**: `s_loss` was batch-averaged before β weighting — prerequisite for Bug #1 fix. Fixed in v17.
6. **v_ref is constant**: v_ref = w_sim ≈ 0.5 for all prompts (Sim(m_ref, m_ref)=1, Δp=0). Not a code bug but a mathematical property of the reward function. β still adapts via v_top variance.

### Observations
7. **v16 trades similarity for success rate**: Similarity drops ~0.1 vs baselines, but SR improves dramatically on LogP (+0.18) and MR (+0.07). On SR×Sim composite, v16 wins LogP but is mixed elsewhere.
8. **QED is hard**: All methods plateau at ~30-35% success rate on QED. No method shows clear advantage.

---

## 5. What Exists vs What's Needed for Paper

### ✅ Available on DSI (usable for paper)

| Asset | Status |
|-------|--------|
| RePO baseline (4 epoch, eval'd) | ✅ But wrong reward weights |
| v15.1 Retrain (sigmoid_gap β only, eval'd) | ✅ |
| v16 (full AdaRePO, eval'd) | ✅ |
| v17 (per-sample β fix) | ⏳ Training resume pending |
| Training curves for all versions | ✅ From logs |
| MolOpt eval (LogP/MR/QED) for RePO, v15.1, v16 | ✅ |
| Method formalization doc | ✅ docs/method_formalization.md |

### ❌ Missing for Paper

| Gap | Priority | Notes |
|-----|----------|-------|
| **Fair RePO baseline (0.5/0.5 weights)** | 🔴 Critical | Current baseline uses different reward weights |
| **v17 eval** | 🔴 High | Training almost done (700/748), then need eval |
| **Multi-seed runs on DSI** | 🟡 Medium | Have 3-seed on Polaris (shorter), but DSI is single-seed |
| **Zero-shot baseline (Qwen2.5-3B)** | 🟡 Medium | Needed as reference point |
| **Ablation: β-only vs β+temp vs β+temp+membank** | 🟡 Medium | v15.1=β only, v16=all three. But no "β+temp only" or "β+membank only" |
| **MolEdit / MolCustom eval** | 🟢 Nice-to-have | Only MolOpt eval exists |

---

## 6. Mapping to Polaris Experiments

| DSI (single seed, 748 steps) | Polaris (3 seeds, 120 steps) |
|------------------------------|------------------------------|
| RePO baseline (0.3/0.7) | v16v17 baseline (0.5/0.5) ✅ 3 seeds |
| v15.1 Retrain (sigmoid_gap) | — (no equivalent) |
| v16 (full AdaRePO) | v16_Polaris ❌ FAILED (walltime) |
| v17 (per-sample β) | v17_Polaris ❌ FAILED (walltime) |

**Polaris advantage**: Multi-seed with error bars.
**DSI advantage**: Full 4-epoch training, MolOpt eval complete.

---

## 7. Recommended Experiment Plan for Paper

### Option A: Minimal (use what we have)
- Main result table: v16 vs v15.1 Retrain vs RePO (note the reward weight caveat)
- Training curves figure: 4-version comparison
- Add v17 eval once done
- Use Polaris 3-seed baselines for error bars on RePO

### Option B: Fair comparison (recommended)
1. Re-run RePO baseline on DSI with 0.5/0.5 weights (1 job, ~12h)
2. Wait for v17 resume + eval
3. Main table: RePO (0.5/0.5) vs v15.1 (β only) vs v16 (full, buggy β) vs v17 (full, fixed β)
4. This gives a clean ablation: base → +dynamic β → +adaptive temp+membank → +per-sample β fix

### Option C: Full (if time permits)
- Option B + 3-seed runs for top methods + zero-shot baseline + MolEdit/MolCustom eval
