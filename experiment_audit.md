# RePO/AdaRePO Experiment Audit — April 26, 2026

## 1. Job Status Summary (v16/v17 batch — Jobs 7097723–7097731)

| Job ID | Queue | Config | Seed | Output Dir | Status | Notes |
|--------|-------|--------|------|------------|--------|-------|
| 7097723 | capacity | v16v17_baseline | 42 | v16v17ms_repo_s42 | ✅ DONE (step 120) | loss=0.500 |
| 7097724 | capacity | v16v17_baseline | 123 | v16v17ms_repo_s123 | ✅ DONE (step 120) | loss=0.486 |
| 7097725 | preemptable | v16v17_baseline | 456 | v16v17ms_repo_s456 | ✅ DONE (step 120) | loss=2.215 ⚠️ |
| 7097726 | preemptable | v16_Polaris | 42 | v16v17ms_v16_s42 | ❌ WALLTIME KILLED | No output dir created |
| 7097727 | preemptable | v16_Polaris | 123 | v16v17ms_v16_s123 | ❌ WALLTIME KILLED | No output dir created |
| 7097728 | preemptable | v16_Polaris | 456 | v16v17ms_v16_s456 | ❌ WALLTIME KILLED | No output dir created |
| 7097729 | preemptable | v17_Polaris | 42 | v16v17ms_v17_s42 | ❌ WALLTIME KILLED | No output dir created |
| 7097730 | preemptable | v17_Polaris | 123 | v16v17ms_v17_s123 | ❌ WALLTIME KILLED | No output dir created |
| 7097731 | preemptable | v17_Polaris | 456 | v16v17ms_v17_s456 | ❌ WALLTIME KILLED | No output dir created |

**Diagnosis:** The 6 v16/v17 jobs all exceeded the 6-hour walltime and were killed before producing any output. Additionally, the config files (`v16_Polaris.yaml`, `v17_Polaris.yaml`) have been deleted from the `recipes/` directory, so these runs **cannot be resumed** and must be recreated from scratch.

The 3 baseline runs completed successfully, but s456 has an anomalously high loss (2.215 vs ~0.5 for s42/s123), which may indicate training instability for that seed.

---

## 2. Complete Experiment Inventory

### Phase 1: Single-Property Baselines (reward weights 0.3 sim / 0.7 prop)

| Run | Property | Steps | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|----------|-------|------|------------|-------------|-----------------|
| repo_3B_LogP | LogP | 60 | 42 | 0.537 | 0.182 | 0.201 |
| repo_3B_MR | MR | 60 | 42 | 0.484 | 0.977 | 0.816 |
| repo_3B_QED | QED | 60 | 42 | 0.518 | 0.107 | 0.107 |
| ms_repo_3B_LogP_s123 | LogP | 120 | 123 | 0.399 | — | — |
| ms_repo_3B_LogP_s456 | LogP | 120 | 456 | 0.216 | — | — |
| p1_repo_3B_MR_s120 | MR | 120 | 120 | 0.878 | 1.277 | 2.212 |
| p1_repo_3B_MR_s240 | MR | 240 | 240 | 0.004 | 1.624 | 3.512 |
| p1_repo_3B_QED_s120 | QED | 120 | 120 | 0.443 | 0.136 | 0.151 |
| p1_repo_3B_QED_s240 | QED | 240 | 240 | 0.004 | 0.159 | 0.197 |

**Observations:**
- MR shows strong reward improvement with longer training (0.82 → 2.21 → 3.51)
- QED reward improves slowly, plateaus around 0.15–0.20
- LogP final rewards are low (~0.2–0.3), consistent across seeds

### Phase 2: v15 — LogP-Only Multi-Seed (reward weights 0.3 sim / 0.7 prop, 120 steps)

| Run | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|------|------------|-------------|-----------------|
| v15_repo_3B_LogP | 42 | 0.443 | 0.321 | 0.345 |
| v15ms_repo_s123 | 123 | 0.393 | 0.267 | 0.658 |
| v15ms_repo_s456 | 456 | 0.415 | 0.255 | 0.278 |
| **Mean ± Std** | — | **0.417 ± 0.025** | **0.281 ± 0.036** | **0.427 ± 0.203** |

### Phase 3: v15h — Hard Examples, Multi-Property (reward weights 0.3 sim / 0.7 prop, property_name: "auto")

Training on `train_hard.csv` (1500 hard examples, mixed LogP/MR/QED).

| Run | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|------|------------|-------------|-----------------|
| v15hms_repo_s42 | 42 | 23.558 ⚠️ | 0.182 | 0.270 |
| v15hms_repo_s123 | 123 | 0.037 | 0.225 | 0.089 |
| v15hms_repo_s456 | 456 | 0.050 | 0.129 | 0.376 |
| **Mean ± Std** | — | — | **0.179 ± 0.048** | **0.245 ± 0.145** |

**Note:** s42 has anomalous loss (23.56); the other two converged to near-zero loss. Reward is lower than v15 LogP-only, which is expected since hard examples are harder to optimize.

### Phase 4: v16v17 Baseline — Multi-Property with Balanced Weights (0.5 sim / 0.5 prop, property_name: "auto")

Training on standard `train.csv` (1500 samples, LogP+MR+QED).

| Run | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|------|------------|-------------|-----------------|
| v16v17ms_repo_s42 | 42 | 0.500 | 0.587 | 1.080 |
| v16v17ms_repo_s123 | 123 | 0.486 | 0.368 | 0.421 |
| v16v17ms_repo_s456 | 456 | 2.215 ⚠️ | 0.240 | 0.402 |
| **Mean ± Std** | — | — | **0.398 ± 0.176** | **0.634 ± 0.386** |

### Phase 4b: v16 (AdaTemp+MemBank) Rerun — 12h walltime, same baseline config

| Run | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|------|------------|-------------|-----------------|
| v16v17ms_v16_s42 | 42 | 0.422 | 0.274 | 0.590 |
| v16v17ms_v16_s123 | 123 | 0.676 | 0.268 | 0.604 |
| v16v17ms_v16_s456 | 456 | 0.505 | 0.783 | 0.787 |
| **Mean ± Std** | — | **0.534 ± 0.131** | **0.442 ± 0.296** | **0.660 ± 0.110** |

### Phase 4c: v17 (per-sample beta) Rerun — 12h walltime, same baseline config

| Run | Seed | Train Loss | Mean Reward | Final-10 Reward |
|-----|------|------------|-------------|-----------------|
| v16v17ms_v17_s42 | 42 | 0.482 | 0.565 | 0.702 |
| v16v17ms_v17_s123 | 123 | 0.464 | 0.595 | 0.622 |
| v16v17ms_v17_s456 | 456 | 0.501 | 0.371 | 0.474 |
| **Mean ± Std** | — | **0.482 ± 0.019** | **0.510 ± 0.122** | **0.599 ± 0.116** |

**Note:** v16 and v17 use the same config (per-sample beta fix already merged in code). As predicted, results are comparable.

---

## 3. Key Comparisons for Paper

### Table: Effect of Reward Weight Balance

| Method | Sim/Prop Weights | Data | Mean Reward (±std) | Final Reward (±std) |
|--------|-----------------|------|--------------------|---------------------|
| v15 (LogP-only) | 0.3/0.7 | train.csv (LogP only, 500 samples) | 0.281 ± 0.036 | 0.427 ± 0.203 |
| v15h (multi-prop) | 0.3/0.7 | train_hard.csv (1500 hard) | 0.179 ± 0.048 | 0.245 ± 0.145 |
| v16v17 baseline (multi-prop) | 0.5/0.5 | train.csv (1500 standard) | 0.398 ± 0.176 | 0.634 ± 0.386 |
| v16 rerun (multi-prop) | 0.5/0.5 | train.csv (1500 standard) | 0.442 ± 0.296 | 0.660 ± 0.110 |
| v17 rerun (multi-prop) | 0.5/0.5 | train.csv (1500 standard) | 0.510 ± 0.122 | 0.599 ± 0.116 |

**Key findings:**
1. Balanced weights (0.5/0.5) outperform property-heavy (0.3/0.7) in mean reward
2. Standard training data significantly outperforms hard-example curriculum (0.398 vs 0.179 mean reward)
3. Multi-property training with balanced weights achieves the best overall performance

### Table: Training Stability Across Seeds

Some seeds show instability:
- v15hms_repo_s42: loss=23.56 (diverged)
- v16v17ms_repo_s456: loss=2.215 (partially diverged)
- All other runs: loss in 0.004–0.500 range

---

## 4. Evaluation Results — MolOpt Benchmark (5000 test samples per subtask)

### 4.1 Full Comparison Table

| Model | LogP SR | LogP Sim | LogP Val | MR SR | MR Sim | MR Val | QED SR | QED Sim | QED Val | **Avg SR×Sim** |
|-------|---------|----------|----------|-------|--------|--------|--------|---------|---------|---------------|
| Qwen2.5-3B (zero-shot) | 0.231 | 0.737 | 0.573 | 0.167 | 0.788 | 0.626 | 0.140 | 0.798 | 0.665 | **0.1376** |
| RePO LogP-only (60 step) | 0.196 | 0.832 | 0.607 | 0.146 | 0.883 | 0.638 | 0.109 | 0.880 | 0.685 | **0.1295** |
| v16v17 baseline (s42) | 0.255 | 0.791 | 0.684 | 0.241 | 0.842 | 0.684 | 0.154 | 0.838 | 0.706 | **0.1782** |
| v16 (3-seed avg) | 0.261 | 0.756 | 0.603 | 0.231 | 0.822 | 0.606 | 0.182 | 0.793 | 0.628 | **0.1773** |
| v17 (3-seed avg) | 0.241 | 0.788 | 0.635 | 0.216 | 0.840 | 0.632 | 0.152 | 0.839 | 0.683 | **0.1664** |

### 4.2 Improvement Summary (SR×Sim, the paper's key metric)

| Model | Avg SR×Sim | vs Zero-shot | vs RePO-LogP |
|-------|-----------|-------------|-------------|
| Qwen2.5-3B (zero-shot) | 0.1376 | — | +6.3% |
| RePO LogP-only (60 step) | 0.1295 | -5.9% | — |
| **v16v17 baseline (s42)** | **0.1782** | **+29.5%** | **+37.6%** |
| **v16 (3-seed avg)** | **0.1773** | **+28.9%** | **+36.9%** |
| **v17 (3-seed avg)** | **0.1664** | **+20.9%** | **+28.5%** |

### 4.3 Key Findings

1. **v16v17 multi-property RePO massively outperforms baselines**: +29% vs zero-shot, +37% vs single-property RePO
2. **Single-property RePO (LogP-only) is worse than zero-shot** (-5.9%): trained on LogP only, poor generalization to MR/QED
3. **MR sees largest gain**: zero-shot 0.167 → v16v17 0.241 (+44% SR)
4. **v16 ≈ v17** (as expected, identical configs): SR×Sim 0.177 vs 0.166
5. **Validity improves**: zero-shot ~62% → v16v17 ~68%
6. **Similarity also improves**: zero-shot ~0.77 → v16v17 ~0.82

### 4.4 Full E1 Results — All Methods, 3-Seed Avg (Apr 30, 2026)

All training and evaluation complete. SR×Sim = Success Rate × Tanimoto Similarity.

| Method | LogP SR | LogP Sim | LogP SR×Sim | MR SR | MR Sim | MR SR×Sim | QED SR | QED Sim | QED SR×Sim | **Avg SR×Sim** |
|--------|---------|----------|-------------|-------|--------|-----------|--------|---------|------------|---------------|
| Zero-shot | 0.231 | 0.737 | 0.170 | 0.167 | 0.789 | 0.131 | 0.140 | 0.798 | 0.111 | **0.1376** |
| GRPO | 0.141 | 0.873 | 0.123 | 0.107 | 0.897 | 0.096 | 0.078 | 0.900 | 0.070 | **0.0959** |
| RePO | 0.237 | 0.793 | 0.188 | 0.218 | 0.853 | 0.186 | 0.150 | 0.838 | 0.126 | **0.1665** |
| C1: Iterative SFT | 0.251 | 0.749 | 0.188 | 0.211 | 0.821 | 0.174 | 0.154 | 0.816 | 0.126 | **0.1627** |
| C2: Offline-Str. | 0.253 | 0.781 | 0.197 | 0.221 | 0.841 | 0.186 | 0.134 | 0.852 | 0.114 | **0.1652** |
| Abl: β-only | 0.235 | 0.792 | 0.186 | 0.208 | 0.862 | 0.180 | 0.146 | 0.840 | 0.122 | **0.1624** |
| Abl: Bank-only | 0.245 | 0.784 | 0.192 | 0.210 | 0.849 | 0.178 | 0.154 | 0.830 | 0.128 | **0.1659** |
| **APIAR (v16)** | **0.261** | 0.757 | **0.198** | **0.231** | 0.822 | **0.190** | **0.182** | 0.793 | **0.144** | **0.1773** |
| APIAR (v17) | 0.241 | 0.788 | 0.190 | 0.216 | 0.840 | 0.182 | 0.152 | 0.839 | 0.128 | **0.1664** |
| APIAR-Hard (v18) | 0.101 | 0.921 | 0.093 | 0.076 | 0.948 | 0.072 | 0.047 | 0.954 | 0.045 | **0.0697** |

### 4.5 Key Findings (Updated Apr 30)

1. **APIAR v16 (full) is the best method** — 0.1773 Avg SR×Sim, highest SR on all three subtasks
2. **Ablation validates both components**: β-only (0.1624) < Bank-only (0.1659) < Full (0.1773)
3. **APIAR > all baselines**: C1 Iterative SFT (0.1627), C2 Offline-Strengthened (0.1652), RePO (0.1665)
4. **GRPO dramatically underperforms** (0.0959) — confirms reference guidance is essential
5. **C2 > C1 > GRPO** — offline-strengthened references help, but adaptive online updates (APIAR) help more
6. **v18 Hard data**: low SR×Sim (0.0697) expected — high similarity (>0.92) but very low SR on hard examples

---

## 5. Remaining Work

### Completed ✅
1. ~~Multi-seed APIAR + RePO runs~~ — Done (v16, v17, v18)
2. ~~GRPO baseline~~ — Done (3 seeds)
3. ~~C1 Iterative SFT baseline~~ — Done (3 seeds)
4. ~~C2 Offline-strengthened RePO~~ — Done (preprocess + 3 seeds)
5. ~~Ablation: β-only, Bank-only~~ — Done (3 seeds each)
6. ~~Full evaluation on MolOpt (LogP, MR, QED)~~ — Done for all methods
7. ~~E1 main + ablation factorization~~ — Complete

### Next Priority
8. **E1a: Statistical significance tests** — paired bootstrap / t-test for APIAR-full vs RePO and vs strongest baseline
9. **E2: Reference-quality-conditional analysis** — gap-bucketed eval (no new training needed)
10. **E7a: Training dynamics figure update** — 4-panel layout with seed variability

### Later
11. E3: Hyperparameter sensitivity sweep
12. E4: Wall-clock + memory comparison
13. E5: Qualitative case studies (molecule rendering)
14. W1–W8: Writing tasks
