# Table 1: Single-Objective Molecular Optimization on TOMG-Bench

**Setup**: Qwen2.5-3B-Instruct, OpenMolIns light (1500 samples per task group), 4 epochs.
**Evaluation**: TOMG-Bench test set, metrics = Success Rate (SR), Similarity (Sim), SR×Sim.

> Mirrors Table 1 from RePO (arXiv:2603.05900). Two separate models are trained:
> one for property-based tasks (LogP/MR/QED) and one for structure-based tasks (Add/Del/SubComponent).

---

## A. Structure-Based Optimization (MolEdit)

| Method | Add SR | Add Sim | Add SR×Sim | Del SR | Del Sim | Del SR×Sim | Sub SR | Sub Sim | Sub SR×Sim |
|--------|--------|---------|------------|--------|---------|------------|--------|---------|------------|
| Qwen2.5-3B (zero-shot)     | 0.123 | 0.573 | 0.071 | 0.250 | 0.601 | 0.150 | 0.151 | 0.657 | 0.099 |
| **RePO paper**†             | 0.307 | 0.778 | 0.239 | — | — | — | — | 0.802 | — |
| **RePO** (our repro)        | 0.408 | 0.722 | 0.295 | 0.339 | 0.752 | 0.255 | 0.502 | 0.752 | 0.377 |
| **AdaRePO** (ours)          | **0.458** | 0.718 | **0.329** | **0.389** | 0.754 | **0.293** | **0.600** | 0.760 | **0.456** |

*† Paper-reported values (partial — extracted from text, full table is in PDF Figure).*

---

## B. Property-Based Optimization (MolOpt)

| Method | LogP SR | LogP Sim | LogP SR×Sim | MR SR | MR Sim | MR SR×Sim | QED SR | QED Sim | QED SR×Sim | Avg SR×Sim |
|--------|---------|----------|-------------|-------|--------|-----------|--------|---------|------------|------------|
| Qwen2.5-3B (zero-shot)          | 0.309 | 0.628 | 0.194 | 0.249 | 0.630 | 0.157 | 0.222 | 0.613 | 0.136 | 0.162 |
| **RePO paper** (single-prop)†   | 0.379 | 0.684 | 0.259 | 0.314 | 0.645 | 0.203 | 0.312 | 0.756 | 0.236 | 0.233 |
| **RePO** (our repro, joint)     | 0.443 | 0.711 | 0.315 | 0.505 | 0.709 | 0.358 | 0.348 | 0.722 | 0.251 | 0.308 |
| AdaRePO v15.1 (+dynamic β)      | 0.489 | 0.735 | 0.359 | 0.504 | 0.713 | 0.359 | 0.328 | 0.744 | 0.244 | 0.321 |
| AdaRePO v16 (+AdaTemp+MemBank)   | **0.669** | 0.609 | **0.408** | **0.574** | 0.600 | **0.345** | 0.330 | 0.635 | 0.210 | **0.321** |
| AdaRePO v17 (+per-sample β fix)  | 0.531 | 0.615 | 0.327 | 0.445 | 0.620 | 0.276 | **0.338** | 0.642 | 0.217 | 0.273 |

**† RePO paper uses single-property training (one model per property) with sim/prop weights 0.3/0.7.**
**Our RePO repro uses joint 3-property training with same 0.3/0.7 weights.**
**All AdaRePO variants use 0.5/0.5 sim/prop weights.**

---

## Notes

### Confounds to address
1. **Reward weight mismatch**: RePO baseline = 0.3/0.7 (sim/prop), AdaRePO = 0.5/0.5. Higher similarity weight in AdaRePO naturally favors Sim at cost of SR.
2. **Single seed** — All DSI runs use seed=42, no error bars.

### Key observations
- **v16 achieves highest SR** on LogP (+51% over RePO) and MR (+14%), at the cost of lower similarity (~0.1 drop).
- **v15.1 achieves best SR×Sim balance** — modest SR gains with preserved similarity.
- **v17 underperforms v16** on SR despite higher training rewards — possible overfitting to training reward; similarity drops without compensating SR gains.
- **QED remains hard** — all methods plateau at ~33% SR, no method clearly dominates.
- **Joint training helps** — our RePO reproduction outperforms the paper's single-property RePO on MR (0.505 vs 0.314) and LogP (0.443 vs 0.379), likely positive transfer.

---

## Job Tracker

| Job ID | Description | Status |
|--------|-------------|--------|
| 832397 | v17 eval (MolOpt) | ✅ COMPLETED |
| 832527 | Zero-shot eval (MolOpt) | ✅ COMPLETED |
| 832539 | Zero-shot eval (MolEdit) | ✅ COMPLETED |
| 832541 | RePO structure training | ❌ FAILED (disk quota during ckpt save) |
| 832542 | AdaRePO v16-struct training | ❌ CANCELLED |
| **835534** | RePO structure training (resubmit) | ✅ COMPLETED |
| **835535** | AdaRePO v16-struct training (resubmit) | ✅ COMPLETED |
| **838829** | RePO struct eval | ✅ COMPLETED |
| **838830** | AdaRePO v16-struct eval | ✅ COMPLETED |

## Data sources
- RePO (our repro, MolOpt): `/net/scratch/caom/repo_project/outputs/eval_ckpt748/`
- v15.1 Retrain (MolOpt): `/net/scratch/caom/repo_project/outputs/eval_v15_1_retrain/`
- v16 (MolOpt): `/net/scratch/caom/repo_project/outputs/eval_v16/`
- v17 (MolOpt): `/net/scratch/caom/repo_project/outputs/eval_v17/`
- Zero-shot: `/net/scratch/caom/repo_project/outputs/eval_zeroshot/`
- RePO struct: `/net/scratch/caom/repo_project/outputs/eval_repo_struct/`
- AdaRePO v16-struct: `/net/scratch/caom/repo_project/outputs/eval_v16_struct/`
