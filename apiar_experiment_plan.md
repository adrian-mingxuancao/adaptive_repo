# APIAR Experiment Plan

**Paper:** APIAR (Active Policy Improvement with Adaptive Reference) — extension of RePO for instruction-conditioned molecular optimization.
**Target venue:** NeurIPS / ICLR.
**Status:** Algorithm and appendix drafts done. Multi-seed runs of the main config currently in progress. Experiment section is a skeleton with placeholders.

This plan turns the current `experiments.tex` skeleton into a defensible submission. Tasks are ordered by priority and have explicit acceptance criteria so they can be executed and checked off independently.

---

## Conventions

- **Task IDs:** `E*` = experiments to run, `C*` = code changes, `W*` = writing / cleanup.
- **Priority:** P0 = blocker (paper not defensible without it), P1 = high value (meaningfully strengthens claims), P2 = polish.
- **Seeds:** Default to ≥3 seeds (5 if affordable) for any new run. Report mean ± std and 95% CI.
- **Reference checkpoint config:** Use the same backbone, optimizer, rollout budget, reward weights, and decoding settings as the existing RePO run. Any deviation must be flagged in the corresponding result table.
- **Result aggregation:** All numbers in tables come from a single, version-controlled aggregation script (see `C0`). No per-table re-aggregation.

---

## Cluster Strategy

The 40GB GPUs on Polaris cannot reproduce RePO's published Table 1 numbers (model + rollout config does not fit at full fidelity). DSI's 80GB A100s match the RePO setup. Cluster assignment is therefore driven by **whether a result must be apples-to-apples with RePO Table 1 absolute numbers**:

- **DSI (80GB)** — required for any result that enters the main results / ablation table or that compares absolute SR against RePO's published numbers. This includes: E1 (main + ablation), C1/C2 baselines (because they enter Table 1), E2 (because it conditions APIAR–RePO gap on a per-example property — both methods must be trained under the canonical config).
- **Polaris (40GB)** — fine for relative-comparison or mechanism-confirmation experiments where absolute SR levels do not need to match RePO's published numbers: E3 (sensitivity, looks at trends), E4 (wall-clock, independent measurement), E5 (post-hoc qualitative analysis on existing checkpoints), E7 (training dynamics, looks at curve shapes).

Any Polaris-trained run that ends up in the paper must include a footnote in its caption stating that absolute values are not directly comparable to Table 1 due to reduced rollout / batch config, and that the comparison of interest is the relative gap between methods trained under identical Polaris config.

---

## P0 — Blockers

### C0. Build a single result-aggregation script

- **Why:** Multi-seed numbers must be aggregated identically across all tables. Hand-aggregation will produce inconsistencies that reviewers catch.
- **Deliverable:** `scripts/aggregate_results.py` that reads per-seed run logs and produces one CSV per (table, benchmark, method) combination, with columns: `method, benchmark, subtask, metric, mean, std, ci95_lo, ci95_hi, n_seeds, seed_list`.
- **Acceptance:** Running the script regenerates every numeric cell currently `[TODO]` in `experiments.tex` once the underlying runs exist. Re-running it twice produces byte-identical CSVs.

### C1. Add Iterative SFT / ReST-style baseline

- **Why:** This is the most natural "poor man's APIAR" baseline. Without it, reviewers will argue the gains come from self-distillation in general, not from APIAR's specific mechanisms.
- **Spec:** Every $T$ training steps (suggest $T=$ epoch length, or matched to APIAR memory-bank update frequency), take the current policy, sample $N$ candidates per training example, keep the best-of-$N$ by reward (subject to validity), and run one SFT epoch on those targets. Resume RL. Repeat.
- **Deliverable:** A baseline runnable with the same launcher as RePO, using the same backbone / optimizer / rollout budget. Total compute should match APIAR within 10%.
- **Acceptance:** Produces seeds-aligned results for TOMG-Bench MolOpt LogP/MR/QED that can flow into `aggregate_results.py`.

### C2. Add offline-strengthened-reference RePO baseline

- **Why:** Disentangles "adaptive update during training" from "the reference is just stronger." If APIAR ≈ offline-strengthened RePO, the paper's contribution becomes "find good references," not "do it adaptively."
- **Spec:** Before training, for each training example, sample $M$ candidates from the *base* policy (suggest $M=64$ at higher temperature), select the highest-reward valid candidate that satisfies $Q$, and use it as a fixed reference $\tilde m_{\mathrm{ref},i}$ in place of the dataset reference. Train RePO with this $\tilde m_{\mathrm{ref},i}$. Everything else identical.
- **Deliverable:** Same as C1.
- **Acceptance:** Same as C1. Additionally, log the average reward gap $\bar R(\tilde m_{\mathrm{ref}}) - \bar R(m_{\mathrm{ref}})$ in `aggregate_results.py` so we can report how much we strengthened the reference.

### E1. Main results table — full factorization

- **Why:** Current `experiments.tex` has main and ablation tables with inconsistent variants. Reviewers want one consistent factorization.
- **Spec:** Run all four variants on TOMG-Bench MolOpt (LogP, MR, QED), 3+ seeds each:
  1. RePO (existing static reference)
  2. APIAR β-only (adaptive switching, fixed reference = dataset reference)
  3. APIAR bank-only (fixed $\beta=\beta_{\max}$, active reference updating)
  4. APIAR full
- **Plus baselines:** Base model, GRPO, RePO with offline-strengthened reference (C2), Iterative SFT (C1).
- **Deliverable:** Replaces the placeholder Table~\ref{tab:molopt_main}. Column structure stays the same. Both Table~\ref{tab:molopt_main} and Table~\ref{tab:ablation_results} should use this same set of variants (no asymmetric subsets).
- **Acceptance:** Every cell has a number with std/CI. Statistical significance test (paired bootstrap or paired t-test, 10k resamples) reported for APIAR-full vs RePO and APIAR-full vs strongest baseline; p-values in the table caption or a footnote.

### E2. Reference-quality-conditional analysis (merged robustness + failure-mode)

- **Why:** APIAR's central claim is that fixed-reference guidance becomes restrictive when the reference is far from optimal. The cleanest test of this is to condition the APIAR–RePO gap on a per-example measure of how much room there is to improve over the reference. The ZINC dataset already provides a golden molecule for each prompt, so the natural quality controller is `gap_i = R(golden_i; c_i) − R(m_ref_i; c_i)`. No artificial reference-degradation protocol is needed.
- **Spec:**
  - **Train once per method**, not once per quality bucket. Both APIAR-full and RePO are trained on the full training set under the canonical config (must run on DSI, since these results enter the main argument).
  - **Eval-time bucketing.** At evaluation, partition the test set into bins by `gap_i` quantile (e.g., quintiles: bottom 20% through top 20%). Report per-bin SR, SR×Sim, and the APIAR–RePO gap.
  - **Headline figure:** x-axis = `gap_i` quantile (or continuous scatter with a regression line and shaded CI), y-axis = APIAR–RePO SR gap. Central prediction: the gap grows with `gap_i`, going to zero in the bottom bin (reference is already near-golden, no headroom for the bank to do anything) and being largest in the top bin (reference is far from golden, bank gets activated).
  - **Confirm dataset construction with the dataset collaborator** before running: how is `golden_i` defined (theoretical optimum vs strong baseline vs human-curated reference)? This affects how the "gap" is interpreted in the writeup. If `golden_i` is itself noisy or limited-capacity, the bottom-bin behavior may not be "saturated" but "co-stuck in a local optimum" — discussion needs to acknowledge this.
- **Deliverable:**
  - Replaces Figure~\ref{fig:ref_quality} (`gap`-conditional APIAR–RePO comparison).
  - Replaces Table~\ref{tab:ref_robustness}: per-bin SR / SR×Sim / Improve-over-ref for both methods. Improve-over-ref defined as the fraction of test examples where the **single greedy decode** achieves higher reward than the dataset reference.
  - One additional row in the table: aggregate over all examples (matches Table~\ref{tab:molopt_main}, sanity check).
- **Acceptance:** Single trained checkpoint per method per seed (no retraining per bucket). Figure shows continuous trend with CI. Bottom and top bin behavior both reported honestly even if APIAR is not better in the bottom bin — that is the predicted behavior, not a weakness.

### W1. Clean up stale content in `experiments.tex`

- **Why:** Lines 178–367 contain commented-out content from a previous RL paper (RLPD/SAC, D4RL, Meta-World), a duplicate `\section{Experiments}` declaration, and ControlGPT/LinkerGPT toolkit text. The duplicate section header in particular is dangerous if uncommented accidentally.
- **Deliverable:** `experiments.tex` ends cleanly after the last APIAR-relevant subsection. All commented-out non-APIAR content is removed (move to a separate `_archive.tex` if desired, do not keep inline).
- **Acceptance:** `grep -i "rlpd\|d4rl\|meta-world\|controlgpt\|linkergpt" experiments.tex` returns nothing. Only one `\section{Experiments}` declaration in the file.

### W2. Fix TOMG-Bench citation

- **Why:** Line 10 cites `\citep{li2026repo}` for TOMG-Bench, but TOMG-Bench is an independent benchmark paper, not the RePO paper.
- **Deliverable:** Correct citation for TOMG-Bench added to bib; experiments.tex updated.
- **Acceptance:** TOMG-Bench cited correctly on every appearance. RePO citation reserved for the RePO paper itself.

---

## P1 — High value

### E3. Hyperparameter sensitivity sweep

- **Why:** APIAR introduces 6 new hyperparameters ($k, \alpha, \delta, K, \beta_{\min}, \beta_{\max}$). Without sensitivity analysis, the paper looks like it depends on careful tuning.
- **Spec:** Pick MolOpt-LogP as the test bed. Sweep three values for each of $\alpha$, $K$, $\delta$ (one-at-a-time, others at default). 1 seed per cell is acceptable for sensitivity (state this explicitly).
- **Deliverable:** Appendix table `tab:apiar-sensitivity` with one row per (hyperparam, value), columns SR / SR×Sim / Validity. Brief paragraph in main text section 4.3 referencing it.
- **Acceptance:** Table shows that no single hyperparameter dominates the gain — i.e., performance varies less across reasonable hyperparameter ranges than the gap between APIAR and RePO.

### E4. Compute / wall-clock comparison

- **Why:** APIAR adds memory-bank ops, per-instance $\beta$, and canonicalization. Without an explicit comparison, reviewers will assume overhead is large.
- **Spec:** On one MolOpt subtask, log per-step wall-clock time for RePO and APIAR-full (median over 100 steps after warmup). Same hardware, same rollout backend. Also log peak GPU memory.
- **Deliverable:** One sentence in section 4.1 (Setup) plus a small line in the appendix table: "APIAR adds X% wall-clock and Y% peak memory vs RePO under matched rollout budget."
- **Acceptance:** Numbers logged with hardware and software versions; reproducible from a single command.

### E5. Qualitative case studies

- **Why:** Molecule papers are expected to show molecules. Cheap to add, high reviewer-positivity.
- **Spec:** Pick 4–6 representative test examples where APIAR's memory bank was activated (i.e., $m^* \neq m_{\mathrm{ref}}$ at end of training). Show: source molecule, dataset reference, APIAR-discovered $m^*$, with property values for each. Prefer cases that span LogP, MR, and QED.
- **Deliverable:** New figure `fig:case_studies` with rendered SMILES (use RDKit). New subsection (~half page) "Qualitative analysis" between sections 4.5 and 4.6, OR appendix figure with brief main-text pointer if space is tight.
- **Acceptance:** Figure renders cleanly; each panel has source / ref / discovered SMILES with property deltas labeled.

### E6. (merged into E2)

The previous "failure-mode analysis" task is folded into E2: bucketing by `gap_i` quantile in E2 already exposes the regime where APIAR fails to help (low-gap / near-optimal reference). No separate experiment needed. Discussion in the paper should explicitly call out the bottom bin as the regime where APIAR ≈ RePO.

### E7. Training dynamics figure (Figure 4)

- **Why:** Already in the skeleton as a placeholder. Important for mechanism verification.
- **Spec:** 4 panels, all logged during training of APIAR-full and RePO on MolOpt-LogP:
  1. Reward (mean rollout reward) vs step.
  2. Total loss / policy loss vs step.
  3. Average $\beta_{\mathrm{guide}}$ across the batch vs step (APIAR only).
  4. Memory-bank usage: fraction of examples with $|\mathcal{B}(c_i)| > 0$ vs step, plus mean reward gap $\bar R(m^*) - \bar R(m_{\mathrm{ref}})$ when bank non-empty.
- **Deliverable:** Replaces placeholder Figure~\ref{fig:training_dynamics}.
- **Acceptance:** All four panels show curves with shaded variability across seeds. Panel (c) should visibly decrease over training; panel (d) should show the bank progressively filling.

### W3. Define $R$ and $Q$ per benchmark in the setup

- **Why:** Generalization section (4.5) uses MolEdit, MolCustom, MuMOInstruct without specifying what $R$ and $Q$ look like in each. MolCustom in particular has no source molecule, so the editing-style $Q$ doesn't apply.
- **Deliverable:** A small table or itemized block in the Setup section listing, for each benchmark: reward components, weights, validity criterion, $Q$ predicate (or "$Q \equiv \mathrm{Valid}$" if no admissibility constraint applies).
- **Acceptance:** Every benchmark used in the experiments has explicit $R$ and $Q$ definitions traceable to a config file.

### W4. Add seeds / statistical-test commitment to Setup

- **Why:** Multi-seed protocol must be stated upfront.
- **Deliverable:** One paragraph in section 4.1 stating: number of seeds, how they're aggregated, what statistical test is used for significance, what the reported error bar represents (std, CI, or SE — pick one and stick with it).
- **Acceptance:** Same protocol cited consistently across every table caption.

---

## P2 — Polish

### W5. Rewrite predictive Discussion paragraphs after results land

- Each `Discussion.` paragraph in the current draft is written in the future tense ("we expect..."). Mark each with `% TODO: rewrite after results` and revisit once numbers exist. Discussion should describe what was *observed*, not what was *expected*.

### W6. Map every research question to a section

- RQ (i)–(iv) listed in the intro paragraph. Currently RQ (i)→4.2, (ii)→4.3, (iii)→4.4, (iv)→4.5; section 4.6 (dynamics) and any new failure-mode section don't map to a stated RQ. Either add an RQ (v) "does the adaptive mechanism behave as designed" or fold the dynamics analysis into the relevant earlier subsection.

### W7. Tighten the Generalization section or move to appendix

- Currently a 4-column aggregate table — minimal information density. Either expand with subtask breakdowns and discussion (becomes a real section), or compress to a paragraph with the table moved to appendix. Decide based on how strong the results are.

### W8. Precise definition of all metrics in Setup

- Validity (RDKit `MolFromSmiles` non-None? Pass all of: parse, sanitize, no fragmentation?). Tanimoto similarity (Morgan radius? bit length?). Make sure the spec matches the prior work being compared against.

---

## Suggested execution order

```
[DONE]   Multi-seed runs of base APIAR + RePO
[DONE]   C0  aggregate_results.py             (quick_results.py working)
[DONE]   C1  Iterative SFT baseline           (3 seeds trained + evaluated)
[DONE]   C2  offline-strengthened RePO        (preprocess + 3 seeds trained + evaluated)
[DONE]   E1  main + ablation factorization    (all 3-seed evals complete, see below)
         ⤷  MISSING: statistical significance tests (paired bootstrap / t-test)
[next]   E2  reference-quality-conditional analysis (gap-bucketed eval)
         E7  training dynamics figure         (partial: apiar_training_dynamics_main.png exists)
[then]   E3  hyperparam sensitivity           ┐
         E4  wall-clock comparison            ├ (parallel, can run on Polaris)
         E5  qualitative case studies         ┘
[then]   W1  cleanup experiments.tex          ┐
         W2  fix TOMG-Bench citation          │
         W3  R/Q per benchmark                │
         W4  seeds/stats protocol in setup    ├ (writing tasks, parallel)
         W5  rewrite Discussion paragraphs    │
         W6  RQ mapping                       │
         W7  generalization section decision   │
         W8  metric definitions               ┘
```

---

## Tracking

Use the checkboxes below as the agent makes progress. Mark a task done only after the deliverable exists *and* the acceptance criterion is met.

- [x] C0 — aggregation script (quick_results.py + aggregate_results.py)
- [x] C1 — Iterative SFT baseline (3 seeds: s42/s123/s456 trained + eval complete)
- [x] C2 — offline-strengthened RePO baseline (preprocess done, 3 seeds trained + eval complete)
- [x] E1 — main + ablation factorization (all variants × 3 subtasks × 3 seeds — eval complete Apr 30)
  - [x] E1a — statistical significance tests (paired bootstrap: all comparisons p<0.001***)
- [x] E2 — reference-quality-conditional analysis (gap-bucketed eval — monotonic APIAR advantage confirmed)
- [ ] E3 — hyperparameter sensitivity sweep (Job 7107700, capacity queue, 9 configs)
- [ ] E4 — wall-clock + memory comparison (Job 7107702, capacity queue)
- [x] E5 — qualitative case studies (6 examples rendered, 2 per subtask)
- [x] E7 — training dynamics figure
  - [x] E7a — 4-panel layout with seed variability shading (e7a_training_dynamics_4panel.pdf)
- [ ] W1 — cleanup stale content in experiments.tex
- [ ] W2 — fix TOMG-Bench citation
- [ ] W3 — R/Q per benchmark in Setup
- [ ] W4 — seeds/stats protocol paragraph
- [ ] W5 — rewrite Discussion paragraphs after results
- [ ] W6 — RQ-to-section mapping
- [ ] W7 — Generalization section decision (expand or appendix)
- [ ] W8 — precise metric definitions

---

## Notes for the coding agent

- **Do not invent results.** If a run hasn't produced numbers yet, leave the cell as `[TODO]` and update the tracking checklist with what is and isn't blocked.
- **Log everything.** Every new run should write per-step metrics to a structured log file consumable by `aggregate_results.py`. No copy-pasting numbers.
- **Match compute.** When adding C1/C2 baselines, match total wall-clock to APIAR-full within ~10% (document any deviation). Reviewers will check whether baselines are under-trained.
- **One config file per experiment.** Every run referenced in a table should be reproducible from a single committed config. Tables in the paper should cite the config name in a footnote during draft phase (remove before submission).
- **Keep the original 3-seed APIAR/RePO runs.** Do not re-run them under a new protocol unless absolutely necessary. The new baselines (C1, C2) should slot into the same evaluation pipeline.
