"""
AdaRePO Training & Evaluation Summary — QED Subtask
Generates publication-quality figures for the AdaRePO experiment.
"""
import json
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Load training metrics ──
with open('/tmp/adarepo_qed_metrics.json') as f:
    metrics = json.load(f)

steps = [m['step'] for m in metrics]
losses = [m['loss'] for m in metrics]
rewards = [m['reward'] for m in metrics]
s_losses = [m['s_loss'] for m in metrics]
s_losses_w = [m['s_loss_weighted'] for m in metrics]
kl = [m['kl'] for m in metrics]
beta_guide = [m['beta_guide_mean'] for m in metrics]
v_gap = [m['v_top_minus_v_ref'] for m in metrics]
priority_std = [m['priority_weight_std'] for m in metrics]
comp_len = [m['completion_length'] for m in metrics]
epochs = [m['epoch'] for m in metrics]
lr = [m['learning_rate'] for m in metrics]

# ── Load evaluation detailed results ──
eval_path = '/net/scratch2/qinanh/agent_drug_discovery/adaptive_repo/evaluation_results/ada_repo_3B_QED/ada_repo_3B_QED/open_generation/MolOpt/QED_detailed_results.csv'
eval_data = []
with open(eval_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        eval_data.append(row)

# Parse eval data
validities = [row['validity'] == 'True' for row in eval_data]
successes = [int(row['success']) for row in eval_data]
similarities = [float(row['similarity']) for row in eval_data if row['validity'] == 'True' and float(row['similarity']) > 0]
qed_changes = [float(row['qed_change']) for row in eval_data if row['validity'] == 'True' and float(row['similarity']) > 0]

OUT_DIR = '/net/scratch2/qinanh/agent_drug_discovery/adaptive_repo/analysis/figures'
import os
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Figure 1: Training Curves (2x3 grid)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle('AdaRePO Training Curves — QED Optimization (4 Epochs, Qwen2.5-3B)',
             fontsize=14, fontweight='bold', y=0.98)

# 1a. Loss
ax = axes[0, 0]
ax.plot(steps, losses, color='#2196F3', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Total Loss')
ax.set_title('(a) Training Loss')
ax.grid(True, alpha=0.3)
# Add epoch markers
for e in [1, 2, 3]:
    idx = next((i for i, ep in enumerate(epochs) if ep >= e), None)
    if idx:
        ax.axvline(steps[idx], color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        ax.text(steps[idx], ax.get_ylim()[1]*0.95, f'E{e}', fontsize=8, color='gray', ha='center')

# 1b. Reward
ax = axes[0, 1]
ax.plot(steps, rewards, color='#4CAF50', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('Reward (smile_optimization)')
ax.set_title('(b) Mean Reward')
ax.grid(True, alpha=0.3)

# 1c. Guidance Loss (raw vs weighted)
ax = axes[0, 2]
ax.plot(steps, s_losses, color='#FF9800', linewidth=1.5, alpha=0.6, label='s_loss (raw)')
ax.plot(steps, s_losses_w, color='#F44336', linewidth=1.5, alpha=0.8, label='β·s_loss (weighted)')
ax.set_xlabel('Step')
ax.set_ylabel('Guidance Loss')
ax.set_title('(c) Guidance Loss: Raw vs Weighted')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 1d. Dynamic Beta & v_top - v_ref
ax = axes[1, 0]
ax2 = ax.twinx()
l1 = ax.plot(steps, beta_guide, color='#9C27B0', linewidth=1.5, alpha=0.8, label='β_guide')
l2 = ax2.plot(steps, v_gap, color='#00BCD4', linewidth=1.5, alpha=0.6, label='v_top − v_ref')
ax.set_xlabel('Step')
ax.set_ylabel('β_guide', color='#9C27B0')
ax2.set_ylabel('v_top − v_ref', color='#00BCD4')
ax.set_title('(d) Dynamic Beta & Reward Gap')
lines = l1 + l2
labels = [l.get_label() for l in lines]
ax.legend(lines, labels, fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

# 1e. KL Divergence
ax = axes[1, 1]
ax.plot(steps, kl, color='#E91E63', linewidth=1.5, alpha=0.8)
ax.set_xlabel('Step')
ax.set_ylabel('KL Divergence')
ax.set_title('(e) KL Divergence from Reference')
ax.grid(True, alpha=0.3)

# 1f. Priority Weight Std
ax = axes[1, 2]
ax.plot(steps, priority_std, color='#795548', linewidth=1.5, alpha=0.8)
ax.fill_between(steps, 0, priority_std, color='#795548', alpha=0.15)
ax.set_xlabel('Step')
ax.set_ylabel('Priority Weight Std')
ax.set_title('(f) Priority Weighting Variance')
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'{OUT_DIR}/training_curves.png', dpi=150, bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/training_curves.pdf', bbox_inches='tight')
print(f"Saved training_curves.png/pdf")

# ============================================================
# Figure 2: Evaluation Results (2x2 grid)
# ============================================================
fig2, axes2 = plt.subplots(2, 2, figsize=(13, 10))
fig2.suptitle('AdaRePO Evaluation — QED Optimization (TOMG-Bench, N=5000)',
              fontsize=14, fontweight='bold', y=0.98)

# 2a. Summary bar chart
ax = axes2[0, 0]
metrics_names = ['Validity', 'Success Rate', 'Similarity']
metrics_vals = [0.6826, 0.095, 0.9006]
colors = ['#4CAF50', '#2196F3', '#FF9800']
bars = ax.bar(metrics_names, metrics_vals, color=colors, alpha=0.85, edgecolor='white', linewidth=1.5)
for bar, val in zip(bars, metrics_vals):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{val:.1%}' if val < 1 else f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
ax.set_ylim(0, 1.1)
ax.set_ylabel('Score')
ax.set_title('(a) Overall Metrics')
ax.grid(True, axis='y', alpha=0.3)

# 2b. Similarity distribution (valid molecules only)
ax = axes2[0, 1]
ax.hist(similarities, bins=50, color='#FF9800', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(np.mean(similarities), color='red', linestyle='--', linewidth=1.5, label=f'Mean={np.mean(similarities):.3f}')
ax.set_xlabel('Tanimoto Similarity')
ax.set_ylabel('Count')
ax.set_title(f'(b) Similarity Distribution (N={len(similarities)} valid)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 2c. QED change distribution
ax = axes2[1, 0]
positive = [q for q in qed_changes if q > 0]
negative = [q for q in qed_changes if q < 0]
zero = [q for q in qed_changes if q == 0]
ax.hist(qed_changes, bins=60, color='#9C27B0', alpha=0.7, edgecolor='white', linewidth=0.5)
ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
ax.axvline(np.mean(qed_changes), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean={np.mean(qed_changes):.4f}')
ax.set_xlabel('QED Change (generated − original)')
ax.set_ylabel('Count')
ax.set_title(f'(c) QED Change Distribution')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
# Add annotation
n_pos = len(positive)
n_neg = len(negative)
n_zero = len(zero)
ax.text(0.98, 0.95, f'Improved: {n_pos}\nWorsened: {n_neg}\nUnchanged: {n_zero}',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8))

# 2d. Scatter: similarity vs QED change
ax = axes2[1, 1]
sim_arr = np.array(similarities)
qed_arr = np.array(qed_changes)
# Color by success
colors_scatter = ['#4CAF50' if q > 0 else '#F44336' for q in qed_changes]
ax.scatter(sim_arr, qed_arr, c=colors_scatter, alpha=0.3, s=8, edgecolors='none')
ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Tanimoto Similarity')
ax.set_ylabel('QED Change')
ax.set_title('(d) Similarity vs QED Change')
ax.grid(True, alpha=0.3)
# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=8, label=f'Improved ({n_pos})'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label=f'Worsened ({n_neg})')]
ax.legend(handles=legend_elements, fontsize=9, loc='lower left')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(f'{OUT_DIR}/evaluation_results.png', dpi=150, bbox_inches='tight')
fig2.savefig(f'{OUT_DIR}/evaluation_results.pdf', bbox_inches='tight')
print(f"Saved evaluation_results.png/pdf")

# ============================================================
# Figure 3: AdaRePO Mechanism Overview
# ============================================================
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 5))
fig3.suptitle('AdaRePO Adaptive Mechanisms — QED Experiment', fontsize=14, fontweight='bold', y=1.02)

# 3a. Beta evolution over training
ax = axes3[0]
ax.scatter(steps, beta_guide, c=epochs, cmap='viridis', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
ax.set_xlabel('Step')
ax.set_ylabel('β_guide')
ax.set_title('(a) Dynamic β Over Training')
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Epoch')
ax.grid(True, alpha=0.3)

# 3b. Priority std vs reward
ax = axes3[1]
ax.scatter(rewards, priority_std, c=steps, cmap='plasma', s=30, alpha=0.7, edgecolors='white', linewidth=0.5)
ax.set_xlabel('Mean Reward')
ax.set_ylabel('Priority Weight Std')
ax.set_title('(b) Priority Variance vs Reward')
cbar = plt.colorbar(ax.collections[0], ax=ax, label='Step')
ax.grid(True, alpha=0.3)

# 3c. Completion length evolution
ax = axes3[2]
ax.plot(steps, comp_len, color='#607D8B', linewidth=1.5, alpha=0.8)
ax.fill_between(steps, 200, comp_len, color='#607D8B', alpha=0.1)
ax.set_xlabel('Step')
ax.set_ylabel('Completion Length (tokens)')
ax.set_title('(c) Generation Length Over Training')
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig3.savefig(f'{OUT_DIR}/mechanism_analysis.png', dpi=150, bbox_inches='tight')
fig3.savefig(f'{OUT_DIR}/mechanism_analysis.pdf', bbox_inches='tight')
print(f"Saved mechanism_analysis.png/pdf")

# ============================================================
# Print text summary
# ============================================================
print("\n" + "="*70)
print("AdaRePO EXPERIMENT SUMMARY — QED Molecular Optimization")
print("="*70)
print(f"""
Model:          Qwen2.5-3B-Instruct + AdaRePO fine-tuning
Training:       4 epochs, 60 steps, ~3.3 hours on 4× NVIDIA A40
                DeepSpeed ZeRO-3 + vLLM inference (3 train + 1 gen GPU)
Dataset:        OpenMolIns QED subset (light scale)

── Training Dynamics ──
  Loss:         {losses[0]:.3f} → {losses[-1]:.3f}  (↓{(1-losses[-1]/losses[0])*100:.0f}%)
  Guidance:     {s_losses[0]:.3f} → {s_losses[-1]:.3f}  (↓{(1-s_losses[-1]/s_losses[0])*100:.0f}%)
  Reward:       {rewards[0]:.3f} → {rewards[-1]:.3f}
  KL:           {kl[0]:.4f} → {kl[-1]:.4f}
  β_guide:      {np.mean(beta_guide):.3f} ± {np.std(beta_guide):.3f} (adaptive)

── AdaRePO Extensions ──
  Dynamic Beta: sigmoid_gap mode, β_max=1.0, α=5.0
    → β adapts based on gap between best generation and reference
    → Mean β ≈ 0.49, indicating balanced RL + guidance regime
  Priority Learning: ENABLED
    → Weight std: {np.mean(priority_std):.3f} (samples get different focus)
    → Higher weight for prompts with high reward variance + frontier reward

── Evaluation (TOMG-Bench QED, N=5000) ──
  Validity:     68.3% (3413/5000 valid SMILES)
  Success Rate: 9.5%  (475/5000 improved QED in target direction)
  Similarity:   0.901 (high structural preservation)

  QED Changes:  {n_pos} improved, {n_neg} worsened, {n_zero} unchanged
  Mean ΔQ:      {np.mean(qed_changes):.4f}

── Key Observations ──
  1. Loss converged well (0.62 → 0.09), guidance loss dropped 83%
  2. KL remained moderate (0.22), no mode collapse
  3. Dynamic β stabilized around 0.49, indicating the model learned
     to match reference quality (v_top ≈ v_ref)
  4. High validity (68%) and similarity (0.90) show the model
     generates chemically valid, structurally similar molecules
  5. Success rate (9.5%) suggests room for improvement — longer
     training or multi-property objectives may help
""")
