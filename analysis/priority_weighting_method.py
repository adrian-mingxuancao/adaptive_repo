"""
Generate method description PDF for AdaRePO Priority-Weighted Learning.
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = '/net/scratch2/qinanh/agent_drug_discovery/adaptive_repo/analysis/figures'

fig, axes = plt.subplots(3, 1, figsize=(10, 16), gridspec_kw={'height_ratios': [1.1, 1.3, 0.6]})
for ax in axes:
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

# ── Page 1: Background + Dynamic Beta ──
ax = axes[0]
y = 0.97

ax.text(0.5, y, 'AdaRePO: Adaptive Reference-guided Policy Optimization\nwith Priority-Weighted Learning',
        fontsize=15, fontweight='bold', ha='center', va='top')
y -= 0.09

ax.text(0.0, y, '1. Background: RePO Objective', fontsize=12, fontweight='bold', color='#1a237e', va='top')
y -= 0.06
ax.text(0.02, y, 'RePO extends GRPO by adding a guidance loss toward a reference molecule $m^*$:',
        fontsize=10, va='top')
y -= 0.06
ax.text(0.3, y, r'$\mathcal{L}_{\mathrm{RePO}} = \mathcal{L}_{\mathrm{RL}} + \mathcal{L}_{\mathrm{guide}}$',
        fontsize=13, va='top')
y -= 0.07

ax.text(0.0, y, '2. Dynamic Beta Guidance', fontsize=12, fontweight='bold', color='#1a237e', va='top')
y -= 0.06
ax.text(0.02, y, 'AdaRePO modulates guidance strength with an adaptive $\\beta$ based on policy vs reference quality:',
        fontsize=10, va='top')
y -= 0.06
ax.text(0.2, y, r'$\mathcal{L}_{\mathrm{AdaRePO}} = \mathcal{L}_{\mathrm{RL}} + \beta \cdot \mathcal{L}_{\mathrm{guide}}$',
        fontsize=13, va='top')
y -= 0.07
ax.text(0.02, y, 'where $\\beta$ is computed per-batch via the sigmoid-gap schedule:', fontsize=10, va='top')
y -= 0.06
ax.text(0.2, y, r'$\beta = \beta_{\max} \cdot \sigma\!\left(-\alpha \cdot (v_{\mathrm{top}} - v_{\mathrm{ref}})\right)$',
        fontsize=13, va='top')
y -= 0.07
ax.text(0.02, y,
        '$v_{\\mathrm{top}}$: mean reward of top-$k$ generations per prompt\n'
        '$v_{\\mathrm{ref}}$: reward of the reference molecule $m^*$\n'
        '$\\alpha$: sigmoid sharpness (default 5.0),  $\\beta_{\\max}$: max guidance (default 1.0)',
        fontsize=9.5, va='top', linespacing=1.6)
y -= 0.13
ax.text(0.02, y,
        'Intuition: When $v_{\\mathrm{top}} > v_{\\mathrm{ref}}$ (policy beats reference), $\\beta \\to 0$ (pure RL).\n'
        'When $v_{\\mathrm{top}} < v_{\\mathrm{ref}}$ (policy struggles), $\\beta \\to \\beta_{\\max}$ (strong guidance).',
        fontsize=9.5, va='top', style='italic', color='#444', linespacing=1.6)

# ── Page 2: Priority Weighting ──
ax = axes[1]
y = 0.97

ax.text(0.0, y, '3. Priority-Weighted Learning', fontsize=12, fontweight='bold', color='#1a237e', va='top')
y -= 0.05
ax.text(0.02, y,
        'Standard GRPO treats all prompts equally. AdaRePO assigns a priority weight $w^{(q)}$\n'
        'to each prompt $q$, focusing gradient on the most informative samples.',
        fontsize=10, va='top', linespacing=1.5)
y -= 0.08

ax.text(0.02, y, 'Variance signal — high reward variance = model uncertain = high learning value:',
        fontsize=10, fontweight='bold', va='top')
y -= 0.05
ax.text(0.15, y, r'$w_{\mathrm{var}}^{(q)} = \sigma\!\left(\lambda \cdot \mathrm{std}\{r_1^{(q)}, \ldots, r_G^{(q)}\}\right)$',
        fontsize=13, va='top')
y -= 0.07

ax.text(0.02, y, 'Frontier signal — prompts near learning frontier (intermediate reward) are most valuable:',
        fontsize=10, fontweight='bold', va='top')
y -= 0.05
ax.text(0.15, y, r'$w_{\mathrm{front}}^{(q)} = \exp\!\left(-\frac{(\bar{r}^{(q)} - \mu_c)^2}{2\,\delta^2}\right)$',
        fontsize=13, va='top')
y -= 0.07

ax.text(0.02, y, 'Combined priority (normalized, with floor):', fontsize=10, fontweight='bold', va='top')
y -= 0.05
ax.text(0.15, y, r'$\tilde{w}^{(q)} = w_{\mathrm{var}}^{(q)} \cdot w_{\mathrm{front}}^{(q)}$',
        fontsize=13, va='top')
y -= 0.06
ax.text(0.15, y, r'$w^{(q)} = \max\!\left(w_{\min},\;\frac{\tilde{w}^{(q)}}{\frac{1}{B}\sum_{q}\tilde{w}^{(q)}}\right)$',
        fontsize=13, va='top')
y -= 0.07

ax.text(0.02, y, 'Weighted advantages:', fontsize=10, fontweight='bold', va='top')
y -= 0.045
ax.text(0.15, y, r'$\hat{A}_i^{(q)} = w^{(q)} \cdot A_i^{(q)}$', fontsize=13, va='top')
y -= 0.07

ax.text(0.02, y, 'Complete AdaRePO objective:', fontsize=10, fontweight='bold', va='top')
y -= 0.05
ax.text(0.08, y,
        r'$\mathcal{L} = -\frac{1}{|\mathcal{B}|}\sum_{q,i} \hat{A}_i^{(q)} \cdot \frac{\pi_\theta(y_i|x)}{\pi_{\mathrm{old}}(y_i|x)} + \beta \cdot \mathcal{L}_{\mathrm{guide}}(m^*)$',
        fontsize=13, va='top')
y -= 0.06
ax.text(0.12, y, '(priority-weighted RL)                              (adaptive guidance)',
        fontsize=9, va='top', style='italic', color='#666')
y -= 0.06

ax.text(0.02, y, 'Parameters:', fontsize=10, va='top')
y -= 0.04
params_text = ('$\\lambda=2.0$ (variance scale)    $\\mu_c=0.3$ (frontier center)    '
               '$\\delta=0.3$ (frontier width)    $w_{\\min}=0.2$ (floor)')
ax.text(0.05, y, params_text, fontsize=9.5, va='top')

# ── Page 3: Hyperparameter table ──
ax = axes[2]
y = 0.92
ax.text(0.0, y, '4. Hyperparameters (QED Experiment)', fontsize=12, fontweight='bold', color='#1a237e', va='top')
y -= 0.12

headers = ['Parameter', 'Symbol', 'Value']
rows = [
    ['Beta max', r'$\beta_{\max}$', '1.0'],
    ['Sigmoid sharpness', r'$\alpha$', '5.0'],
    ['Top-k fraction', r'$k/G$', '0.33'],
    ['Variance scale', r'$\lambda$', '2.0'],
    ['Frontier center', r'$\mu_c$', '0.3'],
    ['Frontier width', r'$\delta$', '0.3'],
    ['Min weight', r'$w_{\min}$', '0.2'],
    ['Num generations', '$G$', '3'],
    ['Learning rate', 'lr', '5e-6'],
    ['Grad accumulation', '—', '16'],
    ['Epochs', '—', '4'],
]

cols = [0.05, 0.4, 0.7]
# Header
for j, h in enumerate(headers):
    ax.text(cols[j], y, h, fontsize=10, fontweight='bold', va='top')
ax.plot([0.03, 0.85], [y - 0.04, y - 0.04], color='black', linewidth=0.8)
y -= 0.09
# Rows
for row in rows:
    for j, val in enumerate(row):
        ax.text(cols[j], y, val, fontsize=9.5, va='top')
    y -= 0.07

fig.tight_layout(pad=1.5)
fig.savefig(f'{OUT_DIR}/priority_weighting_method.pdf', bbox_inches='tight')
fig.savefig(f'{OUT_DIR}/priority_weighting_method.png', dpi=150, bbox_inches='tight')
print("Saved priority_weighting_method.pdf/png")
