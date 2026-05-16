# AdaRePO v15.1 Retrain (Full 4 Epochs) vs RePO — Final Comparison

## Key Result: AdaRePO v15.1 beats RePO by +4.2% on avg SR×Sim

| Task | RePO SR | v15.1 SR | RePO Sim | v15.1 Sim | RePO Val | v15.1 Val | RePO SR×Sim | v15.1 SR×Sim |
|------|---------|----------|----------|-----------|----------|-----------|-------------|--------------|
| LogP | 0.443 | **0.489** | 0.711 | **0.735** | 0.779 | **0.794** | 0.315 | **0.359** |
| MR | **0.505** | 0.504 | 0.709 | **0.713** | **0.769** | 0.773 | 0.358 | **0.359** |
| QED | **0.348** | 0.328 | 0.722 | **0.744** | 0.781 | **0.814** | **0.251** | 0.244 |
| **Avg** | 0.432 | **0.440** | 0.714 | **0.731** | 0.776 | **0.794** | 0.308 | **0.321** |

## Training Jobs
- Job 812647: Initial train, reached step 200, failed (disk overflow during ckpt save)
- Job 813476: Resume from ckpt-150, reached step ~673 (epoch 3.6), timeout at 12h
- Job 814309: Resume from ckpt-650, completed to step 748 (epoch 4.0), 1h40m
- Job 814353: Evaluation (generate predictions + evaluate), 16min

## Sources
- RePO DSI: `repo_propopt_791601.out`, `repo_propopt_resume_795397.out`
- AdaRePO v15.1 Retrain: `v15_1_retrain_{812647,813476,814309}.out`

## Generated Files
- `fig_reward_advantage.png` — Training reward, loss, advantage, KL curves
- `fig_rl_diagnostics.png` — Dynamic beta, guidance loss, completion length, grad norm
- `fig_eval_comparison.png` — Eval bar chart (SR, Sim, Val, SR×Sim)
- `eval_summary.csv` — Eval metrics table
- `training_summary.csv` — Training metrics at key steps
