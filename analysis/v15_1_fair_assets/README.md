# AdaRePO v15.1/v16 vs RePO - DSI Fair Comparison

Generated from local scratch training logs. Curves aggregate duplicate log rows by epoch and apply EMA smoothing for presentation readability.

## Sources

- RePO DSI: `/net/scratch/caom/repo_project/logs/repo_propopt_791601.out`, `/net/scratch/caom/repo_project/logs/repo_propopt_resume_795397.out`
- v15.1 Boosted: `/net/scratch/caom/repo_project/logs/v15_1_boosted_812287.out`
- v15.1 Retrain: `/net/scratch/caom/repo_project/outputs/ada_repo_dsi_v15_1_retrain/trainer_state.json`
- v16: `/net/scratch/caom/repo_project/outputs/ada_repo_dsi_v16/trainer_state.json`

## Eval Sources

- RePO: `/net/scratch/caom/repo_project/outputs/eval_ckpt748/predictions/checkpoint-748/open_generation/MolOpt`
- v15.1 Retrain: `/net/scratch/caom/repo_project/outputs/eval_v15_1_retrain/predictions/checkpoint-748/open_generation/MolOpt`
- v16: `/net/scratch/caom/repo_project/outputs/eval_v16/predictions/checkpoint-748/open_generation/MolOpt`

## Generated Files

- `fig_v15_1_fair_reward_advantage.png`
- `fig_v15_1_fair_rl_diagnostics.png`
- `fig_v15_1_v16_eval_srxsim.png`
- `training_summary.csv`
- `table_v15_1_v16_eval_comparison.csv`
- `table_v15_1_v16_eval_comparison.md`
