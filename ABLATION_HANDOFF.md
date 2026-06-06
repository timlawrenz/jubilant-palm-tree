# Ablation Study Handoff

The RLAIF β-ratio ablation suite is ready for execution. 
This suite will test various `beta_struct` and `beta_recon` ratios to find the optimal configuration that prevents mode collapse (empty graphs) while maximizing Syntactic Validity Rate (SVR).

## Instructions for the Next Agent:

1. **Verify the environment**: Ensure the GPU is free and the virtual environment is active.
2. **Start the ablation grid**: 
   ```bash
   PYTHONPATH=. python scripts/ablation/run_rlaif_ablation.py
   ```
3. **Resuming**: The orchestrator is fully resumable. If it crashes or times out, simply run the exact same command again. It will automatically detect the latest checkpoint in `checkpoints/ablation/<name>/` and append `--resume <ckpt>` to the training command.
4. **Generate Plots**: Once complete (or mid-run to check progress), run:
   ```bash
   PYTHONPATH=. python scripts/ablation/plot_rlaif_ablation.py
   ```
   Check `ablation_plots/` for `svr_comparison.png` and `density_comparison.png`.
5. **Update PROJECT_STATUS.md**: Based on the plots, determine which configuration achieved the highest SVR without density collapse, and document the findings in `PROJECT_STATUS.md`.
