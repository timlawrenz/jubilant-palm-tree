# Git Workflow: Branch-to-Experiment Mapping

To prevent code and results from tangling, this repository uses a strict experiment-branching model.

## The `main` Branch
`main` is for **infrastructure, validated tools, and documentation only**.
- It contains the pre-training code, the `GraphValidator` (and its tests), the Execution Engine, and these docs.
- It **does not** contain highly volatile experimental loss functions or mid-run ablation scripts.

## The `exp/*` Branches
Every distinct research hypothesis gets an `exp/<name>` branch.
- **Example:** `exp/rlaif-structural-loss`, `exp/decode-time-solver`.
- All scripts, tensorboard logs, and metric CSVs for that experiment are committed *to that branch*. 
- When an experiment concludes (success or failure), a summary is written to `docs/02_EXPERIMENTS_AND_RESULTS.md` on `main`, but the messy code/logs stay frozen on the `exp/` branch.
- We never merge an `exp/` branch into `main` unless its code graduates to core infrastructure. Future agents will check out a fresh `exp/` branch from `main` to start a new experiment.
