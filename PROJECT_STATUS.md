# Project Status Tracker

**Last Updated**: 2026-05-05 17:30 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6 - RLAIF FINE-TUNING (RWR) 🚀

### What's Happening Right Now?
After 7 failed PPO training runs (KL explosion, flat SVR due to credit assignment issues), we pivoted to **Reward-Weighted Regression (RWR)** — a simpler REINFORCE-style approach that avoids the structural problems of per-step PPO on our 20-step ODE.

**RWR is currently training** (PID 955623, started 2025-07-05 22:02 UTC).

### Architecture: RLAIF-RWR
The 20-step Euler ODE generates a graph, we score it, then reinforce the good trajectories:
- **Rollout**: Run ODE with no grad, adding σ noise at each step
- **Reward**: GraphValidator's 5 Laws of Physics with weighted signal + 2.5× jackpot
- **Update**: Pick random subset of steps, backprop advantage-weighted MSE
- **KL anchor**: ||v_policy - v_ref||² — MSE vs frozen reference (β=0.01)
- **Key advantage over PPO**: single optimizer.step() per accumulation window, no ratio/clip machinery

### Why PPO Failed (7 runs)
1. All 20 ODE steps get identical scalar advantage — dilutes gradient signal
2. Per-batch normalization with batch_size=4 gives garbage signal
3. PPO ratio machinery adds complexity without helping when signal is weak
4. KL exploded or clip fraction collapsed to 0 in every configuration tried

### Published Blog Post
[Eradicating Syntax: The Neural Universal Machine](https://lawrenz.com/2026/05/05/eradicating-syntax-the-neural-universal-machine.html) — documents the full Phase 5 results.

### Next Immediate Action
- **Monitor RWR training**: `tail -f runs/rwr_training.log` or TensorBoard `runs/rlaif_rwr_*`
- Watch for SVR trending upward from ~30% baseline
- KL should stay < 0.1 (currently 0.003 at step 2 — healthy)
- If SVR improves: let it run, potentially tune sigma schedule
- If SVR flat after 50+ steps: investigate naive_discretizer as bottleneck

---

## 📋 NEXT ACTION CHECKLIST

### ⚡ ACTIVE: RLAIF Training
```bash
# Start DDPO-RLAIF training
python -m src.rlaif.train_rlaif --checkpoint checkpoints/num_dit_epoch_340.pt

# Monitor with TensorBoard
tensorboard --logdir=runs
```

### After RLAIF Convergence:
 **OPTION A: Evaluate Raw SVR**
 - Measure naive-threshold SVR (no Constraint Solver) on held-out motif sequences
 - Compare to baseline (pre-RLAIF naive SVR)
 
 **OPTION B: σ→0 Deterministic Evaluation**
 - Set σ=0 (purely deterministic ODE) and evaluate SVR
 - This is the ultimate test: can the model generate perfect graphs without any stochasticity?
 
 **OPTION C: Expand the Universal Motifs**
 - Analyze the `literal_pool` density from the 22k dataset
 - Upgrade heavily-used strings into dedicated macro-motifs

---

## 📚 QUICK REFERENCE

### Key Files
- **This file**: Overall project status and next actions
- **PROJECT_LOG.md**: Chronological development log and metric breakthroughs
- **README.md**: Current phase architecture summary
- **docs/neural-universal-machine-architecture.md**: Detailed math and DiT pipeline definitions
- **src/rlaif/**: DDPO-RLAIF implementation (reward, PPO, rollout buffer, training script)

### Key Commands
```bash
# RLAIF Training
python -m src.rlaif.train_rlaif --checkpoint checkpoints/num_dit_epoch_340.pt

# Pre-training (Phase 5, complete)
python src/train.py

# Validation Analysis
python scripts/analyze_tb.py

# Run tests
python -m pytest tests/ -v

# Execution Engine Demo
python src/execution_engine/demo.py
```

### RLAIF Hyperparameters
| Parameter | Value | Notes |
|-----------|-------|-------|
| σ (exploration) | 1.0 → 0.1 | Annealed via decay |
| β (KL) | 0.1 | Prevents mode collapse |
| ε (PPO clip) | 0.2 | Standard |
| LR | 1e-5 | 10× lower than pre-training |
| Rollout batch | 4 graphs | ×20 steps = 80 fwd passes |
| PPO epochs | 4 | Per rollout |
| KL early stop | 0.05 | Prevents destructive updates |

### Key Questions to Ask AI Assistant
- "What phase am I in?" → Phase 6: RLAIF (DDPO)
- "What should I do next?" → Run RLAIF training, monitor reward + SVR curves
- "Is pre-training complete?" → Yes, Epoch 343, 100% SVR with Constraint Solver
- "What is the VRAM budget?" → ~700MB active (300MB buffer + 200MB×2 DiTs)
