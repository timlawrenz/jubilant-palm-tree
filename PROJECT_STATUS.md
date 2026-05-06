# Project Status Tracker

**Last Updated**: 2026-05-05 17:30 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6 - RLAIF FINE-TUNING (Structural Loss) 🚀

### What's Happening Right Now?
After PPO (7 runs, all failed) and REINFORCE/RWR (flat SVR), we pivoted to **differentiable structural loss** — directly backpropagating structural constraint violations through the last ODE step. **This is working: SVR jumped from 30% to 70%.**

**Training is running** (PID 1096151, started 2025-07-05 23:14 UTC).

### Architecture: Differentiable Structural Loss
Instead of REINFORCE (sample + scalar reward + high-variance gradient), we compute differentiable versions of the 5 structural constraints on the continuous ODE output:
- **Execution out-degree**: soft degree must match motif type (1 for sequence, 2 for conditional, etc.)
- **Edge sharpness**: push presence logits away from 0.5 ambiguity zone
- **Terminal sink**: at least one node with near-zero exec out-degree
- **No orphans**: all nodes must have total degree ≥ 1
- **Data in-degree**: condition/loop nodes need exactly 1 data input
- **KL anchor**: MSE between policy velocity and frozen reference velocity

### Key Results (first 110 batches)
| Metric | Start | Current | Status |
|--------|-------|---------|--------|
| SVR | ~30% | **~70%** | ✅ **Improving!** |
| KL | 0.000 | 0.011 | ✅ Rock-solid |
| Struct Loss | ~56 | ~10 | ✅ Trending down |

### Why This Works (and REINFORCE didn't)
1. **Dense gradients**: every element of the output gets a gradient, not just a scalar advantage
2. **No noise**: no σ noise in rollout = no 98K-dimensional variance problem
3. **No credit assignment**: loss is computed directly on x_final, gradients flow through one model call
4. **Cheap**: ~11s/batch vs 82s/step for RWR — 7× faster per gradient update

### Published Blog Post
[Eradicating Syntax: The Neural Universal Machine](https://lawrenz.com/2026/05/05/eradicating-syntax-the-neural-universal-machine.html) — documents the full Phase 5 results.

### Next Immediate Action
- **Monitor training**: `tail -f runs/struct_training.log` or TensorBoard `runs/rlaif_struct_*`
- Watch for SVR continuing to climb toward >90%
- Structural loss should keep decreasing
- KL should stay < 0.1
- If SVR plateaus at ~70%: try increasing β_struct or reducing β_kl

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
