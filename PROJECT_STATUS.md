# Project Status Tracker

**Last Updated**: 2026-05-07 20:20 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6B - FIXING EDGE COLLAPSE, RE-TRAINING NEEDED

### Summary
The first RLAIF training run (10 epochs on vast.ai RTX 4090) reported "100% SVR" but this metric was **misleading** — it measured `reward > 0` (partial credit), not all-5-laws-pass. Independent evaluation revealed the model collapsed to generating **0-1 edges** (empty graphs) that trivially pass out-degree and terminal sink but fail orphan/acyclic checks. **True SVR was 0% for active weights, 2.3% for EMA weights.**

### Root Cause: Reward Hacking via Edge Collapse
1. Sharpness loss (weight 0.5) rewarded pushing ALL presence logits to 0 or 1 — pushing to 0 (no edges) was the path of least resistance
2. Orphan loss (weight 0.3) was too weak to counteract sharpness
3. SVR metric used `reward > 0` instead of `GraphValidator.evaluate_batch`
4. KL clamped at 10.0 allowed the model to diverge far enough to reach the degenerate minimum

### Fixes Applied (commit 9586f28)
1. **Edge density loss** (weight 2.0): penalizes when total edges < valid node count
2. **Sharpness only on present edges**: no longer rewards collapsing to zero
3. **Reweighted losses**: orphan 0.3→1.0, sharpness 0.5→0.1
4. **Fixed SVR metric**: now uses `GraphValidator.evaluate_batch` (true all-5-laws pass rate)
5. **Per-law logging**: TensorBoard now tracks each law's pass rate individually

### Honest Evaluation Results (from evaluate_checkpoints.py)
| Metric | Baseline (E340) | RLAIF Active | RLAIF EMA |
|--------|-----------------|-------------|-----------|
| True SVR (all 5 laws) | 0.3% | 0.0% | 2.3% |
| Avg edges | 474 | 0-1 | 27 |
| Out-degree pass | 7% | 100% | 81% |
| Orphan pass | 100% | 6.7% | 75% |
| Acyclic pass | 99% | 13% | 96% |

### Published Blog Post
[Eradicating Syntax: The Neural Universal Machine](https://lawrenz.com/2026/05/05/eradicating-syntax-the-neural-universal-machine.html) — documents the Phase 5 results.

---

## 📋 NEXT ACTION CHECKLIST

### After Fixing Edge Collapse (NOW):
 **Re-train on vast.ai** with corrected structural loss
 - Density loss prevents empty-graph collapse
 - True SVR metric via GraphValidator  
 - Start from base DiT (epoch 340) — collapsed checkpoints are not useful
 
 **Evaluate new checkpoints** using `scripts/evaluate_checkpoints.py`
 - True all-5-laws SVR, per-law pass rates, edge statistics
 
 **Redo visualizations** once a valid checkpoint is found
 
 **Update blog post** with honest results

---

## �� QUICK REFERENCE

### Key Files
- **This file**: Overall project status and next actions
- **PROJECT_LOG.md**: Chronological development log and metric breakthroughs
- **README.md**: Current phase architecture summary
- **docs/neural-universal-machine-architecture.md**: Detailed math and DiT pipeline definitions
- **src/rlaif/**: RLAIF implementation (reward, structural loss, training script)
- **src/rlaif/train_rlaif.py**: Main training script with KL clamp, EMA, NaN guard, resume support
- **src/rlaif/structural_loss.py**: Differentiable structural constraints

### Key Checkpoints
- **Base DiT**: `checkpoints/num_dit_epoch_340.pt` (pre-RLAIF, ~30% naive SVR)
- **Best RLAIF**: `checkpoints/rlaif/vastai_run/rlaif_struct_epoch_7.pt` (100% SVR, lowest struct loss)
- **Final RLAIF**: `checkpoints/rlaif/vastai_run/rlaif_struct_epoch_10.pt` (100% SVR, includes EMA)
- All 10 epoch checkpoints in `checkpoints/rlaif/vastai_run/`

### Key Commands
```bash
# RLAIF Training (with all stabilization)
python -m src.rlaif.train_rlaif \
  --checkpoint checkpoints/num_dit_epoch_340.pt \
  --lr 1e-6 --kl-clamp 10.0 --max-epochs 10 \
  --batch-size 8 --grad-steps 1 --save-every 1

# Resume from checkpoint
python -m src.rlaif.train_rlaif \
  --checkpoint checkpoints/num_dit_epoch_340.pt \
  --resume checkpoints/rlaif/vastai_run/rlaif_struct_epoch_7.pt

# Visualization
python -m scripts.visualize_denoising --checkpoint checkpoints/rlaif/vastai_run/rlaif_struct_epoch_7.pt

# Pre-training (Phase 5, complete)
python src/train.py

# Run tests
python -m pytest tests/ -v
```

### RLAIF Hyperparameters (Final)
| Parameter | Value | Notes |
|-----------|-------|-------|
| β_kl (KL weight) | 1.0 | Anchors to reference model |
| β_struct (structural weight) | 0.1 | Structural loss coefficient |
| KL clamp | 10.0 | Prevents explosive KL |
| LR | 1e-6 | Low for stable RLAIF |
| EMA decay | 0.999 | Smoothed weight tracking |
| Grad steps | 1 | Last ODE step only (memory) |
| Batch size | 8 | On 24GB GPU (4 on 8GB) |
| ODE steps | 20 | Full denoising trajectory |

### Key Questions to Ask AI Assistant
- "What phase am I in?" → Phase 6B: Fixing edge collapse, re-training needed
- "What should I do next?" → Re-train with corrected loss on vast.ai
- "Is pre-training complete?" → Yes, Epoch 343, 100% SVR with Constraint Solver
- "Best checkpoint?" → Base DiT `checkpoints/num_dit_epoch_340.pt` (previous RLAIF checkpoints are collapsed)
