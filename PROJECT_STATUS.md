# Project Status Tracker

**Last Updated**: 2026-05-07 20:20 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6 - RLAIF COMPLETE ✅ 100% SVR ACHIEVED

### Summary
After PPO (7 runs, all failed) and REINFORCE/RWR (flat SVR), we pivoted to **differentiable structural loss** — directly backpropagating structural constraint violations through the last ODE step. After stabilization work (KL clamp, NaN guard, EMA), a full 10-epoch training run on a rented RTX 4090 (vast.ai) achieved **100% Structural Validation Rate sustained for 6 consecutive epochs**.

### Architecture: Differentiable Structural Loss
Instead of REINFORCE (sample + scalar reward + high-variance gradient), we compute differentiable versions of the 5 structural constraints on the continuous ODE output:
- **Execution out-degree**: soft degree must match motif type (1 for sequence, 2 for conditional, etc.)
- **Edge sharpness**: push presence logits away from 0.5 ambiguity zone
- **Terminal sink**: at least one node with near-zero exec out-degree
- **No orphans**: all nodes must have total degree ≥ 1
- **Data in-degree**: condition/loop nodes need exactly 1 data input
- **KL anchor**: MSE between policy velocity and frozen reference velocity (clamped at 10.0)

### Final Training Results (vast.ai RTX 4090, 10 epochs)
| Epoch | SVR | Struct Loss | KL | Time |
|-------|-----|-------------|-----|------|
| 1 | 86.8% | 60.88 | 0.12 | 61 min |
| 2 | 87.8% | 40.87 | 2.27 | 61 min |
| 3 | 89.2% | 30.98 | 4.06 | 61 min |
| 4 | 89.6% | 9.62 | 8.05 | 53 min |
| 5 | 100.0% | 0.89 | 10.00 | 48 min |
| 6 | 100.0% | 0.91 | 10.00 | 48 min |
| **7** | **100.0%** | **0.78** | 10.00 | 48 min |
| 8 | 100.0% | 0.90 | 10.00 | 49 min |
| 9 | 100.0% | 0.85 | 10.00 | 48 min |
| 10 | 100.0% | 0.92 | 10.00 | 49 min |

**Best checkpoint: Epoch 7** (lowest structural loss 0.78)  
**Total training time**: ~8.7 hours (~$3-4 on vast.ai)

### Why This Works (and REINFORCE didn't)
1. **Dense gradients**: every element of the output gets a gradient, not just a scalar advantage
2. **No noise**: no σ noise in rollout = no 98K-dimensional variance problem
3. **No credit assignment**: loss is computed directly on x_final, gradients flow through one model call
4. **Cheap**: ~7s/batch on 4090
5. **KL clamp**: prevents catastrophic divergence that caused NaN explosions on local GPU

### Key Stabilization Techniques
- **KL clamp at 10.0**: prevents explosive KL from destabilizing weights
- **NaN guard**: skips batches with NaN/Inf total loss before calling .backward()
- **EMA (0.999 decay)**: exponential moving average of weights for robust final model
- **Per-epoch checkpointing**: every epoch saved, enabling recovery from any failure

### Published Blog Post
[Eradicating Syntax: The Neural Universal Machine](https://lawrenz.com/2026/05/05/eradicating-syntax-the-neural-universal-machine.html) — documents the Phase 5 results.

---

## 📋 NEXT ACTION CHECKLIST

### After RLAIF Convergence (NOW):
 **OPTION A: Evaluate Raw SVR on Held-Out Data**
 - Load best checkpoint (epoch 7 or EMA weights from epoch 10)
 - Measure naive-threshold SVR (no Constraint Solver) on held-out motif sequences
 - Compare to baseline (pre-RLAIF naive SVR ~30%)
 
 **OPTION B: σ→0 Deterministic Evaluation**
 - Set σ=0 (purely deterministic ODE) and evaluate SVR
 - This is the ultimate test: can the model generate perfect graphs without any stochasticity?
 
 **OPTION C: Expand the Universal Motifs**
 - Analyze the `literal_pool` density from the 22k dataset
 - Upgrade heavily-used strings into dedicated macro-motifs

 **OPTION D: Upload Best Model to HuggingFace**
 - Push epoch 7 checkpoint and/or EMA weights
 - Update model card with RLAIF training details

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
- "What phase am I in?" → Phase 6: RLAIF complete, 100% SVR achieved
- "What should I do next?" → Evaluate on held-out data, upload to HuggingFace
- "Is pre-training complete?" → Yes, Epoch 343, 100% SVR with Constraint Solver
- "Best checkpoint?" → `checkpoints/rlaif/vastai_run/rlaif_struct_epoch_7.pt`
