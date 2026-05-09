# Project Status Tracker

**Last Updated**: 2026-05-09 19:30 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6C — ACYCLIC LOSS TRAINING (RUN 5)

### Summary
Three critical validator bugs were discovered and fixed (commit e7cb6aa):
1. `_check_acyclic_data` returned inverted results (True for cyclic, False for acyclic)
2. `_check_in_degree` required exactly 1 data input for COND/LOOP (should be ≥1)
3. `_check_out_degree` disallowed COND with 1 exec out (valid in compressed format)

The training dataset now validates at **83.2% SVR** (was 0% with buggy validator). All prior SVR measurements were inflated.

### Corrected Evaluation (Run 4 checkpoints, fixed validator)
| Metric | Dataset | Baseline | Best Run4 |
|--------|---------|----------|-----------|
| **SVR** | **83.2%** | 0% | 0% |
| out_degree | 91.8% | 3.8% | 27.5% |
| in_degree | 89.8% | 2.5% | 2.5% |
| orphan | 100% | 100% | 96.2% |
| **acyclic** | **100%** | **2.5%** | **0%** |
| terminal | 100% | 25% | 70% |

**Key insight:** acyclic_data was the #1 bottleneck (0% pass rate) with NO structural loss term.

### Current Training: Run 5
- **Config:** β_struct=0.01, β_recon=1.0, β_kl=1.0, lr=1e-5, 10 epochs
- **New:** NOTEARS acyclic loss (tr(exp(A))-n via power series k=2..4, weight 2.0)
- **New:** Expanded in-degree loss covers STATE writes, targets ≥1 instead of ==1
- **Running on:** vast.ai RTX 4090 (ssh -p 32809 root@23.158.136.85)
- **Evacuating to:** checkpoints/rlaif/vastai_run5/

---

## 📋 NEXT ACTION CHECKLIST

### Active (Run 5):
- [ ] Monitor Run 5 training (10 epochs, ~14 hours)
- [ ] Evaluate checkpoints with corrected validator
- [ ] Compare acyclic_data pass rates vs Run 4 (expect significant improvement)

### After Run 5:
- [ ] If acyclic improves but SVR still 0%: tune β_struct or add per-law weighting
- [ ] If SVR > 0%: evaluate best checkpoint thoroughly (>200 samples)
- [ ] Update blog post with corrected results and validator bug discovery
- [ ] Create visualizations of improved graphs

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
- **Base DiT**: `checkpoints/num_dit_epoch_340.pt` (pre-RLAIF, 0% SVR with corrected validator)
- **Best Run 4**: `checkpoints/rlaif/vastai_run4/rlaif_struct_epoch_3.pt` (EMA, best out_degree/terminal but 0% acyclic)
- **Run 5 (in progress)**: `checkpoints/rlaif/vastai_run5/` (with acyclic loss)

### Key Commands
```bash
# RLAIF Training (current config — Run 5)
PYTHONPATH=. python3 src/rlaif/train_rlaif.py \
  --checkpoint checkpoints/num_dit_epoch_340.pt \
  --beta-struct 0.01 --beta-recon 1.0 --beta-kl 1.0 \
  --lr 1e-5 --max-epochs 10 --batch-size 4

# Pre-training (Phase 5, complete)
python src/train.py

# Run tests
python -m pytest tests/ -v
```

### RLAIF Hyperparameters (Run 5 — Current)
| Parameter | Value | Notes |
|-----------|-------|-------|
| β_kl (KL weight) | 1.0 | Anchors to reference model |
| β_struct (structural weight) | 0.01 | Low: reconstruction dominates |
| β_recon (reconstruction weight) | 1.0 | MSE to ground truth graph |
| KL clamp | 10.0 | Prevents explosive KL |
| LR | 1e-5 | Higher than Run 1-3 for faster learning |
| EMA decay | 0.999 | Smoothed weight tracking |
| Grad steps | 3 | Last 3 ODE steps with grad |
| Batch size | 4 | On RTX 4090 |
| ODE steps | 20 | Full denoising trajectory |

### Structural Loss Weights (Run 5)
| Component | Weight | Notes |
|-----------|--------|-------|
| out_degree | 1.0 | Exec out-degree per motif type |
| sharpness | 0.1 | Only on present edges |
| type_sharpness | 0.1 | Edge type decisiveness |
| terminal | 0.5 | At least one exit node |
| orphan | 1.0 | All nodes connected |
| data_in | 1.0 | COND/LOOP/STATE-write need ≥1 data input |
| **acyclic** | **2.0** | **NEW: NOTEARS tr(exp(A))-n, k=2..4** |
| density | 2.0 | Prevent edge collapse |

### Key Questions to Ask AI Assistant
- "What phase am I in?" → Phase 6C: Acyclic loss training (Run 5 active)
- "What should I do next?" → Monitor Run 5, evaluate checkpoints
- "What bugs were found?" → 3 validator bugs (acyclic inverted, in_degree too strict, out_degree too strict)
- "Best checkpoint?" → Run 4 E3 EMA for out_degree/terminal, but 0% acyclic with corrected validator
