# Project Status Tracker

**Last Updated**: 2026-05-17 02:15 UTC  
**Purpose**: Quick orientation guide to understand current state and next actions

---

## 🎯 CURRENT STATUS: PHASE 6E — RUN 9 BLOCKED (OOM Crashes)

### Summary
Run 9 training on Vast.ai repeatedly failed due to CUDA Out-Of-Memory (OOM) errors during axial attention QKV projections. 
An AMP fix (bfloat16 autocast + `use_reentrant=True` for gradient checkpointing) has been implemented in the local working tree (`src/models/dit.py` and `src/rlaif/train_rlaif.py`) but remains **uncommitted** and has not yet produced a fully successful run.

**Immediate blockers**: 
1. **Critical Technical Debt**: The `GraphValidator` (which defines the SVR metric) lacks unit tests. Given a history of severe bugs in the validator skewing results, building a test suite is the highest priority before analyzing further SVR metrics.
2. Validate and commit the Run 9 AMP fix, and confirm successful training.

GPU usage during failed runs: crashed at >24GB.

### Summary
Run 8 completed all 10 epochs — first stable full training run with sharpened NOTEARS.
**Acyclic solved** (1.7% → 100%), but model collapses to near-empty graphs by epoch 5.
**Epoch 1 is the sweet spot**: SVR 7.1% (first confirmed non-zero SVR with corrected validator).

### Run 8 Evaluation Results (80 samples, corrected validator)
| Metric | Dataset | Baseline | **E1 Active** | E1 EMA | E5 Active | E10 Active |
|--------|---------|----------|---------------|--------|-----------|------------|
| **SVR** | 83.2% | 0% | **7.1% ±2.1** | 3.8% | 0% | 0% |
| out_degree | 91.8% | 5.4% | **100%** | 100% | 93.3% | 93.8% |
| in_degree | 89.8% | 3.8% | **43.8%** | 10.0% | 63.3% | 66.2% |
| orphan | 100% | 100% | 12.5% | 45.8% | 6.2% | 3.8% |
| **acyclic** | 100% | 1.7% | **71.7%** | 17.9% | **100%** | **100%** |
| terminal | 100% | 25.4% | **100%** | 97.9% | 96.7% | 93.8% |
| edges | ~15 | 486±1266 | **3±10** | 21±24 | 0±2 | 0±1 |

### Key Findings
1. **Sharpened NOTEARS works**: acyclic 1.7% → 71.7% (E1) → 100% (E5+) — completely solved
2. **Mode collapse**: model generates near-empty graphs by E5 (edges: 3→0) to trivially satisfy constraints
3. **SVR peaks early then collapses**: 7.1% at E1, 0% by E5 — the model over-optimizes structure at the expense of content
4. **KL saturation**: KL hits clamp (10.0) from E2 onward, unable to anchor model

### Root Cause of Collapse
The model discovered: fewer edges = easier to satisfy ALL structural constraints. With 0 edges:
- No cycles → acyclic=100%, no out-degree violations → out_degree=100%, etc.
- The density loss (weight=2.0) was insufficient to counteract this
- KL clamp at 10.0 couldn't prevent drift once saturated

---

## 🆕 Run 8 Remote Evaluation (RTX 4090, 50 samples, epochs 2-9)
### Completed: 2026-05-18
Full evaluation on remote RTX 4090 (epoch 10 was corrupted). Results saved to `eval_results_epoch2-9.txt`.

**Best checkpoint**: Epoch 9 EMA — SVR 6.0% ± 2.8%

| Epoch | Weights | SVR% | ± | Acyclic% | Term% | Edges | Orphan% |
|-------|---------|------|---|----------|-------|-------|---------|
| base | active | 0.0 | 0.0 | 2.0 | 27.3 | 201±337 | 100.0 |
| 2 | active | 4.7 | 2.5 | 71.3 | 98.0 | 2±2 | 13.3 |
| 2 | ema | 4.0 | 2.8 | 66.7 | 100.0 | 2±2 | 14.7 |
| 5 | ema | 6.0 | 3.3 | 71.3 | 100.0 | 2±2 | 17.3 |
| 7 | active | 4.0 | 2.8 | 75.3 | 100.0 | 1±2 | 13.3 |
| 9 | active | 4.7 | 0.9 | 76.7 | 99.3 | 1±2 | 15.3 |
| 9 | ema | **6.0** | **2.8** | **76.7** | **99.3** | 2±2 | 16.0 |

**Consistency with Run 8 local eval**: Confirms SVR peaks early then plateaus.
Edge count collapsed to 1-2 (from 201 baseline). Acyclic 65-77% (from 1.7% baseline).
Epoch 9 EMA is the recommended checkpoint.

---

## 📋 NEXT ACTION CHECKLIST

### Immediate:
- [ ] Add minimum edge density penalty to structural loss (penalize <5 edges)
- [ ] Consider early stopping or KL clamp increase to prevent collapse
- [ ] Run 9: test with edge density floor + potentially higher β_recon

### Best Checkpoint:
- **Run 8 E1 Active**: `checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt` — SVR 7.1%
- Consider deeper eval with 200+ samples to confirm

### Longer Term:
- [ ] If edge density floor prevents collapse: full 10-epoch training with SVR > 0%
- [ ] Update blog post with Run 8 results
- [ ] Create visualizations of epoch 1 graphs (first structurally valid generated graphs)

---

## 📚 QUICK REFERENCE

### Key Files
- **This file**: Overall project status and next actions
- **PROJECT_LOG.md**: Chronological development log and metric breakthroughs
- **README.md**: Current phase architecture summary
- **src/rlaif/train_rlaif.py**: Main training script with loss clamping, EMA, NaN guard, resume
- **src/rlaif/structural_loss.py**: Differentiable structural constraints (sharpened NOTEARS)

### Key Checkpoints
- **Base DiT**: `checkpoints/num_dit_epoch_340.pt` (pre-RLAIF, 0% SVR)
- **Best Run 8 E1**: `checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt` (SVR 7.1%, acyclic 71.7%)

### Key Commands
```bash
# RLAIF Training (Run 8 config)
PYTHONPATH=. python3 -m src.rlaif.train_rlaif \
  --beta-struct 0.2 --beta-recon 0.5 --lr 5e-6 --max-epochs 10 --batch-size 4

# Evaluate checkpoints
python3 -m scripts.evaluate_checkpoints \
  --checkpoints-dir checkpoints/rlaif/vastai_run8 --num-samples 80 --epochs "1,5,10"
```

### Training Run History
| Run | Epochs | SVR | Acyclic | Key Change | Issue |
|-----|--------|-----|---------|------------|-------|
| 5 | 10 | 0% | 1.2% | NOTEARS (weak beta=0.01) | Acyclic ineffective |
| 6 | 2 | 25%* | -- | Sharpened NOTEARS (beta=0.5) | Recon explosion, NaN crash |
| 7 | 1 | -- | -- | + recon clamp | Still NaN at batch 220 |
| **8** | **10** | **7.1%** | **100%** | + struct clamp, lr=5e-6, beta=0.2 | Mode collapse (edges->0) |
