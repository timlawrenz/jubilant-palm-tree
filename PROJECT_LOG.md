# Project Log: May 1, 2026

## Milestone 4: Pre-Training Completion & SVR Breakthrough
**Status:** Pre-training Concluded / Ready for RLAIF

### Summary of Findings:
The implementation of the Hybrid Loss (Optimal Transport MSE + Categorical Cross-Entropy) successfully resolved the Continuous/Discrete Representation Gap. The DiT was deployed on a 400-epoch, 3-Phase Curriculum to scale the topological routing up to massive 128-node graphs.

At Epoch 343, the model achieved a monumental breakthrough: **100.00% Syntactic Validity Rate (SVR)** on the 128-node validation batch. 

### Key Architectural Validations:
*   **The Judicial Constraint Solver:** Replaced naive probability thresholding with mathematically rigid Top-K arity snapping and logit-weighted branch conflict resolution. This proved essential in translating the continuous DiT heat-map into flawlessly executable, acyclic matrices.
*   **Dynamic Gradient Accumulation & Checkpointing:** The $O(N^2)$ VRAM explosion caused by the Axial Attention blocks processing $128 \times 128$ matrices was successfully mitigated without losing gradient stability.

### Next Immediate Technical Step:
The continuous Flow Matching loss definitively plateaued at ~0.135. The model has extracted maximum topological value from the passive continuous objective. Training was terminated early at Epoch 350 to conserve compute.

1.  **RLAIF Implementation:** Utilize the epoch 340 checkpoint to begin the Reinforcement Learning fine-tuning phase.
2.  **Dense Reward Structure:** Deploy PPO to actively punish the network for generating topological errors (e.g. orphans or dead-end terminal sinks) using the `GraphValidator` rules as a direct reward signal. Includes a KL Divergence Anchor to prevent mode-collapse.

---

## Milestone 5: Phase 6 — RLAIF via Differentiable Structural Loss
**Status:** Failed Run → Root Cause Analysis → Corrected Re-Training In Progress

### The PPO/REINFORCE Dead End (May 2–4)
Seven PPO attempts and multiple REINFORCE/RWR runs all failed:
- **PPO**: Computing exact log-likelihoods of 20-step ODE trajectories requires the trace of the Jacobian across all steps — computationally intractable, immediate OOM.
- **REINFORCE/RWR**: The scalar reward signal across a 98,304-dimensional continuous action space (128×128×6 matrix) produces catastrophically high variance. SVR remained flat at ~0.3%.
- **Key insight**: The "Gaussian Step" DDPO formulation (treating each ODE step's velocity prediction as the mean of a Gaussian, so log-prob ∝ −MSE) was theoretically sound but still too noisy for convergence.

### The Differentiable Structural Loss Breakthrough (May 4–5)
Instead of sampling + scalar reward, we compute **differentiable versions** of the 5 structural constraints directly on the continuous ODE output. Gradients flow through the last Euler step into the DiT weights.

**Loss components (v1):**
| Component | Weight | Purpose |
|-----------|--------|---------|
| Exec out-degree | 1.0 | Match motif-specific edge counts |
| Edge sharpness | 0.5 | Push presence logits away from ambiguity |
| Type sharpness | 0.3 | Push type logits toward one-hot |
| Terminal sink | 0.5 | Ensure ≥1 node with zero exec out-degree |
| Orphan prevention | 0.3 | Penalize nodes with total degree = 0 |
| Data in-degree | 0.5 | Condition/loop nodes need 1 data input |
| KL anchor | 1.0 | MSE between policy and frozen reference velocities |

**Stabilization techniques:**
- KL clamped at 10.0 to prevent explosive divergence (was causing NaN on local GPU)
- NaN guard: skip batches with NaN/Inf loss before `.backward()`
- EMA (0.999 decay) for robust weight averaging
- Per-epoch checkpointing for recovery

### The Vast.ai Training Run (May 7)
Rented an RTX 4090 (24GB, AMD EPYC) on vast.ai (~$0.40/hr). Local RTX 2070 (8GB) could not train without OOM at batch_size > 4.

- **Setup**: `git clone` + `pip install` + HuggingFace checkpoint download. Zero manual uploads needed.
- **Config**: batch_size=8, grad_steps=1, lr=1e-6, max_epochs=10
- **Data evacuation**: Local cron rsync every 15 min + completion watcher script
- **Runtime**: ~7s/batch, ~48–61 min/epoch, ~8.7 hours total (~$3-4)

**Training metrics showed rapid convergence:**
| Epoch | "SVR" | Struct Loss | KL |
|-------|-------|-------------|-----|
| 1 | 86.8% | 60.88 | 0.12 |
| 5 | 100.0% | 0.89 | 10.00 |
| 7 | 100.0% | 0.78 | 10.00 |
| 10 | 100.0% | 0.92 | 10.00 |

### The Collapse Discovery (May 8)
Independent evaluation using `scripts/evaluate_checkpoints.py` revealed **the "100% SVR" was a lie**:

**The misleading metric**: Training computed SVR as `(rewards > 0).float().mean()`, where `rewards` came from `compute_reward()` which gives partial credit. A graph passing only out_degree + terminal_sink scores `0.4 + 0.4 − 0.1 − 0.1 − 0.1 = +0.5 > 0` and counts as "valid."

**The true SVR** (using `GraphValidator.evaluate_batch`, requiring all 5 laws to pass):
| Metric | Baseline (E340) | RLAIF Active | RLAIF EMA |
|--------|-----------------|-------------|-----------|
| True SVR | 0.3% | **0.0%** | 2.3% |
| Avg edges | 474 | **0–1** | 27 |
| Orphan pass | 100% | 6.7% | 75% |

**Root cause chain:**
1. Sharpness loss (weight 0.5) rewards pushing ALL presence logits toward 0 or 1
2. Pushing to 0 (no edges) is the path of least resistance — easier than learning valid topology
3. Orphan loss (weight 0.3) is too weak to counteract sharpness (0.5)
4. Zero-edge graphs trivially pass out_degree and terminal_sink checks
5. KL clamped at 10.0 allowed the model to diverge far enough to reach this degenerate minimum
6. The misleading SVR metric masked the collapse entirely during training

This is a textbook case of **reward hacking** / **Goodhart's Law** in RL: the model optimized the measured metric (partial-credit reward > 0) by finding a degenerate shortcut (empty graphs) instead of learning the intended behavior (valid execution graphs).

### The Fix: Anti-Collapse Structural Loss v2 (May 8, commit 9586f28)

**Changes to `structural_loss.py`:**
1. **NEW — Edge density loss** (weight 2.0): `relu(n_valid_nodes − total_soft_edges)² / n_valid²`. Only penalizes *too few* edges, not too many. This directly prevents the empty-graph collapse.
2. **Sharpness now conditional**: only applied to edges with presence > 0.5. No longer rewards collapsing all edges to zero.
3. **Reweighted**: orphan 0.3→1.0, sharpness 0.5→0.1

**Changes to `train_rlaif.py`:**
1. **Fixed SVR metric**: replaced `(rewards > 0).float().mean()` with `GraphValidator.evaluate_batch` `perfect_graphs` — the ground truth all-5-laws pass rate
2. **Per-law logging**: TensorBoard now tracks each of the 5 laws individually
3. **Density component logged**: new `Structural/density` scalar in TensorBoard

**Updated loss weights (v2):**
| Component | Old Weight | New Weight | Rationale |
|-----------|-----------|-----------|-----------|
| Exec out-degree | 1.0 | 1.0 | Already working well |
| Edge sharpness | 0.5 | **0.1** | Was the primary collapse driver |
| Type sharpness | 0.3 | 0.1 | Reduced to match |
| Terminal sink | 0.5 | 0.5 | Unchanged |
| Orphan prevention | 0.3 | **1.0** | Must dominate sharpness |
| Data in-degree | 0.5 | 0.5 | Unchanged |
| **Edge density** | — | **2.0** | **New: prevents empty graphs** |

**Smoke test verification**: zero-edge graphs now incur density loss 0.81 + orphan loss 0.80 (total ~3.2 from anti-collapse terms alone), compared to near-zero penalty previously.

### Corrected Re-Training (May 8, in progress)
New vast.ai RTX 4090 instance. Training from base DiT (epoch 340) with corrected loss v2.

**Early results (first 17 batches)**:
- SVR at step 10: **12.5%** — already dramatically better than 0% in the collapsed run
- Density loss is actively engaging — model cannot collapse to zero edges
- ~7s/batch, ~57 min/epoch estimated

### Lessons Learned
1. **Never trust training metrics alone.** Always run independent evaluation with the ground-truth validator.
2. **Sharpness loss is dangerous.** Encouraging binary decisions (0 or 1) creates a trivial minimum at "all zeros." It must be conditional on edge existence.
3. **Loss weight ratios matter enormously.** Sharpness at 0.5 vs orphan at 0.3 meant the model was *incentivized* to kill edges.
4. **Edge density is a necessary structural constraint.** Without a floor on edge count, the model will always find the empty-graph shortcut.
5. **Ephemeral GPU workflow works.** The vast.ai + rsync evacuation pattern is reliable and cost-effective for ~10hr training runs.