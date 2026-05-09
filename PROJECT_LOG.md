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

### Run 2 Results: Density Loss Prevents Zero-Edge Collapse, But Finds New Degenerate Minimum (May 8)
Second vast.ai run (10 epochs, ~8 hours). The density loss successfully prevented the zero-edge collapse from Run 1, but the model found a new degenerate minimum.

**Active weights evaluation (40 samples each):**
| Metric | Epoch 5 | Epoch 7 | Epoch 10 | Trend |
|--------|---------|---------|----------|-------|
| True SVR | 0% | 0% | 0% | Flat |
| Avg edges | 114.9 | 52.2 | **16.6** | 📉 Still shrinking |
| out_degree | 100% | 100% | 100% | ✅ Trivially satisfied |
| orphan | 12.5% | 10% | 5% | 📉 Getting worse |
| acyclic | 20% | 20% | 20% | Flat — never learned |
| terminal | 100% | 100% | 100% | ✅ Trivially satisfied |

**EMA weights** stayed near baseline (~2,300 edges, 0% SVR) — EMA decay 0.999 is too slow to incorporate the active policy's changes.

**Root cause:** The density loss floor was set to `n_valid_nodes` (~10-20 nodes), but real graphs need ~400-500 edges. The model converged to the floor and sat there. Without any positive signal pulling toward realistic graphs, only penalty terms pushing away from violations, the model has no reason to produce more edges than the bare minimum.

**The key insight (from the user):** "Why can't we compare the resulting graph to the expected graph?" The training loop generates from random noise with NO reference to the ground truth graph. The structural-only loss defines what's *wrong* but never shows what's *right*.

### The Hybrid Loss: Reconstruction + Structure (May 8, commit 992fb24)

**The fix:** Add MSE reconstruction loss against the ground truth graph alongside structural loss.

**Division of labor:**
- **Reconstruction loss (β=1.0):** "generate graphs that *look* like real graphs" — anchors edge density, shape, and scale to the training distribution
- **Structural loss (β=0.1):** "generate graphs that *pass the validator*" — sharpens specific validity rules (degree, cycles, orphans)
- **KL anchor (β=1.0):** prevents catastrophic divergence from the pre-trained reference

**Why this should work:**
1. The flow-matching pre-training used exactly this reconstruction objective and maintained ~474 edges per graph — it never collapsed
2. It plateaued at 0.135 MSE because it treats all matrix elements equally — structural loss knows *which* edges matter
3. Together: reconstruction prevents collapse, structure pushes toward validity
4. Only ONE hyperparameter to tune (β_recon) instead of complex density/orphan weight ratios

**Implementation details:**
- Ground truth is [B, 3, N, N] (presence, edge_type, input_index) but model outputs [B, 6, N, N] (presence, data_type, 4× input_index logits)
- Expand target to 6 channels: one-hot encode input_index into channels 2-5
- Masked MSE over valid node pairs only (padding_mask)
- Clean separation: reconstruction is plain MSE, no per-channel weighting (structural loss already handles that)

### Run 3: Hybrid β_struct=0.1 (May 8)

- Reconstruction loss dropped 20.2 → 0.46 over 10 epochs
- KL hit 10.0 clamp by epoch 3 (same divergence issue as Runs 1-2)
- **Result:** Active E10 had 27 edges (still sparse), EMA E10 had 169 edges, 2.5% SVR EMA
- **Root cause:** Structural loss at β=0.1 still dominated and pushed toward sparsity

### Run 4: Hybrid β_struct=0.01 — Best Run (May 8-9, vast.ai RTX 4090)

Configuration: β_struct=0.01, β_recon=1.0, β_kl=1.0. Let reconstruction dominate.

**Key improvement:** KL stayed at 2.6-5.0, never hit 10.0 clamp. Reconstruction anchor kept model grounded.

Results (80 samples, 10 batches of 8) — these used the **buggy** validator (see below):

| Metric | Baseline | E3 Act | E5 EMA | E7 Act | E10 Act |
|--------|----------|--------|--------|--------|---------|
| SVR | 0% | 1.2% | 1.2% | 0% | 1.2% |
| Edges | 8,110 | 5,268 | 5,647 | 3,368 | 3,480 |
| out_degree | 1% | 19% | 21% | 36% | 35% |
| terminal | 24% | 54% | 74% | 78% | 70% |

No collapse. Genuine structural improvements. But in_degree stuck at 0-5%.

### Milestone 6: Validator Bug Discovery (May 9)

**Three bugs found in `validation.py` that corrupted ALL prior SVR measurements:**

#### Bug 1: `_check_acyclic_data` inverted (critical)
The function returned `True` when cycles were found and `False` when acyclic — exactly backwards. `acyclic_data_pass` was counting cyclic graphs as passing. The "99% acyclic" on baseline was actually "1% acyclic" — the model generates tons of data-plane cycles.

Fix: Invert return values (`return False` when cycle found, `return True` when no cycle).

#### Bug 2: `_check_in_degree` too strict for CONDITION/LOOP
Required exactly 1 data input, but 255/256 CONDITION nodes in the dataset have 2+ inputs (conditions evaluate multi-operand expressions). Changed to `>= 1`.

#### Bug 3: `_check_out_degree` too strict for CONDITION
Required exactly 0 or 2 exec outputs, but compressed motif format can have 1 (partial branch visibility). Changed to allow 0, 1, or 2.

**Corrected dataset validation:** 83.2% SVR (was 0% with buggy validator). The training data itself is now mostly valid.

**Re-evaluation with corrected validator (Run 4):**

| Metric | Dataset | Baseline | E3 EMA | E5 EMA | E7 Act | E10 Act |
|--------|---------|----------|--------|--------|--------|---------|
| SVR | 83.2% | 0% | 0% | 0% | 0% | 0% |
| out_degree | 91.8% | 3.8% | 12.5% | 16.2% | 23.8% | 27.5% |
| in_degree | 89.8% | 2.5% | 6.2% | 3.8% | 0% | 2.5% |
| orphan | 100% | 100% | 100% | 100% | 97.5% | 96.2% |
| acyclic | 100% | 2.5% | 1.2% | 0% | 0% | 0% |
| terminal | 100% | 25% | 28.8% | 63.8% | 65% | 70% |

**The real bottleneck is acyclic_data (0%)** — there was NO structural loss term for data-plane acyclicity. The model had zero gradient signal to avoid data cycles.

### NOTEARS Acyclic Loss + Expanded In-Degree (May 9)

**Three fixes to `structural_loss.py`:**

1. **NOTEARS acyclic loss (weight 2.0):** Uses the trace of the matrix exponential: `h(A) = tr(exp(A)) - n`. Computed via truncated power series `sum_k tr(A^k)/k!` for k=2..8. Each non-zero diagonal in A^k indicates a k-length cycle. Differentiable, efficient (128×128 bmm × 7 iterations), only 2.7GB VRAM.

2. **Expanded in-degree loss (weight 0.5→1.0):** Now covers STATE write nodes (motif=5 with exec_out > 0) in addition to CONDITION/LOOP. Uses soft exec out-degree as differentiable write indicator via `sigmoid(10 × (exec_out - 0.5))`.

3. **Changed in-degree target from "exactly 1" to "at least 1":** Uses `ReLU(1 - data_in)²` instead of `(data_in - 1)²`. Allows multiple data inputs (matching the relaxed validator rules).

**Run 3 started** on same vast.ai instance with hybrid loss. Early signs: recon=20.1 (high initially, will drop), struct=64.1, ~7s/batch.