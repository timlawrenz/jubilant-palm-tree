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

---

## Errata: Published Materials Corrections (May 10, 2026)

The validator bugs discovered in Milestone 6 affect claims in the following published materials. This section documents the specific corrections needed.

### 1. Published Blog Post: "Eradicating Syntax: The Neural Universal Machine"
**URL:** https://lawrenz.com/2026/05/05/eradicating-syntax-the-neural-universal-machine.html

**Affected claims:**

| Section | Original Claim | Correction |
|---------|---------------|------------|
| Title area | "100% Syntactic Validity" | Measured with buggy validator (acyclic check inverted). The Judicial Constraint Solver post-processing still achieved high structural quality, but the 100% SVR figure is unreliable. Dataset itself validates at 83.2% with the corrected validator. |
| "The Breakthrough: 100% SVR at 128 Nodes" | "100.00% Syntactic Validity Rate" | Same as above. The pre-training + Constraint Solver pipeline produces high-quality graphs, but the measurement was inflated by the `_check_acyclic_data` bug returning `True` for cyclic graphs. |
| Law 2: Data In-Degree | "Condition and Loop nodes require **exactly 1** incoming data edge" | Corrected to **≥1**. 255/256 COND nodes in the training dataset have 2+ data inputs (conditions evaluate multi-operand expressions like `a < b`). |
| Law 1: Execution Out-Degree | "Condition, Loop: Exactly 0 or 2" | Corrected to allow 0, 1, or 2. Compressed motifs can legitimately have 1 exec output when one branch falls outside the compression window. |
| "What's Next: RLAIF" section | Describes PPO with base rewards and jackpot multiplier | PPO/REINFORCE approach was abandoned as intractable. Actual approach uses differentiable structural losses (NOTEARS acyclic, degree constraints, density) combined with reconstruction MSE and KL anchor. |

### 2. In-Repo Documents (Corrected)
- **README.md**: Roadmap items 3-4 updated to reflect corrected SVR and actual RLAIF approach.
- **docs/neural-universal-machine-architecture.md**: Section 7 rewritten for differentiable structural loss. Laws 1-2 descriptions corrected. Errata note added.

### 3. PROJECT_LOG.md — Milestone 4 (This File)
- The "100.00% SVR" claim in Milestone 4 was measured with the buggy validator. The Constraint Solver was doing real work (arity snapping, conflict resolution), so the graphs were structurally sound in most respects, but data-plane acyclicity was not being checked correctly. This does NOT invalidate the pre-training work — only the specific SVR measurement.
- The "Deploy PPO" plan in Milestone 4's next steps was abandoned (see Milestone 5).

### What Remains Valid
- The execution engine, Fibonacci proof, and Turing-completeness argument are unaffected.
- The dataset compression pipeline (74→6 motifs, literal extraction) is unaffected.
- The DiT architecture (permutation, cross-hatch, axial attention) is unaffected.
- The Judicial Constraint Solver's arity snapping and conflict resolution logic is unaffected.
- The pre-training loss convergence (~0.135) is unaffected.
- Only the SVR measurement (which depends on the validator) was corrupted.

---

## Milestone 6: Sharpened NOTEARS and Mode Collapse Discovery
**Status:** Run 8 Complete — Key Findings for Paper

### Paper-Relevant Findings (Runs 5–9)

#### 1. Naive NOTEARS is Ineffective on Soft Adjacency (Run 5)
Standard NOTEARS `tr(exp(A))-n` applied to a continuous soft adjacency matrix (~0.25 everywhere) produces **uniform gradient** across all edges. It cannot distinguish cycle-forming edges from background noise. Result: acyclic pass rate unchanged at 1.2% after 10 epochs.

**Insight for paper:** NOTEARS was designed for binary or near-binary adjacency matrices. When applied to the intermediate soft outputs of a generative model (where all entries are ≈0.25 from sigmoid), the matrix exponential's gradient is dominated by the uniform bulk, not by specific cycles. This is a previously undocumented failure mode.

#### 2. Sharpened Adjacency Restores NOTEARS Effectiveness
Applying `sigmoid(10 × (A − 0.5))` before the NOTEARS computation polarizes the soft adjacency into near-binary values. This concentrates the gradient onto edges the model has committed to (>0.5 → ~1) and suppresses gradient from uncommitted edges (<0.5 → ~0). Result: acyclic pass rate 1.7% → 71.7% (epoch 1) → 100% (epoch 5).

**Contribution:** A simple, zero-parameter modification that makes NOTEARS compatible with continuous generative model outputs. Applicable to any DAG-constrained generation setting using flow matching or diffusion.

#### 3. Training Stability Requires Triple Clamping
Sharpened NOTEARS is gradient-volatile — the structural loss oscillates 3–207× between batches (vs ~1–5× for per-node losses). Three clamps were necessary for stable training:
- **Structural loss clamp** (max=50): prevents sharpened NOTEARS gradient spikes
- **Reconstruction loss clamp** (max=100): prevents forward-pass explosion
- **KL divergence clamp** (max=10): bounds policy drift from reference

Without all three, training consistently produces NaN weights by batch ~220 (reproduced across Runs 6, 7). Learning rate reduction (1e-5 → 5e-6) was also required.

#### 4. Mode Collapse via Edge Erasure (Runs 8-9)
With structural constraints active, the model discovered a degenerate optimum: **generate graphs with zero edges**. An empty graph trivially satisfies acyclicity and degree constraints.

This manifested as edge count collapsing: 486 (baseline) → 3 (epoch 1) → 0 (epoch 5). SVR simultaneously dropped from 7.1% → 0% because empty graphs lack required program structure.

**The Final Ablation Proof**: 
A rigorous hyperparameter grid was run testing $\beta_{recon}$ vs $\beta_{struct}$. As the plots below demonstrate, even when heavily favoring reconstruction, the model either structurally collapses (density drops to the floor) or entirely fails to improve Syntactic Validity (SVR flatlines near 0%).

![SVR Comparison](assets/rlaif_ablation/svr_comparison.png)
*Fig 1: SVR fails to climb regardless of structural/reconstruction weighting.*

![Density Comparison](assets/rlaif_ablation/density_comparison.png)
*Fig 2: Edge density collapses to the minimum target boundary, proving the model is evading the structural penalties by erasing edges rather than learning valid topology.*

Because an exponential penalty ($Tr(e^A)$) cannot be statically balanced against a quadratic penalty (MSE) in a smooth generative model, this avenue of research has been **concluded as a negative result**. Hard constraints must be enforced deterministically at decode-time.

#### 5. Ground-Truth Edge Count as Density Floor (Run 9 — In Progress)
Rather than a heuristic "1 edge per valid node" density target, we use the actual edge count from each training example's ground truth graph. This provides a per-sample density floor that is 4× stronger than the heuristic on empty graphs (density_loss ≈ 3.8 vs 0.9).

### Quantitative Results Summary
| Run | Config Change | Acyclic | SVR | Edges | Outcome |
|-----|--------------|---------|-----|-------|---------|
| 5 | Naive NOTEARS (β=0.01) | 1.2% | 0% | 56±41 | NOTEARS ineffective |
| 8 E1 | Sharpened NOTEARS (β=0.2) | 71.7% | **7.1%** | 3±10 | First confirmed SVR |
| 8 E5 | Same, longer training | **100%** | 0% | 0±2 | Mode collapse |
| 8 E10 | Same, longer training | **100%** | 0% | 0±1 | Full collapse |
| 9 | + target edge density | TBD | TBD | TBD | In progress |

### Ablation Story Arc (for paper structure)
1. Pre-training alone → 0% SVR (no structural awareness)
2. Per-node structural losses (out-degree, terminal) → 0% SVR, but individual metrics improve 3–5×
3. Naive NOTEARS for acyclicity → fails (uniform gradient on soft adjacency)
4. Sharpened NOTEARS → acyclic solved, first SVR > 0%, but mode collapse to empty graphs
5. Target-aware density floor → prevents collapse (Run 9, pending)

**Run 3 started** on same vast.ai instance with hybrid loss. Early signs: recon=20.1 (high initially, will drop), struct=64.1, ~7s/batch.# Project Status Tracker

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
RLAIF Checkpoint Evaluation Results (Epochs 2-9, 50 samples, 3 seeds)
======================================================================

Device: cuda
Dataset: 3779 valid graphs (max_nodes <= 64)

CHECKPOINT EVALUATION RESULTS
======================================================================

Epoch    Weights  SVR%     ±      OutDeg   InDeg    Orphan   Acyclic  Term     Edges       
------------------------------------------------------------------------------------------
base     active   0.0      0.0    10.0     4.7      100.0    2.0      27.3     201±337
2        active   4.7      2.5    99.3     42.0     13.3     71.3     98.0     2±2
2        ema      4.0      2.8    100.0    37.3     14.7     66.7     100.0    2±2
3        active   4.7      4.1    99.3     40.0     14.0     70.7     97.3     2±2
3        ema      2.0      1.6    99.3     38.0     16.0     64.7     99.3     2±2
4        active   4.7      2.5    99.3     43.3     20.0     70.7     98.0     2±2
4        ema      4.7      2.5    99.3     46.7     15.3     69.3     99.3     2±2
5        active   2.0      1.6    98.7     40.0     14.7     67.3     99.3     2±2
5        ema      6.0      3.3    99.3     44.7     17.3     71.3     100.0    2±2
6        active   2.0      1.6    99.3     44.0     16.0     72.0     100.0    2±2
6        ema      4.7      3.8    98.7     45.3     18.0     76.7     100.0    2±2
7        active   4.0      2.8    99.3     50.7     13.3     75.3     100.0    1±2
7        ema      4.7      2.5    99.3     47.3     16.7     73.3     99.3     2±2
8        active   2.7      2.5    99.3     50.0     14.7     73.3     99.3     2±2
8        ema      3.3      0.9    99.3     49.3     10.7     72.7     99.3     2±2
9        active   4.7      0.9    99.3     54.0     15.3     76.7     99.3     1±2
9        ema      6.0      2.8    99.3     46.0     16.0     76.7     99.3     2±2

======================================================================
RECOMMENDED: Epoch 9 (ema weights) — SVR 6.0% ± 2.8%
  Path: checkpoints/rlaif/rlaif_struct_epoch_9.pt
======================================================================

Key Findings:
=============
1. R-LAIF models converge toward sparse, valid DAGs (70-77% acyclic)
2. All R-LAIF epochs show >97% terminal rate (vs 27% baseline)
3. SVR remains low (2-6%) - model learns sparsity but not correctness
4. Epoch 9 EMA is best: SVR 6.0% ± 2.8%
5. Edge count collapsed from 201 (baseline) to 1-2 (R-LAIF)
## Decode-Time Constraint Solving (Active Pivot)\n\nBecause the RLAIF approach resulted in mode collapse, the project pivoted to enforcing hard constraints at decode time. The generator (DiT) emits a probabilistic heatmap, and a deterministic algorithm repairs the topology to guarantee acyclicity and exact degree arithmetic without altering the network weights.\n\n### Baseline SVR Re-Measurement (Pre-Trained DiT + Naive Discretizer)\nBefore implementing advanced cycle-breaking algorithms, we established the true baseline of the raw `num_dit_epoch_340.pt` checkpoint passing through the original naive continuous-to-discrete solver. Because all previous measurements were skewed by the `GraphValidator` bugs, this is the first honest assessment of the base model's zero-shot structural performance:\n\n*   **Total Samples**: 160 (5 batches of 32)\n*   **Average Edge Count**: 342.4 (Healthy, non-collapsed density)\n*   **Overall SVR**: **0.00%**\n    *   No Orphan Pass: 100.00%\n    *   Terminal Sink Pass: 24.38%\n    *   Execution Out-Degree Pass: 3.12%\n    *   Data In-Degree Pass: 2.50%\n    *   Acyclic Data Pass: 1.25%\n\n### Decode-Time Acyclicity Repair
Based on the baseline results showing only 1.25% of generated graphs were acyclic, we implemented deterministic cycle-breaking in the decode step (`src/models/constraint_solver.py`). The algorithm extracts the continuous probability score (`soft_presence`) generated by the DiT heatmap, finds data-plane cycles using Depth-First Search, and removes the "back-edge" with the *lowest* confidence score.

**Re-Measurement Results (Pre-Trained DiT + Constraint Solver):**
*   **Total Samples**: 160 (5 batches of 32)
*   **Average Edge Count**: 352.4 (Maintains dense graph topology)
*   **Acyclic Data Pass**: **100.00%** (Up from 1.25%)
*   **Overall SVR**: **1.88%** (Up from 0.00%)

**Conclusion**: The acyclicity repair successfully guarantees cycle-free data planes without collapsing edge density (the mode collapse issue seen in RLAIF is entirely avoided). The SVR has lifted off zero. 

The remaining bottlenecks are the Degree Arithmetic laws (In/Out degree pass ~6%). The next algorithmic solver iteration must implement deterministic top-K arity snapping for Execution and Data edges.\n
