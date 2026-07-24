# Project Status Tracker

**Last Updated**: 2026-07-23
**Purpose**: Quick orientation guide — the ground-truth read on current state, verified against disk, git, and test results.

> **This file is authoritative for "where are we now."** For the strategic backlog (what's next, what's ruled out, what's open), see [`docs/03_EXPERIMENT_TREE.md`](docs/03_EXPERIMENT_TREE.md). For the detailed chronological log, see [`docs/02_EXPERIMENTS_AND_RESULTS.md`](docs/02_EXPERIMENTS_AND_RESULTS.md).

---

## 🎯 CURRENT STATUS: Exp 1.5 Concluded (AMBIGUOUS) — PIVOT to Bill-of-Materials Enrichment

### Where we are

**Exp 1.5 (Routing Fidelity) has concluded.** The DiT's typed-F1 is 0.0852 (1.85× above random baseline 0.0461, but far below the 0.50 controllability threshold). The Jaccard ablation (0.5045) confirms the motif signal genuinely steers generation — the Executive Branch is **steerable but underspecified**. The 1D motif array (e.g., `[Boundary, Sequence, Condition, ...]`) cannot disambiguate between many different programs that share the same motif sequence.

The **thesis is not falsified** (routing is distinguishable from random; output changes with different motifs), but the Bill of Materials needs enrichment before the Legislative Branch can usefully emit it from human intent.

### Headline result: Exp 1.5 — AMBIGUOUS (PIVOT)

| Metric | Value | Gate threshold | Status |
|---|---|---|---|
| Typed-F1 | **0.0852** | ≥ 0.50 | ❌ Below PASS |
| Δ above random | **+0.0391** (1.85×) | ≥ 0.20 | ❌ Below PASS |
| Jaccard (ablation) | **0.5045** | < 0.70 | ✅ Controllable |
| 2σ of random? | **No** (~22× SEM above) | within? | ✅ Not random |
| Best graph typed-F1 | **0.6000** | — | ✅ Can route faithfully sometimes |
| Edge over-generation | **6.5×** (181 vs 28 gt) | — | ⚠️ Key diagnostic |

Raw logs: `docs/assets/exp/routing-fidelity/fidelity_eval.txt`, `controllability_ablation.txt`

### Immediate next action

**[P1] Plan the BoM enrichment:** The Executive Branch is alive and steerable — now the conditioning signal needs teeth. Write a short plan enumerating the enrichment candidates (edge-count conditioning, degree-profile conditioning, partial-adjacency seeding) with pre-registered gates for each. Do NOT start Exp 2 (intent→topology) until at least one enrichment arm improves typed-F1 to >0.20.

### Summary of completed work

| Phase | Outcome | SVR / Fidelity | Artifacts |
|---|---|---|---|
| **DiT Pre-Training** (340 epochs, 128-node) | Flow Matching converged, MSE plateaued ~0.135 | 0% (true SVR with corrected validator) | `checkpoints/num_dit_epoch_340.pt` |
| **RLAIF Fine-Tuning** (Runs 5–9) | **Concluded — Negative.** Mode collapse: edges→0. Exponential NOTEARS penalty + quadratic MSE statically unstable. | Peak 7.1% E1 → 0% by E5 | `docs/assets/feat/rlaif-ablation/` |
| **Decode-Time Constraint Solving** | **Winning approach.** Deterministic repair of DiT heatmap — no training, no mode collapse. | **0% → 10%** 6-law SVR | `src/models/constraint_solver.py` |
| **6th Law (Global Spine)** | Added entry/exit controllability. Fixed SVR to honest 10%. | 10% strict SVR | `src/models/validation.py` |
| **Interpreter Harness Fix** | Fixed entry-detection bug. **100% of GT + generated 6-law graphs halt.** | 6-law ⟺ executability | `docs/assets/exp/fix-interpreter-entry-detection/` |
| **Exp 1.5 — Routing Fidelity** | **AMBIGUOUS — PIVOT.** DiT steerable (Jaccard 0.50) but 1D motif underspecifies programs (typed-F1 0.085). Best graph 0.60. | 1.85× above random; 6.5× edge over-gen | `docs/assets/exp/routing-fidelity/` |

### Test suite status

| Test file | Tests | Result |
|---|---|---|
| `tests/test_validation.py` | 13 | ✅ All pass (includes 6th-law/global-spine coverage) |
| `tests/test_constraint_solver.py` | 6 | ✅ All pass |
| `tests/test_naive_discretizer.py` | — | ✅ All pass |
| `tests/test_execution_engine.py` | — | ⚠️ 2 failures (see below) |

**Known test failures** (non-blocking for Exp 1.5):
- `test_fibonacci_execution`: `NotImplementedError` — the interpreter can't run the flagship Fibonacci demo end-to-end yet.
- `test_interpreter_infinite_loop_guard`: the test fixture is malformed (missing `Exit(1)` exec edge for a LOOP node), so it fails on a structural check before reaching the step-limit guard it intends to test.

These are downstream of the Executive Branch — they affect the interpreter, not the DiT or solver. Fix before publishing or resuming the Legislative Branch.

---

## 📋 Key Files

| File | Role |
|---|---|
| `PROJECT_STATUS.md` | **This file** — current state, authoritative for "where are we" |
| `docs/03_EXPERIMENT_TREE.md` | Strategic backlog — what's ACTIVE/NEXT/TBD/CONCLUDED, with macro-questions |
| `docs/02_EXPERIMENTS_AND_RESULTS.md` | Chronological research log with detailed metrics, plots, and errata |
| `docs/01_VISION_AND_ARCHITECTURE.md` | Full architecture spec: three branches, 6-Laws, interpreter |
| `README.md` | Public-facing overview, mostly accurate (roadmap item 4 needs a label update) |
| `src/models/constraint_solver.py` | Decode-time deterministic repair (the 10% SVR engine) |
| `src/models/validation.py` | GraphValidator — the 6-Law checker, now with global-spine support |
| `src/models/dit.py` | DiT architecture (AMP fix committed) |
| `src/rlaif/train_rlaif.py` | RLAIF training script (concluded arm, preserved for reference) |
| `src/rlaif/structural_loss.py` | Differentiable structural constraints including sharpened NOTEARS |

---

## 📚 Key Checkpoints

| Checkpoint | Path | SVR | Notes |
|---|---|---|---|
| Base DiT (340) | `checkpoints/num_dit_epoch_340.pt` | 0% | Pre-training only, no structural awareness |
| Best RLAIF E1 | `checkpoints/rlaif/vastai_run8/rlaif_struct_epoch_1.pt` | 7.1% | Sharpened NOTEARS, collapses after E1 |
| Best with solver | `checkpoints/num_dit_epoch_340.pt` + `ConstraintSolver` | **10%** | Current best pipeline (no training needed) |

---

## 🗺 Training Run History (RLAIF arm — concluded)

| Run | Epochs | SVR | Acyclic | Key Change | Issue |
|-----|--------|-----|---------|------------|-------|
| 5 | 10 | 0% | 1.2% | NOTEARS (weak β=0.01) | Acyclic ineffective |
| 6 | 2 | 25%* | — | Sharpened NOTEARS (β=0.5) | Recon explosion, NaN crash |
| 7 | 1 | — | — | + recon clamp | Still NaN at batch 220 |
| **8** | **10** | **7.1%** | **100%** | + struct clamp, lr=5e-6, β=0.2 | Mode collapse (edges→0) |
| 9 | — | — | — | + target edge density | OOM crashes (resolved in local tree, not fully trained) |

> The above history is preserved for reference. RLAIF is concluded as a negative result. The winning approach is decode-time deterministic repair (constraint solver), not gradient-based structural penalties.

---

## 🔜 Immediate Next Actions

1. **[P1] Run Exp 1.5 (Routing Fidelity).** This is the experiment that determines whether the core thesis is alive. Inference-only, no training needed. Execute per `.hermes/plans/2026-06-05_routing_fidelity.md` — with two guardrails: (a) assert `augment_permutation=False` end-to-end before trusting F1 numbers, (b) promote the controllability ablation (Task 4) to the headline metric.
2. **[P2] Fix or quarantine the 2 failing interpreter tests.** At minimum, fix the malformed loop-guard test fixture. Determine whether `test_fibonacci_execution` can be made to pass before citing the Fibonacci demo in publications.
3. **[P3] Clean up ghost-project artifacts.** Move `DISCONTINUATION_NOTICE.md` and related Phase 4B docs into `archive/` or add a banner clarifying they refer to the *pre-pivot* GNN effort, not NUM. Decide whether `mcp-unreal/` (untracked) belongs in this repo.
4. **[After Exp 1.5 resolves] Decide the Legislative Branch path.** If fidelity is high, Exp 2 (build the LLM → motif-sequence conditioning path) is next. If fidelity is low but output changes with different motifs, the Bill of Materials needs enrichment. If motifs barely move output, conditioning is the bug to fix before anything else.
