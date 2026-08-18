# Project Status Tracker

**Last Updated**: 2026-08-18 (Exp 3 fully PASS + SVR closure; Exp 2 gate pre-registered)
**Purpose**: Quick orientation guide — the ground-truth read on current state, verified against disk, git, and test results.

> **This file is authoritative for "where are we now."** For the strategic backlog (what's next, what's ruled out, what's open), see [`docs/03_EXPERIMENT_TREE.md`](docs/03_EXPERIMENT_TREE.md). For the detailed chronological log, see [`docs/02_EXPERIMENTS_AND_RESULTS.md`](docs/02_EXPERIMENTS_AND_RESULTS.md).

---

## 🎯 CURRENT STATUS: Exp 3 PASS (Tier-1) — AR Paradigm Routes, Exp 2 UN-GATED

### Where we are

**Exp 3 (autoregressive edge-list generation) PASSED — fully (Tier-1 + Tier-2).** The pivot was right: replacing the dense-matrix DiT with an AR transformer that emits the edge list as tokens, conditioned on the same motif array, achieves **held-out typed-F1 0.776 / 0.777 across two seeds** (gate 0.20; diffusion-best across 4 arms was 0.109; random baseline 0.074). Edge counts match ground truth (15.5 vs 15.8). Verified on 512 held-out graphs disjoint from training (programmatic assert), with the learning curve e49→e99→e150 (0.739→0.770→0.776) ruling out memorization.

**The bottleneck was the DiT-over-dense-adjacency implementation, not the A→C thesis.** Exp 2 (Legislative Branch: LLM compiles intent → Bill of Materials) is **FORMALLY UN-GATED** (pre-registered Tier-2 criterion met: Δ0.0009 across seeds).

| Exp | Verdict | Held-out typed-F1 |
|---|---|---|
| 1.5 Routing Fidelity | AMBIGUOUS | 0.085 |
| 1.6–1.9 Enrichment arms (A–D) | FAIL ×3, AMBIGUOUS ×1 (best 0.109) | ≤0.109 |
| **3 AR Edge-List** | **PASS (2 seeds)** | **0.776 / 0.777** |

### Summary of completed work

| Phase | Outcome | SVR / Fidelity | Artifacts |
|---|---|---|---|
| **DiT Pre-Training** (340 epochs, 128-node) | Flow Matching converged, MSE ~0.135 | 0% (true SVR) | `checkpoints/num_dit_epoch_340.pt` |
| **RLAIF Fine-Tuning** (Runs 5–9) | **Negative.** Mode collapse via edge erasure. | Peak 7.1% → 0% | `docs/assets/feat/rlaif-ablation/` |
| **Decode-Time Constraint Solving** | Deterministic repair — acyclicity + out-degree 100%. | **0% → 10%** | `src/models/constraint_solver.py` |
| **6th Law (Global Spine)** | Entry/exit controllability. SVR fixed to honest 10%. | 10% strict SVR | `src/models/validation.py` |
| **Interpreter Harness Fix** | 100% of GT + generated 6-law graphs halt. | 6-law ⟺ executability | `docs/assets/exp/fix-interpreter-entry-detection/` |
| **Exp 1.5 — Routing Fidelity** | AMBIGUOUS — DiT steerable, motif underspecifies. | typed-F1 0.085 | `docs/assets/exp/routing-fidelity/` |
| **Exp 1.6 — Edge-Count Enrichment (Arm A)** | FAIL — controls edges, reduces fidelity. | −5.8% vs baseline | `docs/assets/exp/edge-count-enrichment/` |
| **Exp 1.7 — Degree-Profile Enrichment (Arm B)** | AMBIGUOUS — +46% lift, best yet but insufficient. | 0.075 → 0.109 | `docs/assets/exp/degree-profile-enrichment/` |
| **Exp 1.8 — Partial-Adjacency Seeding (Arm C)** | NEGATIVE — scaffold works, unseeded completion F1 = 0. | 0.098 → 0.296\* (unseeded=0) | `docs/assets/exp/partial-seed-enrichment/` |
| **Exp 1.9 — Training-Time Enrichment (Arm D)** | FAIL — signal inert (0.0894 ≈ base 0.0895), null arm worse (0.0769). Enrichment program closed. | ≈0 vs base | `docs/assets/exp/training-time-enrichment/` |
| **Exp 3 — AR Edge-List Generation** | **PASS (Tier-1)** — held-out typed-F1 **0.776** vs gate 0.20; edge counts match (15.5 vs 15.8). | 0.776 (10× above random) | `docs/assets/exp/autoregressive-edge-list/` |

### Immediate next action

**[P1] Execute Exp 2 (Legislative Branch).** Gate pre-registered 2026-08-18 (ledger, commit `________`): end-to-end held-out typed-F1 ≥ 0.20 AND ≥ +0.05 over random-BoM control; no-leak guardrail (LLM sees only method context + literal pool); decomposition rule separates Legislative vs Executive failure. Create branch `exp/legislative-branch`, connect a hosted LLM as the Legislator over the frozen AR Executive, then run the gate.

**[P2] Cleanup untracked artifacts.** `run_teacher_pseudo_labeling.py` (untracked), `mcp-unreal/` (now gitignored), `runs/` (concluded RLAIF leftovers). Retire `scripts/evaluate_tte.py`/`finetune_degree_profile.py` (Exp-1.9-only) once Exp 2's harness is settled.

### Test suite status

| Test file | Tests | Result |
|---|---|---|
| `tests/test_validation.py` | 13 | ✅ All pass |
| `tests/test_constraint_solver.py` | 6 | ✅ All pass |
| `tests/test_fidelity.py` | 3 | ✅ All pass |
| `tests/test_naive_discretizer.py` | — | ✅ All pass |
| `tests/test_execution_engine.py` | 3 | ✅ All pass (Fibonacci returns a=55) |

**Open failure:** `tests/test_reward.py::test_all_pass_gets_jackpot` — stale RLAIF test, non-blocking.

### Branch / repo state

- **Branch:** `exp/autoregressive-edge-list` (Exp 3 conclusion; merge to main after Tier-2)
- **Test suite:** 98 passed, 1 known-stale failure (`test_reward.py::test_all_pass_gets_jackpot`)
- **Untracked artifacts:** `run_teacher_pseudo_labeling.py`, `runs/` (concluded RLAIF leftovers); `mcp-unreal/` gitignored (external repo)
- **GPU scheduler:** job `jpt-ar-1787083532` busy (was used for Exp 3 training/eval; release when Tier-2 completes)

---

## 📋 Key Files

| File | Role |
|---|---|
| `PROJECT_STATUS.md` | **This file** — current state |
| `docs/03_EXPERIMENT_TREE.md` | Strategic backlog |
| `docs/02_EXPERIMENTS_AND_RESULTS.md` | Chronological research log |
| `docs/01_VISION_AND_ARCHITECTURE.md` | Full architecture spec |
| `src/models/constraint_solver.py` | Decode-time deterministic repair |
| `src/models/validation.py` | GraphValidator — 6-Law checker |
| `src/models/model.py` | DiT + `ode_generate_with_degree_bias()` + `ode_generate_with_density_bias()` |
| `src/models/fidelity.py` | Edge-set fidelity metric (typed + untyped F1) |
| `scripts/evaluate_routing_fidelity.py` | Exp 1.5 evaluation harness |
| `scripts/evaluate_edge_count_enrichment.py` | Exp 1.6 density sweep |
| `scripts/ablation_motif_controllability.py` | Motif conditioning ablation |

## 📚 Key Checkpoints

| Checkpoint | Path | Notes |
|---|---|---|
| Base DiT (340) | `checkpoints/num_dit_epoch_340.pt` | Pre-training only |
| Best with solver | `checkpoints/num_dit_epoch_340.pt` + `ConstraintSolver` | **10% SVR** (current best pipeline) |
