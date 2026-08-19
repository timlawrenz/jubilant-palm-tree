# Project Status Tracker

**Last Updated**: 2026-08-18 (Exp 2 formal PASS — A→C demonstrated end-to-end; MQ3 next)
**Purpose**: Quick orientation guide — the ground-truth read on current state, verified against disk, git, and test results.

> **This file is authoritative for "where are we now."** For the strategic backlog (what's next, what's ruled out, what's open), see [`docs/03_EXPERIMENT_TREE.md`](docs/03_EXPERIMENT_TREE.md). For the detailed chronological log, see [`docs/02_EXPERIMENTS_AND_RESULTS.md`](docs/02_EXPERIMENTS_AND_RESULTS.md).

---

## 🎯 CURRENT STATUS: Exp 2 PASS — A→C demonstrated end-to-end; MQ3 goal gate next

### Where we are

**Exp 2 (Legislative Branch) PASSED, reproduced.** An LLM compiles method context (name/repo/file + literal pool — no-leak) into the Bill of Materials, the AR Executive (Exp 3) draws the graph, and end-to-end held-out typed-F1 is **0.490 / 0.496 across two draws** (gate 0.20 → 2.45×; floor 0.26; ceiling 0.776 validating the harness chain). The **founding A→C premise — intent → executable structure — is now demonstrated end-to-end** on 512 real held-out Ruby methods, at ~1.3s/method on local hardware.

The full chain works: **Legislative LLM → BoM → AR Executive → ConstraintSolver → (Justice/interpreter next)**. Both earlier gates fed into this: Exp 3 proved the AR paradigm; the no-leak + decomposition + ceiling-control disciplines built on it.

| Gate | Verdict | Headline |
|---|---|---|
| Exp 3 AR Executive | PASS (2 seeds) | 0.776 / 0.777, SVR 63–72% |
| **Exp 2 Legislative** | **PASS (2 draws)** | **e2e 0.490 / 0.496** |
| MQ3 goal gate | PRE-REGISTERED | benchmark vs LLM codegen (`e923783`) |

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

**[P1] Execute the MQ3 goal gate.** Pre-registered `e923783`: benchmark the full pipeline (LLM → BoM → AR → solver → interpreter) against an LLM emitting Ruby directly, on a held-out task suite, measuring functional correctness + cost-per-correct-program + validity/verifiability. **Precondition:** the executability bridge check (Graph-Walk Interpreter correctly executes ≥90% of solver-valid AR outputs) must pass first — it's the untested link in the chain and the gate is void without it.

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

---

## OPEN DECISION (2026-08-18) — MQ3 functional-correctness axis: representation fork

The MQ3 audit surfaced a project-level fork that no experiment alone can decide:

**A. Keep corpus representation, narrow MQ3's correctness claim.**
Functional correctness measured on linear/branch-free tasks only (executable
today: ~2/3951 corpus graphs + curated linear tasks). Conditional graphs are
documented as non-executable (no {0,1} EXEC branch pair anywhere in the corpus).
Cheap; honest; leaves the branch gap as a stated limitation.

**B. Fix the representation (compressor emits branch edges), recompress, retrain AR.**
Makes conditional graphs executable per the interpreter's demo contract; enables
true functional correctness for the whole curated suite. Cost: corpus-wide
recompression + AR retrain (~1-2 GPU-hours) + re-run Exp 3/2 gates on the new
representation. This is the "unblock" named in the ledger (commit ce1cdf7).

**C. Revisit the deferred founding hypothesis (spec-intent + outside-in).**
The branch-gap finding strengthens the original intuition that flat
sequence/matrix representations fight the tree structure of code. But per the
owner's standing direction, this is POST-MQ3.

**Current position:** option A is executing now (the runnable-arms benchmark).
Options B/C wait for the owner's call after the benchmark lands.
