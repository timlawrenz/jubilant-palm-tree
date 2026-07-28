# Project Status Tracker

**Last Updated**: 2026-07-25
**Purpose**: Quick orientation guide — the ground-truth read on current state, verified against disk, git, and test results.

> **This file is authoritative for "where are we now."** For the strategic backlog (what's next, what's ruled out, what's open), see [`docs/03_EXPERIMENT_TREE.md`](docs/03_EXPERIMENT_TREE.md). For the detailed chronological log, see [`docs/02_EXPERIMENTS_AND_RESULTS.md`](docs/02_EXPERIMENTS_AND_RESULTS.md).

---

## 🎯 CURRENT STATUS: BoM Enrichment — Arms A+B Complete, Arm C Next

### Where we are

**Exp 1.5 (Routing Fidelity)** proved the DiT is steerable (Jaccard 0.50) but the 1D motif array underspecifies programs (typed-F1 0.085). This triggered a BoM enrichment program: can we improve routing fidelity by enriching the conditioning signal?

**Two enrichment arms completed, one remaining:**

| Arm | Mechanism | N | typed-F1 baseline | Best typed-F1 | Δ | Verdict |
|---|---|---|---|---|---|---|
| **A** (Edge-Count) | Global density bias (FiLM presence) | 128 | 0.088 | 0.082 | −5.8% | **FAIL** |
| **B** (Degree-Profile) | Per-node in/out degree bias | 128 | 0.075 | **0.109** | **+46%** | **AMBIGUOUS** |
| **C** (Partial-Seed) | Seed 10–20% gt edges, denoise rest | — | — | — | — | **NEXT** |

**Key finding across both arms:** Mechanisms that target the presence channel uniformly (global density, out-degree bias) strip routing signal and hurt fidelity. The in-degree bias in Arm B took the opposite approach — encourage more edges where the dataset says they should be — and it's the only mechanism that improved fidelity (+46%). The DiT IS receptive to enrichment signals, but the presence channel conflates edge count with edge quality.

**Exp 2 (Legislative Branch) remains gated** at typed-F1 ≥ 0.20. Arm C is the remaining enrichment candidate that directly targets *which* edges rather than *how many*.

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

### Immediate next action

**[P1] Arm C — Partial-Adjacency Seeding.** The remaining enrichment candidate. Seed 10–20% of ground-truth edges into the initial noise state x₀, then run the ODE to complete the rest. This directly guides *which* edges rather than *how many*. Pre-register gate BEFORE writing code.

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

- **Branch:** `main` (clean, up to date with origin)
- **Test suite:** 98/99 passing
- **Untracked artifacts:** `mcp-unreal/` (other project's repo), `runs/` (concluded RLAIF training leftovers)
- **GPU scheduler:** No active jobs or crons for this project

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
