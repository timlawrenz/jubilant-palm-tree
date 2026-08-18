# Project Status Tracker

**Last Updated**: 2026-08-18 (Exp 1.9 concluded)
**Purpose**: Quick orientation guide — the ground-truth read on current state, verified against disk, git, and test results.

> **This file is authoritative for "where are we now."** For the strategic backlog (what's next, what's ruled out, what's open), see [`docs/03_EXPERIMENT_TREE.md`](docs/03_EXPERIMENT_TREE.md). For the detailed chronological log, see [`docs/02_EXPERIMENTS_AND_RESULTS.md`](docs/02_EXPERIMENTS_AND_RESULTS.md).

---

## 🎯 CURRENT STATUS: Enrichment Program Closed (4 arms, all sub-0.20) — Exp 3 Pivot Decided

### Where we are

**Exp 1.5 (Routing Fidelity)** proved the DiT is steerable (Jaccard 0.50) but the 1D motif array underspecifies programs (typed-F1 0.085). The BoM enrichment program — decode-time and training-time — is now **complete and closed.** All four arms failed to reach the 0.20 gate:

| Arm | Mechanism | N | typed-F1 baseline | Best typed-F1 | Δ | Verdict |
|---|---|---|---|---|---|---|
| **A** (Edge-Count) | Global density bias (FiLM presence) | 128 | 0.088 | 0.082 | −5.8% | **FAIL** |
| **B** (Degree-Profile) | Per-node in/out degree bias (decode) | 128 | 0.075 | **0.109** | **+46%** | **AMBIGUOUS** |
| **C** (Partial-Seed) | Seed gt edges, denoise rest | 64 | 0.098 | 0.296\* | +201%\* | **NEGATIVE** |
| **D (Exp 1.9)** | In-channel degree fine-tune (train-time) | 512 | 0.0895 | 0.0894 (signal) / 0.0769 (null) | ≈0 / −0.013 | **FAIL** |

\* includes seeded edges; unseeded completion F1 = 0 (dispositive)

**Key finding:** The failure is NOT the delivery mechanism (decode bias vs. training-time conditioning) — it's the **dense-matrix flow-matching routing capacity itself.** Exp 1.9's signal arm is statistically indistinguishable from the frozen base (0.0894 vs 0.0895), with the null arm (random degrees) slightly *worse* (0.0769). Four convergent negative interventions across decode- and training-time.

**Exp 2 (Legislative Branch) remains BLOCKED** at typed-F1 ≥ 0.20. **Exp 3 (autoregressive edge-list generation) is now the justified pivot** — the paradigm test of whether diffusion-on-adjacency is the wrong inductive bias.

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

### Immediate next action

**[P1] Pre-register Exp 3 (autoregressive edge-list generation).** The enrichment program is closed with convergent negative evidence: the dense-matrix flow-matching DiT cannot route specific programs whether the signal arrives decode-time (Arms A/B/C) or training-time (Arm D). Exp 3 tests whether the *paradigm* (diffusion over dense adjacency) is the wrong inductive bias for sequential control flow — emitting the edge-list autoregressively instead. Pre-register the gate BEFORE writing code, per project discipline.

**[P2] Cleanup untracked artifacts.** `run_teacher_pseudo_labeling.py` (untracked), `mcp-unreal/` (other project's repo), `runs/` (concluded RLAIF leftovers). Also remove the obsolete `scripts/evaluate_tte.py`/`finetune_degree_profile.py` test artifacts if they stay after Exp 3's eval harness is decided.

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
