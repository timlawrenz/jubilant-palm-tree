# Routing Fidelity (Exp 1.5) — Asset Directory

**Verdict: AMBIGUOUS — Bill-of-Materials indictment (PIVOT)**

## Canonical results

- **`fidelity_eval.txt`** — N=512 definitive sweep (RTX 4090). typed-F1 0.085, random baseline 0.046. Best graph 0.600.
- **`controllability_ablation.txt`** — 3 seeds, same noise x₀, different motif sequences. Mean Jaccard 0.5045 (controllable).

## Gate registration

Pre-registered in `docs/02_EXPERIMENTS_AND_RESULTS.md` (commit `49e3266`).

## Scripts

- `scripts/evaluate_routing_fidelity.py`
- `scripts/ablation_motif_controllability.py`
- `src/models/fidelity.py` + `tests/test_fidelity.py`
