# Degree-Profile Enrichment (Exp 1.7, Arm B) — Asset Directory

**Verdict: AMBIGUOUS — +46% lift, best enrichment yet, insufficient for PASS**

## Canonical result

- **`degree_sweep_128.txt`** — **N=128 definitive sweep** (146s, RTX 4090). in_scale=0.05 lifts typed-F1 from 0.075 → 0.109 (+46%). This is the authoritative result cited in the ledger.

## Intermediate result

- `degree_sweep_64.txt` — N=64 confirming run (in_scale=0.05 → 0.110, +32%). Consistent with N=128 definitive.

## Gate registration

Pre-registered in `docs/02_EXPERIMENTS_AND_RESULTS.md`.

## Model

- `src/models/model.py` — `ode_generate_with_degree_bias()`, `EXPECTED_EXEC_OUT`, `EXPECTED_DATA_IN`
