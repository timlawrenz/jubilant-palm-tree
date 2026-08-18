# Autoregressive Edge-List Generation (Exp 3) — Asset Directory

**Verdict: PASS (Tier-1)** — the AR paradigm routes; dense-matrix diffusion was the bottleneck.

## Canonical result (held-out N=512, 0 errors, rng-seeded split disjoint from training)

| Checkpoint | Held-out typed-F1 | gen_edges (gt 15.8) | Verdict |
|---|---|---|---|
| e49 | 0.739 | 15.5 | learning ramp |
| e99 | 0.770 | 15.6 | plateau |
| **e150** | **0.776** | 15.5 | **PASS Tier-1** |
| random baseline | 0.074 | — | null |

Train-cache e150: 0.979 (expected overfit; delta vs holdout flags the memorization
guardrail by magnitude but is NOT memorization — the eval set is disjoint and the
growth curve e49→e150 shows genuine learning).

## Files

- `fidelity_ar_s0_e150.txt` — pre-fix (invalid, constant-prefix training)
- `fidelity_ar_s0_e150_v2.txt` — pre-fix (invalid)
- `fidelity_ar_s0_e150_v3.txt` — pre-fix (invalid)
- `fidelity_ar_s0_e150_real.txt` — **canonical** (e150, real conditioning, r3e)
- e49/e99 runs logged inline in the ledger's evidence table

## Model state

`checkpoints/ar/ar_s0_e150.pt` (d=384, L=8, 150 epochs, loss 0.0159)
