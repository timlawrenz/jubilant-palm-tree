# Training-Time Enrichment (Exp 1.9, Arm D) — Asset Directory

**Verdict: FAIL** — in-channel degree fine-tune is inert; the enrichment program is closed.

## Canonical results (N=512, 0 errors, same harness as Exp 1.5)

| File | Model | typed-F1 | untyped-F1 | gen_edges |
|---|---|---|---|---|
| `fidelity_base340.txt` | Base 340 (frozen control) | 0.0895 | 0.1474 | 181.1 |
| `fidelity_signal_e20.txt` | Signal arm e20 (real degrees) | 0.0894 | 0.1438 | 168.5 |
| `fidelity_null_e20.txt` | Null arm e20 (random degrees) | 0.0769 | 0.1385 | 163.0 |

Signal−null separation: +0.0125 (gate ≥ 0.05). FAIL criterion (≤ decode-best 0.109) fires.

## Training logs

- `checkpoints/tte/train_signal_s0.log` — 20 epochs, loss 0.0976→0.0985, nan_guard=0
- `checkpoints/tte/train_null_s0.log` — 20 epochs, loss →0.0987, nan_guard=0

## Gate registration

Pre-registered in `docs/02_EXPERIMENTS_AND_RESULTS.md` (commit `2ebffbc`, BEFORE code).

## Scripts

- `scripts/finetune_degree_profile.py` — two-arm fine-tune (signal/null)
- `scripts/evaluate_tte.py` — routing-fidelity eval for fine-tuned checkpoints
- Eval runs used the inline `python -c` path (project-known script-hang workaround)