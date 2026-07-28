# Edge-Count Enrichment (Exp 1.6, Arm A) — Asset Directory

**Verdict: FAIL** — density bias controls edge count but reduces fidelity at every scale.

## Canonical result

- **`edge_count_sweep.txt`** — **N=128 definitive sweep** (186s, RTX 4090). Baseline typed-F1 0.088, all biased scales below. This is the authoritative result cited in the ledger.
  
## Pilot / intermediate runs (preserved for reproducibility, not authoritative)

- `edge_count_sweep_8.txt` — N=8 GPU pilot (first run, monotonic decline)
- `edge_count_sweep_32.txt` — N=32 intermediate run
- `smoke_test_cpu.txt` — N=5 CPU verification (max_nodes=64)
- `edge_count_sweep_cpu.txt` — N=5 CPU (stalled)
- `edge_count_sweep_cpu64.txt` — N=5 CPU, max_nodes=64 (timed out)
- `edge_count_sweep_quick.txt` — N=8 GPU pilot (second run, conflicting result — prompted investigation)

## Gate registration

Pre-registered in `docs/02_EXPERIMENTS_AND_RESULTS.md` (commit `fa32677`).

## Scripts

- `scripts/evaluate_edge_count_enrichment.py`
- `scripts/ablation_edge_count.py` (not yet run — FAIL verdict reached before ablation needed)
