# Routing Fidelity of the Executive Branch — Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Measure whether the Structural DiT (Executive Branch) routes a given Bill of Materials (the motif sequence) into the *intended* program graph, or merely into *a* structurally-valid graph with those ingredients. Today we only know generated 6-Law graphs execute; we do NOT know if they reconstruct the ground-truth routing they were conditioned on. This experiment quantifies controllability/fidelity.

**Architecture (the three-branch contract, per `docs/01_VISION_AND_ARCHITECTURE.md`):**
- We feed the DiT a *real* ground-truth motif sequence (the Bill of Materials) — exactly the signal the future Legislative Branch will emit.
- The DiT (Executive) generates a routing heat-map via the standard 20-step flow ODE; the ConstraintSolver (Judicial) projects it to a discrete DAG.
- We then compare the generated graph against the ground-truth graph that the motif sequence came from, on the real (non-padding) node block.

**Key metrics (all computed only over valid `[num_nodes, num_nodes]` block, ignoring padding):**
1. **Edge-set F1** on the presence channel (precision/recall/F1 of generated edges vs. ground-truth edges).
2. **Typed-edge F1**: an edge counts as a true positive only if presence AND edge_type (exec/data) match.
3. **Baseline contrast**: F1 of a random graph with the same edge count, to prove the DiT beats chance.
4. **Ambiguity probe (Task 4)**: feed the SAME noise seed + DIFFERENT motif sequences; confirm the output topology changes. Proves the motif signal actually drives generation (controllability sanity check).

**Tech Stack:** Python, PyTorch. Reuse `NeuralUniversalMachineDiT`, `ExecutionGraphDataset`, `ConstraintSolver`. Pre-trained ckpt: `checkpoints/num_dit_epoch_340.pt`.

**Pre-flight (read these before coding):**
- `src/models/dataset.py` — `__getitem__` returns dict `{adjacency:[3,128,128], motifs:[128], padding_mask:[128,128]}`. Channel 0 = presence, ch1 = edge_type (0 exec / 1 data), ch2 = input_index.
- `src/models/constraint_solver.py` — `ConstraintSolver.discretize_and_repair(continuous_matrix: [B,6,N,N], motifs: [B,N]) -> discrete [B,3,N,N]`.
- `src/models/model.py` — `NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12)`, `forward(x_t:[B,6,N,N], t:[B], motifs:[B,N])`. The 20-step ODE pattern: see `scripts/evaluate_large_n_executability.py` (copy the sampling loop verbatim — it is the canonical generation procedure).

---

### Task 1: Edge-set fidelity metric helper

**Objective:** A pure function that, given a generated discrete adjacency, a ground-truth adjacency, and node count, returns precision/recall/F1 for both untyped and typed edge sets.

**Files:**
- Create: `src/models/fidelity.py`

**Step 1: Write the helper**

```python
import torch

def edge_fidelity(gen_adj: torch.Tensor, gt_adj: torch.Tensor, num_nodes: int) -> dict:
    """
    gen_adj, gt_adj: [3, N, N] discrete adjacency (ch0 presence, ch1 edge_type, ch2 index).
    num_nodes: number of real (non-padding) nodes.
    Returns precision/recall/F1 for untyped (presence-only) and typed (presence+edge_type) edge sets.
    """
    g = gen_adj[:, :num_nodes, :num_nodes]
    t = gt_adj[:, :num_nodes, :num_nodes]

    gen_pres = (g[0] > 0.5)
    gt_pres = (t[0] > 0.5)

    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    # Untyped (presence only)
    tp = int((gen_pres & gt_pres).sum())
    fp = int((gen_pres & ~gt_pres).sum())
    fn = int((~gen_pres & gt_pres).sum())
    up, ur, uf1 = prf(tp, fp, fn)

    # Typed (presence AND matching edge_type)
    type_match = (g[1].round() == t[1].round())
    typed_tp = int((gen_pres & gt_pres & type_match).sum())
    typed_fp = int((gen_pres & ~(gt_pres & type_match)).sum())
    typed_fn = int((gt_pres & ~(gen_pres & type_match)).sum())
    tp_, tr_, tf1 = prf(typed_tp, typed_fp, typed_fn)

    return {
        "untyped_precision": up, "untyped_recall": ur, "untyped_f1": uf1,
        "typed_precision": tp_, "typed_recall": tr_, "typed_f1": tf1,
        "gen_edges": int(gen_pres.sum()), "gt_edges": int(gt_pres.sum()),
    }
```

**Step 2: Commit**
```bash
git add src/models/fidelity.py
git commit -m "feat: add edge-set fidelity metric (untyped + typed P/R/F1)"
```

---

### Task 2: Unit test the fidelity metric

**Objective:** Prove the metric returns F1=1.0 for an identical graph, 0.0 for a disjoint one, and a sane mid value for partial overlap.

**Files:**
- Create: `tests/test_fidelity.py`

**Step 1: Write the test**

```python
import torch
from src.models.fidelity import edge_fidelity

def _adj(edges, n=4):
    a = torch.zeros(3, n, n)
    for (u, v, et) in edges:
        a[0, u, v] = 1.0
        a[1, u, v] = float(et)
    return a

def test_identical_is_perfect():
    a = _adj([(0,1,0),(1,2,0)])
    r = edge_fidelity(a, a, 4)
    assert r["untyped_f1"] == 1.0
    assert r["typed_f1"] == 1.0

def test_disjoint_is_zero():
    a = _adj([(0,1,0)])
    b = _adj([(2,3,0)])
    r = edge_fidelity(a, b, 4)
    assert r["untyped_f1"] == 0.0

def test_wrong_type_breaks_typed_only():
    gt = _adj([(0,1,0)])         # exec edge
    gen = _adj([(0,1,1)])        # same position, data edge
    r = edge_fidelity(gen, gt, 4)
    assert r["untyped_f1"] == 1.0   # position matches
    assert r["typed_f1"] == 0.0     # type mismatch
```

**Step 2: Run**
Run: `PYTHONPATH=. .venv/bin/pytest tests/test_fidelity.py -v`
Expected: 3 passed.

**Step 3: Commit**
```bash
git add tests/test_fidelity.py
git commit -m "test: unit tests for edge fidelity metric"
```

---

### Task 3: Fidelity evaluation script

**Objective:** Over N real ground-truth graphs, generate via DiT+Solver conditioned on each graph's motif sequence, compute fidelity vs. that graph, and contrast against a random-edge baseline of matched density.

**Files:**
- Create: `scripts/evaluate_routing_fidelity.py`

**Step 1: Write the script.** Mirror the sampling loop in `scripts/evaluate_large_n_executability.py` exactly (20-step ODE, `ConstraintSolver.discretize_and_repair`). For each graph in N≥512 samples:
- derive `num_nodes` from `(motifs != 0).sum()`.
- generate the graph from its own motif sequence (use `shuffle=False` and `augment_permutation=False` in the dataset so the ground-truth tensor and conditioning correspond).
- `edge_fidelity(gen_adj, gt_adj, num_nodes)`.
- Build a random baseline: scatter `gt_edges` random edges over the `num_nodes` block, score against ground truth.
- Aggregate means for: untyped F1, typed F1, gen vs gt edge counts, and the random-baseline untyped F1.

Print a results block. Save raw output (see Task 5).

**Step 2: Commit**
```bash
git add scripts/evaluate_routing_fidelity.py
git commit -m "eval: add routing-fidelity evaluation script (DiT reconstruction vs ground truth)"
```

---

### Task 4: Conditioning ablation (controllability probe)

**Objective:** Prove the motif signal actually drives generation: hold the noise seed fixed, vary the motif sequence, confirm the output topology changes.

**Files:**
- Create: `scripts/ablation_motif_controllability.py`

**Step 1: Write the script.** Fix a single random `x` seed (`torch.manual_seed`). Pick 2 distinct real motif sequences A and B. Run the full 20-step ODE + solver for (seed, A) and (seed, B). Report the untyped edge-overlap (Jaccard) between outputA and outputB over the union node block. Low overlap ⇒ motif conditioning genuinely steers routing; high overlap ⇒ the motif signal is being ignored (a red flag to investigate).

**Step 2: Commit**
```bash
git add scripts/ablation_motif_controllability.py
git commit -m "eval: add motif controllability ablation (same seed, different bill-of-materials)"
```

---

### Task 5: Run, capture evidence, document

**Objective:** Execute both scripts, persist raw logs to the branch asset folder, and write up results + interpretation.

**Files:**
- Create dir: `docs/assets/exp/routing-fidelity/`
- Modify: `docs/02_EXPERIMENTS_AND_RESULTS.md`, `docs/03_EXPERIMENT_TREE.md`

**Step 1: Run (GPU job — notify the user before launching; they prefer to run training/GPU jobs, but these are inference-only eval scripts so confirm).**
```bash
mkdir -p docs/assets/exp/routing-fidelity
PYTHONPATH=. .venv/bin/python scripts/evaluate_routing_fidelity.py > docs/assets/exp/routing-fidelity/fidelity_eval.txt 2>&1
PYTHONPATH=. .venv/bin/python scripts/ablation_motif_controllability.py > docs/assets/exp/routing-fidelity/controllability_ablation.txt 2>&1
```

**Step 2: Interpret against the three outcomes:**
- **High typed-F1 (and beats random baseline by a wide margin):** Executive Branch is controllable — the milestone is genuinely proven. Next frontier = Legislative Branch (intent → motifs).
- **Low F1 but valid + low ablation overlap:** DiT makes valid graphs but ignores fine routing detail / motif array underspecifies the program. Bill of Materials needs enrichment.
- **High ablation overlap (motifs barely change output):** conditioning is too weak — that's the bug to fix before anything else.

**Step 3: Update docs.** Add a "Routing Fidelity" subsection to `02_EXPERIMENTS_AND_RESULTS.md` with the numbers and the chosen interpretation. Mark the experiment `[CONCLUDED]` in `03_EXPERIMENT_TREE.md` with a one-line result, and set the follow-on `[NEXT]` accordingly.

**Step 4: Commit + merge.**
```bash
git add docs/assets/exp/routing-fidelity/ docs/02_EXPERIMENTS_AND_RESULTS.md docs/03_EXPERIMENT_TREE.md
git commit -m "docs: evaluate and document routing fidelity of the Executive Branch"
git checkout main && git merge exp/routing-fidelity && git push
```

---

### Risks / Open Questions
- **Permutation alignment:** the dataset shuffles node order when `augment_permutation=True`. MUST set `augment_permutation=False` so the conditioning motif array and the ground-truth adjacency share an index space; otherwise fidelity is meaningless. Verify this in Task 3.
- **Motif ambiguity is a real confound:** many programs share a motif sequence, so even a perfect Executive Branch may not reproduce the *specific* source graph. Interpret low fidelity carefully — it may indict the Bill of Materials (a Legislative-Branch finding) rather than the DiT. The ablation (Task 4) helps disambiguate.
- **Inference-only:** no training; runs on the existing checkpoint. Safe to run without a long GPU commitment, but still confirm with the user before launching per their GPU-usage preference.

### Execution Handoff
Plan complete and saved. This stays inside the proven Executive+Judicial branches and measures whether the DiT is *faithful*, not just *valid* — the core of the "GNN generates a code path" milestone — before we open the Legislative (intent) branch.
