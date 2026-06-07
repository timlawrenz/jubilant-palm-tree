# Validator 6th Law: Global Reachability Spine Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Formally define and implement the "6th Law of Physics" (Global Reachability Spine) in `GraphValidator`. The Executability Audit proved that graphs can pass the first 5 local laws while lacking a single global entry point or a connected execution path. This law enforces that there is exactly one Entry Boundary, and all execution nodes are reachable from it.

**Architecture:**
1. Extend `src/models/validation.py` with `_check_global_spine`.
2. The logic must verify:
   a. Exactly ONE node is a `MotifType.BOUNDARY` with 0 incoming execution edges (The Entry).
   b. Starting a directed BFS/DFS along `EXECUTION` edges from The Entry, every single node that participates in the execution plane (has incoming or outgoing exec edges) must be visited.
3. Update `evaluate_batch` to include `global_spine_pass`.
4. Add unit tests for `_check_global_spine`.
5. Re-run `scripts/evaluate_repaired_solver.py` to see the true SVR drop when held to this strict global standard.

**Tech Stack:** Python, PyTorch, Pytest

---

### Task 1: Implement `_check_global_spine` in GraphValidator

**Objective:** Write the logic for the 6th Law.

**Files:**
- Modify: `src/models/validation.py`

**Step 1: Write the logic**
*(Add inside `GraphValidator`)*

```python
    @classmethod
    def _check_global_spine(cls, valid_nodes, motif_seq, exec_out) -> bool:
        """
        Rule 6: Global Reachability Spine
        1. Exactly ONE Boundary node with 0 incoming execution edges (Entry).
        2. All nodes participating in execution must be reachable from the Entry.
        """
        # Build incoming execution edges
        exec_in = {n: [] for n in valid_nodes}
        for u in valid_nodes:
            for idx, v in exec_out[u]:
                exec_in[v].append(u)

        # 1. Find Entry Boundaries
        entry_nodes = []
        for n in valid_nodes:
            motif = cls.MOTIF_MAP[motif_seq[n].item()]
            if motif == MotifType.BOUNDARY and len(exec_in[n]) == 0:
                entry_nodes.append(n)

        if len(entry_nodes) != 1:
            return False  # Must have exactly one clear entry point

        entry_node = entry_nodes[0]

        # 2. Identify all nodes that participate in the execution plane
        exec_participants = set()
        for n in valid_nodes:
            if len(exec_out[n]) > 0 or len(exec_in[n]) > 0:
                exec_participants.add(n)
                
        # The entry node is always part of the execution plane
        exec_participants.add(entry_node)

        # 3. Directed BFS from Entry along execution edges
        visited = set()
        q = [entry_node]
        
        while q:
            curr = q.pop(0)
            if curr not in visited:
                visited.add(curr)
                for idx, tgt in exec_out.get(curr, []):
                    if tgt not in visited:
                        q.append(tgt)

        # 4. Check if all execution participants were reached
        return exec_participants.issubset(visited)
```
*Note: Update `evaluate_batch` to track `global_spine_pass`, initialize it to 0, aggregate it, and enforce it in the `perfect_graphs` strict AND condition.*

**Step 2: Commit**
```bash
git add src/models/validation.py
git commit -m "feat: implement 6th law (global spine) in graph validator"
```

---

### Task 2: Unit Test the 6th Law

**Objective:** Prove the spine checker accepts valid paths and rejects spineless graphs (e.g. multiple entries, or disconnected loops).

**Files:**
- Modify: `tests/test_validation.py`

**Step 1: Write the test**

```python
def test_check_global_spine():
    from src.models.validation import GraphValidator
    from src.execution_engine.schema import MotifType
    import torch
    
    def get_motif_seq(*motifs):
        return torch.tensor(list(motifs))
        
    valid_nodes = [0, 1, 2, 3]
    # 0: Boundary, 1: Sequence, 2: Boundary, 3: Sequence
    motif_seq = get_motif_seq(1, 2, 1, 2)
    
    # Valid: 0 -> 1 -> 2 (3 is not in exec plane)
    exec_out_valid = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [],
        3: []
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_valid) is True

    # Invalid: Two entries (0 and 2 are boundaries with 0 in-edges)
    exec_out_two_entries = {
        0: [(0, 1)],
        1: [],
        2: [(0, 3)],
        3: []
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_two_entries) is False

    # Invalid: Valid entry, but disconnected loop (3 -> 3)
    exec_out_disconnected = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [],
        3: [(0, 3)]
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_disconnected) is False
```
*Note: Also update `test_evaluate_batch_perfect_graph` in `tests/test_validation.py` to assert `global_spine_pass` == 100.0.*

**Step 2: Run test**
Run: `pytest tests/test_validation.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add tests/test_validation.py
git commit -m "test: add unit tests for 6th law global spine validation"
```

---

### Task 3: Re-evaluate Final SVR

**Objective:** Measure how the previously "perfect" solver performs against the new, complete 6-Law specification.

**Files:**
- Action: Execute `scripts/evaluate_repaired_solver.py`

**Step 1: Execute**
Run: `PYTHONPATH=. python scripts/evaluate_repaired_solver.py > docs/assets/exp/validator-6th-law/strict_eval.txt`
*(Ensure the directory `docs/assets/exp/validator-6th-law` is created first).*

**Step 2: Document Results**
Update `docs/02_EXPERIMENTS_AND_RESULTS.md` reflecting the new, lower, but honest SVR.

**Step 3: Commit**
```bash
git add docs/
git commit -m "docs: measure and log SVR drop against strict 6th law"
```

---

### Execution Handoff
Plan complete. Run this to add the 6th law and reset our baseline.
