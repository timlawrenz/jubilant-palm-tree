# Decode-Time Solver: Index Collision Resolution Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the final deterministic repair step in the Judicial Constraint Solver. The solver must resolve `input_index` collisions (e.g., two branches claiming `index=0`) by falling back on the DiT's categorical logits to ensure distinct routing assignments, thereby unblocking the final ~27% Out-Degree failures and massive ~93% In-Degree failures to maximize SVR.

**Architecture:**
1. Extend `src/models/constraint_solver.py` with `_repair_index_collisions`.
2. Extract the categorical logits (`continuous_matrix[:, 2:6]`) representing the probability distribution across indices (0-3).
3. For **Execution Edges**: If a node has multiple outgoing exec edges with identical indices, sort the edges by their continuous presence probability. The strongest edge keeps its requested index; weaker edges are reassigned to the next highest available probability from their respective categorical logits.
4. For **Data Edges**: If a node has multiple incoming data edges with identical indices (specifically `Message` motifs requiring sequential arguments), apply the same logit-fallback resolution.
5. Add unit tests proving collision resolution.

**Tech Stack:** Python, PyTorch, Pytest

---

### Task 1: Implement Index Collision Resolution

**Objective:** Write the logic to resolve index collisions using categorical logit fallbacks.

**Files:**
- Modify: `src/models/constraint_solver.py`

**Step 1: Write the logic**

```python
    @classmethod
    def _repair_index_collisions(cls, adj: torch.Tensor, index_logits: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list) -> torch.Tensor:
        """
        Resolves input_index collisions for execution and data edges using categorical logits.
        adj is [3, N, N]. Channel 0 = presence, Channel 1 = type, Channel 2 = input_index.
        index_logits is [4, N, N].
        """
        presence = adj[0].bool()
        edge_type = adj[1].int()
        input_index = adj[2].int()

        is_exec = presence & (edge_type == 0)
        is_data = presence & (edge_type == 1)

        def resolve_collisions(edges, is_source_grouped=True):
            # Group edges by source (for exec out) or by target (for data in)
            groups = {n: [] for n in valid_nodes}
            for u in valid_nodes:
                for v in valid_nodes:
                    if edges[u, v]:
                        key = u if is_source_grouped else v
                        groups[key].append((u, v))

            for key, edge_list in groups.items():
                if len(edge_list) <= 1:
                    continue

                # Check for collisions
                used_indices = {}
                for u, v in edge_list:
                    idx = input_index[u, v].item()
                    if idx not in used_indices:
                        used_indices[idx] = []
                    used_indices[idx].append((u, v))

                for idx, colliding_edges in used_indices.items():
                    if len(colliding_edges) <= 1:
                        continue

                    # Collision detected! Sort by presence probability descending
                    colliding_edges.sort(key=lambda e: soft_presence[e[0], e[1]].item(), reverse=True)
                    
                    # The strongest edge gets to keep the index
                    winner = colliding_edges[0]
                    losers = colliding_edges[1:]

                    # Reassign losers using their next best logits
                    occupied = set(input_index[u, v].item() for u, v in edge_list)
                    
                    for lu, lv in losers:
                        logits = index_logits[:, lu, lv]
                        # Sort indices by logit score descending
                        sorted_indices = torch.argsort(logits, descending=True).tolist()
                        
                        # Find the highest scored index that isn't occupied
                        new_idx = None
                        for candidate_idx in sorted_indices:
                            if candidate_idx not in occupied:
                                new_idx = candidate_idx
                                break
                                
                        # If all 4 slots are occupied (rare), just force an arbitrary unused one up to a reasonable cap (e.g. max arity + 1)
                        if new_idx is None:
                            new_idx = max(occupied) + 1 if occupied else 0
                            
                        input_index[lu, lv] = new_idx
                        occupied.add(new_idx)

        # Apply to Execution Out-Edges (grouped by source node)
        resolve_collisions(is_exec, is_source_grouped=True)
        # Apply to Data In-Edges (grouped by target node)
        resolve_collisions(is_data, is_source_grouped=False)

        adj[2] = input_index
        return adj
```

*Note: Update `discretize_and_repair` to extract `index_logits = continuous_matrix[:, 2:6]` and pass it to this method. Order of application: Out-Degree -> In-Degree -> Index Collision -> Acyclicity.*

**Step 2: Commit**
```bash
git add src/models/constraint_solver.py
git commit -m "feat: implement deterministic index collision resolution in constraint solver"
```

---

### Task 2: Unit Test Collision Resolver

**Objective:** Prove that the solver resolves index ties by assigning the weaker edge to its second-choice logit.

**Files:**
- Modify: `tests/test_constraint_solver.py`

**Step 1: Write the test**

```python
def test_repair_index_collisions():
    # Node 0 has 2 outgoing exec edges to 1 and 2. Both currently claim index 0.
    adj = torch.zeros(3, 3, 3)
    adj[0, 0, 1] = 1; adj[1, 0, 1] = 0; adj[2, 0, 1] = 0
    adj[0, 0, 2] = 1; adj[1, 0, 2] = 0; adj[2, 0, 2] = 0
    
    valid_nodes = [0, 1, 2]
    
    # 0->1 is the stronger edge (0.9 vs 0.6)
    soft_presence = torch.zeros(3, 3)
    soft_presence[0, 1] = 0.9
    soft_presence[0, 2] = 0.6
    
    # Provide categorical logits (4 channels)
    index_logits = torch.zeros(4, 3, 3)
    # Edge 0->1 strongly prefers index 0
    index_logits[:, 0, 1] = torch.tensor([5.0, 1.0, 0.0, 0.0])
    # Edge 0->2 prefers index 0, but its second choice is index 1
    index_logits[:, 0, 2] = torch.tensor([4.0, 3.0, 0.0, 0.0])
    
    repaired_adj = ConstraintSolver._repair_index_collisions(adj, index_logits, soft_presence, valid_nodes)
    
    # The stronger edge (0->1) keeps index 0
    assert repaired_adj[2, 0, 1].item() == 0
    # The weaker edge (0->2) should be bumped to its second choice, index 1
    assert repaired_adj[2, 0, 2].item() == 1
```

**Step 2: Run test**
Run: `pytest tests/test_constraint_solver.py::test_repair_index_collisions -v`
Expected: PASS

**Step 3: Commit**
```bash
git add tests/test_constraint_solver.py
git commit -m "test: add unit test for index collision resolution"
```

---

### Task 3: Final SVR Evaluation

**Objective:** Run the completed constraint solver against the 160-sample baseline to measure the final dataset SVR.

**Files:**
- Action: Execute `scripts/evaluate_repaired_solver.py`

**Step 1: Execute**
Run: `PYTHONPATH=. python scripts/evaluate_repaired_solver.py > docs/assets/exp/decode-time-solver/final_solver_eval.txt`

**Step 2: Document Results**
Update `docs/02_EXPERIMENTS_AND_RESULTS.md` with the final benchmark metrics. If SVR climbs significantly toward the 83% dataset ceiling, the Decode-Time Constraint Solver hypothesis is formally validated.

**Step 3: Commit**
```bash
git add docs/
git commit -m "docs: log final constraint solver evaluation with index resolution"
```

---

### Execution Handoff
Plan complete. Run this to execute the Index Collision Resolution and measure the final SVR milestone.