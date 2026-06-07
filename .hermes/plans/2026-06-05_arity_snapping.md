# Decode-Time Solver: Arity Snapping Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the final stages of the Decode-Time Judicial Constraint Solver: deterministic top-K arity snapping for Execution Out-Degree and Data In-Degree. This addresses the remaining ~6% degree bottlenecks to push Syntactic Validity Rate (SVR) toward the 83% dataset ceiling.

**Architecture:**
1. Extend `src/models/constraint_solver.py` with two new repair functions: `_repair_exec_out_degree` and `_repair_data_in_degree`.
2. These functions will query the `EXPECTED_EXEC_OUT` arity definitions.
3. For nodes that over-generate edges (e.g. a Sequence node with 3 exec outputs instead of 1), the solver will use the DiT's continuous `soft_presence` probabilities to keep the highest-confidence edges and prune the rest.
4. For nodes that under-generate (e.g. a Condition node with 0 inputs), the solver will look at the `soft_presence` heatmap for potential valid incoming edges that were suppressed by the global >0.5 threshold, and snap the highest-confidence one to 1.
5. Add unit tests for both snapping behaviors.

**Tech Stack:** Python, PyTorch, Pytest

---

### Task 1: Execution Out-Degree Repair

**Objective:** Ensure Sequence, State, and Message nodes have exactly 1 exec out (or 0 if terminal), and Condition/Loop nodes have exactly 2.

**Files:**
- Modify: `src/models/constraint_solver.py`

**Step 1: Write the logic**
*(Add inside `ConstraintSolver`)*

```python
    @classmethod
    def _repair_exec_out_degree(cls, adj: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list, motifs: list) -> torch.Tensor:
        """
        Enforces execution out-degree limits based on Motif type.
        adj is [3, N, N]. Channel 0 = presence, Channel 1 = type (0=Exec).
        """
        from src.rlaif.structural_loss import EXPECTED_EXEC_OUT
        
        # We only look at execution edges (type == 0)
        is_exec = (adj[1] == 0) & (adj[0] == 1)
        
        for u in valid_nodes:
            motif_id = motifs[u]
            target_out = EXPECTED_EXEC_OUT.get(motif_id, 0.0)
            
            # For this MVP, we enforce a strict max cap based on the motif.
            # Condition/Loop: max 2. Sequence/State/Message: max 1. Boundary: max 1.
            max_allowed = int(target_out)
            if max_allowed == 0 and motif_id != 1: 
                # If target is 0.0 but it's not a boundary, it's padded/invalid, skip
                continue
            if motif_id == 1:
                max_allowed = 1 # Boundary can be 0 or 1
                
            current_out_edges = [v for v in valid_nodes if is_exec[u, v]]
            
            # If over-degree, keep the top-K highest probability edges and prune the rest
            if len(current_out_edges) > max_allowed:
                # Sort by soft probability descending
                current_out_edges.sort(key=lambda v: soft_presence[u, v].item(), reverse=True)
                
                # Prune edges beyond the allowed max
                for v_to_prune in current_out_edges[max_allowed:]:
                    adj[0, u, v_to_prune] = 0 # Drop presence
                    
        return adj
```
*Note: Make sure to update `discretize_and_repair` to call this new method before `_repair_acyclicity`.*

**Step 2: Commit**
```bash
git add src/models/constraint_solver.py
git commit -m "feat: implement execution out-degree arity snapping"
```

---

### Task 2: Data In-Degree Repair

**Objective:** Ensure Condition, Loop, and State-Write nodes have at least 1 data input.

**Files:**
- Modify: `src/models/constraint_solver.py`

**Step 1: Write the logic**

```python
    @classmethod
    def _repair_data_in_degree(cls, adj: torch.Tensor, soft_presence: torch.Tensor, soft_data_type: torch.Tensor, valid_nodes: list, motifs: list) -> torch.Tensor:
        """
        Enforces data in-degree minimums.
        Condition/Loop/State-Writes need >= 1 data input.
        If they have 0, we find the highest probability incoming data edge in the continuous map and force it to 1.
        """
        is_data = (adj[1] == 1) & (adj[0] == 1)
        is_exec = (adj[1] == 0) & (adj[0] == 1)
        
        for v in valid_nodes:
            motif_id = motifs[v]
            
            # Check if this node REQUIRES a data input
            requires_data = False
            if motif_id in (3, 4): # Condition, Loop
                requires_data = True
            elif motif_id == 5: # State (Write)
                # It's a write if it has any outgoing exec edges
                if any(is_exec[v, tgt] for tgt in valid_nodes):
                    requires_data = True
                    
            if not requires_data:
                continue
                
            current_in_edges = [u for u in valid_nodes if is_data[u, v]]
            
            # If under-degree (0 inputs), find the strongest candidate and force it
            if len(current_in_edges) == 0:
                best_u = None
                best_prob = -1.0
                
                for u in valid_nodes:
                    if u == v: continue # Prevent self-loops trivially
                    # The probability of it being a data edge is presence * data_type
                    prob = (soft_presence[u, v] * soft_data_type[u, v]).item()
                    if prob > best_prob:
                        best_prob = prob
                        best_u = u
                        
                if best_u is not None:
                    # Force the edge into existence
                    adj[0, best_u, v] = 1 # Presence
                    adj[1, best_u, v] = 1 # Type = Data
                    # Let the argmax index logits stay as they are in adj[2]
                    
        return adj
```
*Note: Update `discretize_and_repair` to extract `soft_data_type = torch.sigmoid(continuous_matrix[:, 1])` and pass it to this method. Order of application: Out-Degree -> In-Degree -> Acyclicity.*

**Step 2: Commit**
```bash
git add src/models/constraint_solver.py
git commit -m "feat: implement data in-degree rescue via probability threshold boosting"
```

---

### Task 3: Unit Tests

**Objective:** Prove the capping and rescuing behaviors work.

**Files:**
- Modify: `tests/test_constraint_solver.py`

**Step 1: Write the tests**
*(Agent: Write two Pytest functions. `test_repair_exec_out_degree` should construct a Sequence node (motif 2) with 3 outgoing exec edges, give them different probabilities, and assert only the highest probability edge remains. `test_repair_data_in_degree` should construct a Condition node (motif 3) with 0 data inputs, provide a continuous heatmap, and assert the highest-probability source becomes a hardcoded `1`.)*

**Step 2: Run test**
Run: `pytest tests/test_constraint_solver.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add tests/test_constraint_solver.py
git commit -m "test: add unit tests for degree arity snapping"
```

---

### Task 4: Re-evaluate Baseline SVR

**Objective:** Run the full solver pipeline against the 160-sample baseline to measure the final SVR leap.

**Files:**
- Action: Execute `scripts/evaluate_repaired_solver.py`

**Step 1: Execute**
Run: `PYTHONPATH=. python scripts/evaluate_repaired_solver.py > docs/assets/exp/decode-time-solver/full_solver_eval.txt`

**Step 2: Document Results**
Update `docs/02_EXPERIMENTS_AND_RESULTS.md` with the new metrics.

**Step 3: Commit**
```bash
git add docs/
git commit -m "docs: log full constraint solver evaluation results"
```

---

### Execution Handoff
Plan complete. Run this to execute the Degree Arity snapping.