# Decode-Time Solver: Acyclicity Repair Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Implement the first stage of the Decode-Time Judicial Constraint Solver to deterministically repair Data-Plane Cycles. This should boost the `acyclic_data_pass` metric from 1.25% to 100% by breaking cycles at the weakest probabilistic links.

**Architecture:** 
1. We will create a new module `src/models/constraint_solver.py` that wraps the naive discretization step but adds a deterministic repair pass.
2. The repair algorithm will extract the data-edge subgraph.
3. It will use Depth-First Search (DFS) to detect back-edges (cycles).
4. When a cycle is detected, it traces the cycle path and removes the edge with the *lowest* continuous `soft_presence` probability score.
5. It repeats until the data plane is a perfect Directed Acyclic Graph (DAG).

**Tech Stack:** Python, PyTorch, Pytest

---

### Task 1: Create the Constraint Solver Module

**Objective:** Write the core `ConstraintSolver` class with the cycle-breaking algorithm.

**Files:**
- Create: `src/models/constraint_solver.py`

**Step 1: Write the implementation**

```python
import torch
from src.rlaif.naive_discretizer import naive_discretize

class ConstraintSolver:
    @classmethod
    def discretize_and_repair(cls, continuous_matrix: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        """
        Discretizes the continuous heatmap and applies deterministic topological repairs.
        """
        # 1. Start with the naive thresholding
        discrete = naive_discretize(continuous_matrix, motifs)
        
        # 2. Extract soft probabilities to resolve conflicts
        soft_presence = torch.sigmoid(continuous_matrix[:, 0]) # [B, N, N]
        
        B = discrete.shape[0]
        
        # Apply repairs per graph
        for b in range(B):
            valid_nodes = [n for n, m in enumerate(motifs[b].tolist()) if m != 0]
            if not valid_nodes:
                continue
                
            discrete[b] = cls._repair_acyclicity(discrete[b], soft_presence[b], valid_nodes)
            
        return discrete

    @classmethod
    def _repair_acyclicity(cls, adj: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list) -> torch.Tensor:
        """
        Detects data-plane cycles and breaks them by removing the weakest edge.
        adj is [3, N, N] where channel 0 is presence, channel 1 is edge_type (1=Data).
        """
        while True:
            cycle = cls._find_cycle(adj, valid_nodes)
            if not cycle:
                break # DAG achieved!
                
            # Find the weakest link in the cycle
            min_prob = float('inf')
            weakest_edge = None
            
            for u, v in cycle:
                prob = soft_presence[u, v].item()
                if prob < min_prob:
                    min_prob = prob
                    weakest_edge = (u, v)
                    
            # Break the cycle
            u, v = weakest_edge
            adj[0, u, v] = 0 # Remove presence
            
        return adj

    @classmethod
    def _find_cycle(cls, adj: torch.Tensor, valid_nodes: list):
        """Returns a list of edges (u, v) forming a cycle, or None."""
        presence = adj[0].bool().tolist()
        edge_type = adj[1].int().tolist()
        
        # Build adjacency list for DATA edges only
        data_out = {n: [] for n in valid_nodes}
        for u in valid_nodes:
            for v in valid_nodes:
                if presence[u][v] and edge_type[u][v] == 1:
                    data_out[u].append(v)
                    
        visited = set()
        rec_stack = set()
        parent = {}
        
        def dfs(u):
            visited.add(u)
            rec_stack.add(u)
            
            for v in data_out[u]:
                if v not in visited:
                    parent[v] = u
                    cycle = dfs(v)
                    if cycle: return cycle
                elif v in rec_stack:
                    # Cycle detected. Trace back to build the path.
                    path = [(u, v)]
                    curr = u
                    while curr != v:
                        p = parent[curr]
                        path.append((p, curr))
                        curr = p
                    return path[::-1]
                    
            rec_stack.remove(u)
            return None
            
        for n in valid_nodes:
            if n not in visited:
                cycle = dfs(n)
                if cycle: return cycle
                
        return None
```

**Step 2: Commit**
```bash
git add src/models/constraint_solver.py
git commit -m "feat: implement deterministic acyclicity repair in constraint solver"
```

---

### Task 2: Unit Test the Cycle Breaker

**Objective:** Prove the cycle breaker drops the *weakest* edge in a cycle.

**Files:**
- Create: `tests/test_constraint_solver.py`

**Step 1: Write the tests**

```python
import torch
import pytest
from src.models.constraint_solver import ConstraintSolver

def test_find_and_repair_cycle():
    # 3 nodes forming a cycle: 0 -> 1 -> 2 -> 0
    adj = torch.zeros(3, 3, 3)
    # Set presence
    adj[0, 0, 1] = 1
    adj[0, 1, 2] = 1
    adj[0, 2, 0] = 1
    # Set edge_type to 1 (Data)
    adj[1, 0, 1] = 1
    adj[1, 1, 2] = 1
    adj[1, 2, 0] = 1
    
    valid_nodes = [0, 1, 2]
    
    # 1->2 is the weakest edge (0.3)
    soft_presence = torch.zeros(3, 3)
    soft_presence[0, 1] = 0.9
    soft_presence[1, 2] = 0.3
    soft_presence[2, 0] = 0.8
    
    repaired_adj = ConstraintSolver._repair_acyclicity(adj, soft_presence, valid_nodes)
    
    # The weakest edge should be removed
    assert repaired_adj[0, 1, 2].item() == 0
    
    # The strong edges should remain
    assert repaired_adj[0, 0, 1].item() == 1
    assert repaired_adj[0, 2, 0].item() == 1
    
    # Verify no cycles remain
    assert ConstraintSolver._find_cycle(repaired_adj, valid_nodes) is None
```

**Step 2: Run test**
Run: `pytest tests/test_constraint_solver.py -v`
Expected: PASS

**Step 3: Commit**
```bash
git add tests/test_constraint_solver.py
git commit -m "test: add unit test for deterministic acyclicity repair"
```

---

### Task 3: Evaluate the New Solver

**Objective:** Copy the baseline evaluation script and swap in the new `ConstraintSolver`.

**Files:**
- Copy/Modify: `scripts/evaluate_baseline_solver.py` -> `scripts/evaluate_repaired_solver.py`

**Step 1: Create and modify the script**
Copy the baseline script. Change the import:
```python
from src.models.constraint_solver import ConstraintSolver
```
And replace the discretization line:
```python
# Old: discrete = naive_discretize(x, motifs)
# New:
discrete = ConstraintSolver.discretize_and_repair(x, motifs)
```

**Step 2: Run the script**
Run: `PYTHONPATH=. python scripts/evaluate_repaired_solver.py`
Expected: SVR metrics where `acyclic_data_pass` jumps to 100%. Record the output.

**Step 3: Commit**
```bash
git add scripts/evaluate_repaired_solver.py
git commit -m "eval: add script to evaluate deterministically repaired solver"
```

---

### Execution Handoff
Plan complete. Run this to execute the cycle-breaking repair and verify the metric improvement.