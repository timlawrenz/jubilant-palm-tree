# Executability Audit Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Conduct an executability audit of generated, structurally-valid program graphs. Since we lack the Semantic LLM to provide meaningful literals, we will build a `DummySemanticCustodian` that assigns safe, non-crashing operations to the topology. We will then execute the graphs in the `GraphInterpreter` to measure the Halting Rate and instruction density, proving the generated graphs compute (something) without blowing up.

**Architecture:**
1. Create `src/execution_engine/dummy_custodian.py` to translate `(discrete_adj, motifs)` tensors into `ExecutionGraph` Pydantic models. It assigns dummy variables to `STATE` nodes, safe math operations to `MESSAGE` nodes, and constant booleans/integers to inputs so the interpreter doesn't throw `TypeError`s.
2. Build a `SafeInterpreter` wrapper around `GraphInterpreter` that enforces an instruction limit (e.g., 1000 steps) to trap infinite loops, and overrides `_execute_message` to handle the dummy operations.
3. Write `scripts/evaluate_executability.py` to generate batches of graphs, repair them via `ConstraintSolver`, filter for the 100% SVR "perfect" graphs, and execute them.
4. Record the Halting Rate, Infinite Loop Rate, and Crash Rate.

**Tech Stack:** Python, PyTorch, Pydantic

---

### Task 1: Build the Dummy Semantic Custodian

**Objective:** Translate continuous/discrete tensor output into executable Pydantic schemas.

**Files:**
- Create: `src/execution_engine/dummy_custodian.py`

**Step 1: Write the custodian**

```python
import torch
from typing import List
from src.execution_engine.schema import ExecutionGraph, Node, Edge, MotifType, EdgeType

class DummySemanticCustodian:
    """
    Acts as a stand-in for the Semantic LLM.
    Translates raw tensor topologies into ExecutionGraphs with safe, dummy literals.
    """
    
    MOTIF_MAP = {
        1: MotifType.BOUNDARY,
        2: MotifType.SEQUENCE,
        3: MotifType.CONDITION,
        4: MotifType.LOOP,
        5: MotifType.STATE,
        6: MotifType.MESSAGE
    }
    
    @classmethod
    def tensor_to_graph(cls, adj: torch.Tensor, motifs: torch.Tensor) -> ExecutionGraph:
        """
        adj: [3, N, N] discrete adjacency
        motifs: [N] 1D tensor
        """
        valid_nodes = [n for n, m in enumerate(motifs.tolist()) if m != 0]
        
        presence = adj[0].bool().tolist()
        edge_type = adj[1].int().tolist()
        input_idx = adj[2].int().tolist()
        
        nodes: List[Node] = []
        edges: List[Edge] = []
        literal_pool = {
            0: "dummy_var",
            1: "safe_op",
            2: 1 # Safe constant
        }
        
        for n in valid_nodes:
            m_id = motifs[n].item()
            m_type = cls.MOTIF_MAP[m_id]
            
            ptr = None
            if m_type == MotifType.STATE:
                ptr = 0 # Points to "dummy_var"
            elif m_type == MotifType.MESSAGE:
                # Check if it has incoming data. If not, it's a constant.
                has_in = any(presence[src][n] and edge_type[src][n] == 1 for src in valid_nodes)
                ptr = 1 if has_in else 2 # "safe_op" or constant 1
                
            nodes.append(Node(node_id=n, motif=m_type, literal_pointer=ptr))
            
            # Extract edges originating from n
            for tgt in valid_nodes:
                if presence[n][tgt]:
                    edges.append(Edge(
                        source_node=n,
                        target_node=tgt,
                        edge_type=EdgeType(edge_type[n][tgt]),
                        input_index=input_idx[n][tgt]
                    ))
                    
        return ExecutionGraph(nodes=nodes, edges=edges, literal_pool=literal_pool)
```

**Step 2: Commit**
```bash
git add src/execution_engine/dummy_custodian.py
git commit -m "feat: add dummy semantic custodian to translate tensors to execution schemas"
```

---

### Task 2: Build the Safe Interpreter

**Objective:** Extend the interpreter to support instruction limits and safe dummy execution.

**Files:**
- Modify: `src/execution_engine/interpreter.py`

**Step 1: Add execution tracking and safe execution**
*(Agent: Open `src/execution_engine/interpreter.py`. Add a `run_with_limit(self, max_steps=1000)` method that tracks the execution pointer. Add support for `safe_op` in `_execute_message` that simply returns `1` regardless of inputs, and `_evaluate_condition` that always returns `False` or alternates so infinite loops have a chance to terminate if structurally sound.)*

```python
    # Inside GraphInterpreter in src/execution_engine/interpreter.py
    def _execute_message(self, literal_pointer: int, args: list):
        op = self.graph.literal_pool[literal_pointer]
        if op == "safe_op":
            return 1
        # ... existing logic ...

    def run_with_limit(self, max_steps: int = 1000):
        # Find entry boundary
        entry_nodes = [n_id for n_id, n in self.nodes.items() if n.motif == MotifType.BOUNDARY and n_id not in self.data_in and not any(tgt == n_id for edges in self.exec_out.values() for tgt in edges.values())]
        
        if not entry_nodes:
            raise RuntimeError("No entry node found")
            
        current = entry_nodes[0]
        steps = 0
        
        while current is not None:
            if steps >= max_steps:
                return {"status": "infinite_loop", "steps": steps}
                
            steps += 1
            node = self.nodes[current]
            
            if node.motif == MotifType.BOUNDARY and steps > 1:
                return {"status": "halted", "steps": steps}
                
            # Simulate execution routing...
            out_edges = self.exec_out.get(current, {})
            if not out_edges:
                return {"status": "halted", "steps": steps}
                
            if node.motif in (MotifType.CONDITION, MotifType.LOOP):
                # Dummy evaluation: always pick 0 to avoid getting stuck in loops forever, or alternate based on steps
                idx = steps % 2 if len(out_edges) > 1 else list(out_edges.keys())[0]
                current = out_edges.get(idx)
            else:
                current = out_edges.get(0, list(out_edges.values())[0] if out_edges else None)

        return {"status": "halted", "steps": steps}
```

**Step 2: Commit**
```bash
git add src/execution_engine/interpreter.py
git commit -m "feat: add instruction limits and safe dummy operations to interpreter"
```

---

### Task 3: Evaluate Executability

**Objective:** Wire the generator, constraint solver, dummy custodian, and safe interpreter together to evaluate the 100% SVR subset.

**Files:**
- Create: `scripts/evaluate_executability.py`

**Step 1: Write the evaluation script**
*(Agent: Write a script similar to `evaluate_repaired_solver.py`. Generate 500 graphs using the Epoch 340 checkpoint and `ConstraintSolver.discretize_and_repair`. Pass them all to `GraphValidator`. Take ONLY the perfect graphs. Convert them via `DummySemanticCustodian.tensor_to_graph`. Run `GraphInterpreter(graph).run_with_limit()`. Print the percentage that successfully halted vs infinite looped vs crashed.)*

**Step 2: Commit**
```bash
git add scripts/evaluate_executability.py
git commit -m "eval: add executability audit script for generated graphs"
```

---

### Execution Handoff
Plan complete. Execute to find out if the generated valid syntax actually computes anything coherent!