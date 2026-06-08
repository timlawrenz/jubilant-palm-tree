# Interpreter Entry-Detection Fix & Ground Truth Re-Audit Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Fix the `run_with_limit` entry-detection heuristic in `GraphInterpreter` which incorrectly requires entry nodes to have zero *outgoing* execution edges, and re-run the ground-truth executability audit to get a true halting rate.

**Architecture:**
1. Update `src/execution_engine/interpreter.py` to define an entry node identically to the 6th Law: `MotifType.BOUNDARY` with zero *incoming* execution edges. Tie-break multiple entries if necessary (though 6-Law perfect graphs will only have one).
2. Run `scripts/audit_dataset_executability.py` to re-audit the dataset.
3. Update `docs/02_EXPERIMENTS_AND_RESULTS.md` with the new, truthful dataset execution metrics.

**Tech Stack:** Python

---

### Task 1: Fix `run_with_limit` Entry Detection

**Objective:** Correct the entry node detection in `GraphInterpreter`.

**Files:**
- Modify: `src/execution_engine/interpreter.py`

**Step 1: Write the fix**
Change the `entry_nodes` list comprehension in `run_with_limit` to only check for NO incoming execution edges.

```python
        # Find entry boundary: BOUNDARY motif with NO incoming EXECUTION edges
        entry_nodes = [
            n_id for n_id, n in self.nodes.items() 
            if n.motif == MotifType.BOUNDARY and not any(n_id == tgt for edges in self.exec_out.values() for tgt in edges.values())
        ]
```

*(Note: We remove `n_id not in self.exec_out` which wrongly required zero outgoing execution edges).*

**Step 2: Commit**
```bash
git add src/execution_engine/interpreter.py
git commit -m "fix: correct entry-detection heuristic in run_with_limit to allow outgoing execution edges"
```

---

### Task 2: Re-Audit Ground Truth Dataset

**Objective:** Re-run the ground truth audit script to measure the real halting rate.

**Files:**
- Action: Execute `scripts/audit_dataset_executability.py`

**Step 1: Execute**
Run: `PYTHONPATH=. python scripts/audit_dataset_executability.py > docs/assets/exp/fix-interpreter-entry-detection/dataset_audit_fixed.txt`
*(Ensure the directory `docs/assets/exp/fix-interpreter-entry-detection` is created first).*

**Step 2: Note the Results**
Examine the output. We expect the halting rate for the 6-Law-perfect graphs (which was previously 0% due to `error_no_entry`) to be extremely high.

**Step 3: Commit**
```bash
git add docs/assets/exp/fix-interpreter-entry-detection/dataset_audit_fixed.txt
git commit -m "eval: log fixed ground truth dataset executability audit"
```

---

### Task 3: Update Experiment Tree and Results Docs

**Objective:** Document the fixed results in the project ledger.

**Files:**
- Modify: `docs/02_EXPERIMENTS_AND_RESULTS.md`
- Modify: `docs/03_EXPERIMENT_TREE.md`

**Step 1: Update `02_EXPERIMENTS_AND_RESULTS.md`**
Under "Semantics & Execution", add a subsection detailing the harness bug discovery and the corrected halting rate of ground-truth data.

**Step 2: Update `03_EXPERIMENT_TREE.md`**
Mark `[NEXT] Exp 1 — Fix Interpreter Entry-Detection & Re-audit Ground Truth` as `[CONCLUDED]` and add a short result summary.

**Step 3: Commit**
```bash
git add docs/02_EXPERIMENTS_AND_RESULTS.md docs/03_EXPERIMENT_TREE.md
git commit -m "docs: update experiment ledger with fixed dataset executability results"
```

---

### Execution Handoff
Plan complete. Run this to fix the interpreter and discover if the ground truth data actually halts!
