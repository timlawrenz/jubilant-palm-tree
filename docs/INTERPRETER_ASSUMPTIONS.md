# Neural Universal Machine: Interpreter Assumptions & Edge Cases

The `GraphInterpreter` MVP successfully proves that we can execute topological logic matrices without traditional string-parsing. However, any production Virtual Machine relies on a massive foundation of implicit software engineering assumptions—rules about how memory, time, and state behave.

This document serves as an audit of the basic assumptions we make day-to-day when writing and executing code, evaluating which are explicitly covered by our current execution engine and which represent dangerous edge cases (uncovered).

---

## 1. Entry, Exit, and Control Flow

When we run a script, we assume a deterministic start, a deterministic end, and a predictable flow of time.

- [x] **Single Entry Point:** A program starts at exactly one defined location. 
  * *Covered (Partially):* We query for `MotifType.BOUNDARY` and pick the first one. 
  * *Uncovered Edge Case:* The DiT might generate multiple `[Boundary]` nodes (e.g., one for Entry, one for Exit). The current interpreter arbitrarily picks `start_nodes[0]`, which might accidentally start execution at the end of the program.
- [x] **Finite Execution Time:** Programs should not hang indefinitely.
  * *Covered:* We implemented the `max_steps` runtime guard to catch hanging `[Loop]` cycles.
- [x] **Exhaustive Branching:** Every path in a conditional statement should lead to valid execution.
  * *Covered:* The interpreter evaluates the boolean and selects `0` or `1`. It gracefully breaks if an expected path doesn't exist.
- [ ] **Fallthrough / Merging Execution Paths:** After an `if/else` block, execution naturally rejoins the main trunk.
  * *Uncovered:* Our MVP handles branching, but doesn't explicitly enforce how two execution branches safely merge back into a single `[Sequence]` node without duplicating execution pointers.

## 2. Data Flow & Side Effects

Data dependencies form the arguments and mathematical operations of the code.

- [x] **Acyclic Data Resolution:** Data cannot depend on itself (no paradoxes).
  * *Covered:* Handled via the `resolving_set` recursion guard in `_resolve_data`.
- [x] **Deterministic Argument Ordering:** `func(A, B)` is distinct from `func(B, A)`.
  * *Covered:* We enforce deterministic ordering via `sorted(self.data_in[node_id].keys())` (e.g., resolving `input_index` 0 before 1).
- [ ] **Data Node Memoization (Single Evaluation):** If a mathematical operation's output is used by three different downstream nodes, the math should only be computed once.
  * *Uncovered:* Currently, `_resolve_data` recursively re-evaluates the entire tree every time. If a `[Message]` node takes a heavy calculation, it will run three times.
- [ ] **Purity of Data vs. Execution:** Operations with "side-effects" (like `print` or mutating a database) should only trigger when the Execution pointer hits them.
  * *Uncovered:* If the DiT accidentally attaches a `print` `[Message]` node to a `DATA` edge instead of an `EXECUTION` edge, `_resolve_data` will trigger the print purely as a side-effect of looking up a variable. 
- [ ] **Dangling Data Operations:** Code that is written but never assigned or executed does nothing.
  * *Uncovered:* The DiT might generate a beautiful mathematical subgraph that has no outgoing data edges to a `[State]` write or `[Message]` execution. It just floats in the matrix, wasting generation capacity but never breaking the VM.

## 3. Memory, State, and Scoping

How variables and states are stored, isolated, and mutated.

- [ ] **Initialization Before Read:** A variable must exist before it is used.
  * *Uncovered:* Currently, if the graph reads a `[State]` node that hasn't been written to, `self.memory.get()` silently returns `None`. This will cause confusing downstream native Python crashes (e.g., trying to do `None + 5` in a Message node) rather than a clear VM `ReferenceError`.
- [ ] **Scope Isolation (Local vs. Global):** Functions and loops have their own isolated memory contexts to prevent overwriting global variables.
  * *Uncovered:* The MVP uses a single flat `self.memory` dictionary. Everything is a global variable. If we expand to complex subroutines, variables will collide.
- [ ] **Data Types & Coercion:** A string cannot be subtracted from an integer.
  * *Uncovered:* The interpreter relies entirely on Python's native duck-typing. If the DiT passes a string literal to the `<` (less than) operator against an integer, the underlying Python VM throws a fatal `TypeError` not caught by our interpreter sandbox.

## 4. Modularity and Extensibility

Programs rely on reusable blocks of logic.

- [ ] **Subroutine Calls (Call Stack):** Programs can jump to a different graph, execute it, and return to the exact previous location.
  * *Uncovered:* Currently, `[Message]` is strictly hardcoded to a Python standard library (`+`, `-`, `<`, `print`). The interpreter has no concept of a call stack to "pause" the current graph, jump to another DiT-generated `[Boundary]`, and return.
- [ ] **Error Handling (Try/Catch):** Programs can detect runtime errors and gracefully route to a fallback execution path.
  * *Uncovered:* There is no `[Rescue]` or `[Catch]` motif. Any mathematical error (like divide-by-zero in a `[Message]`) instantly fatally crashes the interpreter.

---

### Conclusion for Next Steps

The MVP interpreter successfully models pure deterministic logic flows (the "happy path"). 
However, to make it adversarial-proof against generative noise, the most critical uncovered edge cases to patch are:
1. **Entry Point Ambiguity** (Differentiating Start vs. End Boundaries).
2. **Uninitialized Memory Guards** (Throwing strict VM errors instead of silent `None` propagation).
3. **Data Node Memoization** (Caching `_resolve_data` per execution step to prevent duplicate side-effects).