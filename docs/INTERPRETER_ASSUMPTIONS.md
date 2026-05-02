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

## 5. Discoveries from Traditional Compiler Test Suites

A review of classical interpreter/compiler architectures (e.g., *Crafting Interpreters* (Lox), LLVM IR, Make-A-Lisp) reveals several "everyday assumptions" about runtime execution that traditional syntax trees handle implicitly, but a pure mathematical graph leaves ambiguous.

### Evaluation Order & Short-Circuiting
- [ ] **Short-Circuit Evaluation:** In the statement `A and B`, if `A` is false, `B` should *never* execute. 
  * *Uncovered:* In our graph, the DiT wires both `A` and `B` as data inputs to an `[And]` Message node. Our `_resolve_data` method eagerly resolves all incoming data edges before applying the logic. If `B` has a side-effect or a crash (e.g., `x != 0 and 10/x > 1`), our VM will fatally crash, violating short-circuit assumptions.
- [ ] **Strict Argument Evaluation Order (Left-to-Right):** In `func(a(), b())`, developers assume `a()` is evaluated completely before `b()` begins. 
  * *Uncovered:* If both arguments mutate state, evaluating them out of order changes the program's output. While our `sorted(keys)` ensures deterministic traversal, our graph does not explicitly force temporal sequencing of data nodes without placing them on the `EXECUTION` path.

### Memory & Resource Management
- [ ] **Garbage Collection (GC) / Deallocation:** When a variable leaves scope, its memory is freed.
  * *Uncovered:* Our MVP `self.memory` dictionary grows indefinitely. We have no mechanism to detect when a memory tensor is no longer referenced by any downstream `[State]` read node.
- [ ] **Call Stack Depth Limits:** Recursive subroutine calls will eventually consume all memory if unconstrained.
  * *Uncovered:* If a `[Message]` routes back to the entry `[Boundary]` recursively, Python will hit a `RecursionError` and crash the host process rather than throwing a handled VM `StackOverflowError`.
- [ ] **Shadowing & Closures:** Inner scopes can create variables with the same name as outer scopes without overwriting them, and functions "remember" the state of their creation environment.
  * *Uncovered:* Because our `literal_pool` maps directly to a flat `self.memory` dictionary, any variable assignment overwrites globally. We lack the concept of Environment Frames (linked lists of scope dictionaries).

### Safety & Undefined Behavior
- [ ] **Type Promotion vs. Strict Coercion:** Some languages silently convert `5 + "5"` to `"55"`, others throw errors.
  * *Uncovered:* Our Graph-Walker currently inherits Python's type system by proxy. An AI-native graph needs an explicit internal contract for how multi-dimensional tensors/embeddings coerce types, or we risk Undefined Behavior (UB).

---

### Conclusion for Next Steps

The MVP interpreter successfully models pure deterministic logic flows (the "happy path"). 
However, to make it adversarial-proof against generative noise and feature-complete compared to traditional VMs, the most critical uncovered edge cases to patch are:
1. **Entry Point Ambiguity** (Differentiating Start vs. End Boundaries).
2. **Uninitialized Memory Guards** (Throwing strict VM errors instead of silent `None` propagation).
3. **Data Node Memoization** (Caching `_resolve_data` per execution step to prevent duplicate side-effects).
4. **Environment Frames** (Replacing the flat dictionary with a scoped Stack to prevent variable collisions).
5. **Lazy Evaluation / Short-Circuiting Guards** (Preventing fatal crashes on eagerly evaluated `DATA` edges).