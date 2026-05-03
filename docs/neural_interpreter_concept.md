# Neural Interpreter: Probabilistic Execution & Wave Collapse

## 1. Overview: The Leap to Differentiable Programming
In standard classical execution (like a CPU running a compiled binary), code is evaluated as a particle: it exists in exactly one state at a time. A `[Condition]` evaluates to strictly `True` or `False`, and variables hold discrete, absolute values in memory. 

This document proposes a **Probabilistic Execution Engine** (The "Neural Interpreter") for the Neural Universal Machine. By keeping the execution state as a continuous probability wave right up until it interacts with the physical world, we can create a fully differentiable execution graph.

---

## 2. The Core Mechanics

### 2.1 The Memory Buffer as a Belief State
In a classical interpreter, memory allocation assigns a discrete value (e.g., `x = 5`). In the probabilistic interpreter, the memory buffer holds **probability distributions**. 
* `x` is not `5`; it is a Gaussian distribution centered at 5.
* A boolean flag is a categorical distribution (e.g., 70% `True`, 30% `False`).

### 2.2 Math as Bayesian Inference
When the execution wave hits a `[State]` motif that performs a mathematical operation (e.g., `z = x + y`), it does not add two absolute scalars. Instead, it convolves two probability distributions to create a third, wider distribution. The mathematical operations remain entirely continuous.

### 2.3 Control Flow Superposition & Deferred Collapse
When the execution wave hits a `[Condition]` motif (e.g., `if z > 10`), it does not commit to a single path. It splits.
* The execution state enters a **superposition**, spawning parallel universe states in the VRAM.
* **Mitigating Path Explosion:** To prevent VRAM exhaustion (the $2^N$ branching factor problem), the interpreter employs **Beam Search** or **Monte Carlo Sampling**. The engine aggressively prunes any execution branch whose probability drops below a threshold (e.g., `< 0.01%`), allowing the wave to flow only down mathematically plausible paths.

### 2.4 The `[Message]` Motif is the Observer
In quantum mechanics, a wave function collapses the moment it is measured. In the Neural Universal Machine, the `[Message]` motif (which triggers an API call, database write, or external side effect) acts as the **Observer**.
* You cannot send an API request that is "60% likely to happen."
* The moment the execution wave reaches a `[Message]` node, the engine realizes it can no longer defer evaluation. It evaluates the current probability distribution for that branch, applies an `argmax` (or samples), and **snaps** the state into a discrete reality to construct the concrete JSON payload.

---

## 3. The Ultimate Benefit: Reversible Computing

Because the execution of the code is a continuous probability wave, the entire execution is **differentiable**. 

This unlocks **Reversible Code**:
* You can define a target output state (e.g., "The `[Message]` node must output a payload that results in a `200 OK` status").
* You can literally **backpropagate through the execution of the code** to determine the exact initial inputs required to achieve that output.
* The system transitions from merely generating code to perfectly understanding the mathematical boundaries and input/output gradients of the code it just wrote.

## 4. Integration with the Three Branches
* **Phase 1 (Generation):** The Semantic LLM (Legislative) drafts the Motif ingredients, the Executive DiT routes the probabilistic continuous structure, and the Judicial Solver snaps it into a legal discrete topological DAG.
* **Phase 2 (Execution):** The Neural Interpreter evaluates this valid DAG, deferring mathematical and structural collapse until external side effects require observation.
