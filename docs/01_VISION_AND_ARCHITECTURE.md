# The Neural Universal Machine: AI-Native Execution Graphs

*A research initiative pivoting from human-readable code generation to pure, topological execution matrices.*

---

## The History: The Literal Value Bottleneck

This project initially explored whether Graph Neural Networks (GNNs) could learn to generate valid code by treating Abstract Syntax Trees (ASTs) as graphs. We trained GNN autoencoders on 22,452 Ruby ASTs. 

The models achieved **81% node type accuracy** and **99.5% type diversity**, yet generated exactly **0% syntactically valid code**. 

The root cause was identified as the **Literal Value Bottleneck**. Nearly 47% of all AST nodes were literal values (method names, variable names, strings, numbers). Because these were collapsed into undifferentiated `UNKNOWN` tokens during structural encoding, the model could perfectly reconstruct the skeleton of a program while entirely losing the semantic "content." Furthermore, forcing a one-shot GNN to guess both node types and edge connections simultaneously led to severe mode collapse (the "chicken and egg" problem of graph generation).

For a detailed breakdown of these Phase 4B experiments and metrics, see: [**The Literal Value Bottleneck (Blog Post)**](blog_post.md).

Previous documentation and research state can be found in `README_archive_phase4b.md`.

---

## The Pivot: Eradicating Human Syntax

The current paradigm of using AI to generate code acts as a translation bottleneck: a high-dimensional neural network collapses its probabilistic understanding into a linear, human-readable string of characters, which a compiler then immediately parses *back* into a multi-dimensional graph to execute.

If we remove the human from the loop, a "programming language" designed natively for AI should not be a language at all. It is a mathematical specification for a Directed Acyclic Graph (DAG). 

We are pivoting the research to build a **Neural Universal Machine**. 

### Core Principles
1. **The Death of Variable Names:** Variable names are human mnemonics. The AI-native language relies entirely on directed edges. Data dependencies are pure topological routing.
2. **Eradication of Syntax Sugar:** No parentheses, no brackets, no formatting. The code is saved directly as a sparse adjacency matrix and a minimal feature matrix.
3. **Execution by Graph-Walk:** The matrix is not compiled or parsed; it is traversed directly by a minimal graph-walking interpreter.
4. **Guaranteed Syntax:** Because the model generates mathematical graph topology directly rather than stringing text together, "syntax errors" are mathematically impossible. 

---

## The New Architecture: Scaffold and Fill

To solve the literal value bottleneck and the mode collapse of pure GNN generation, we are adopting a **Motif-Driven Hybrid Architecture**—a system divided into two strict "branches of government":

### 1. The Executive Branch (Structural Matrix Generation)
A **Diffusion Transformer (DiT) / Flow Matching model** is responsible solely for predicting the graph topology. 
*   **Iterative Denoising:** By using diffusion, we give the nodes the computational "time" to negotiate their connections and resolve into mathematically legal states.
*   **Macro-Motifs:** Instead of predicting granular syntax (like `def` or `args`), the DiT predicts Turing-complete structural Motifs (based on the Böhm-Jacopini theorem): `[Sequence]`, `[Condition]`, `[Loop]`, `[State]`, `[Message]`, and `[Boundary]`.
*   The DiT knows nothing about human language. It purely outputs a mathematically valid logic scaffold (an adjacency matrix).

### 2. The Legislative Branch (The Semantic Custodian)
A **Semantic LLM** acts as the data custodian. 
*   It looks at the generated motif scaffold and the human intent, and maps human literals (strings, external function names) into a Constant Pool. 
*   The structural matrix interfaces with these literals purely through integer pointers. The LLM acts as the compiler/renderer for the "content", leaving the structural physics entirely to the DiT.

### 3. The Virtual Machine (Graph-Walk Interpreter)
To execute the generated matrix, we bypass native language ASTs completely. The **Graph-Walk Interpreter** is a minimal execution engine that drops a token onto the `[Boundary]` node and routes it through the matrix's directed edges. 
*   It evaluates `[Condition]` nodes and routes the execution pointer accordingly.
*   It updates a state dictionary when traversing `[State]` nodes.
*   It fires external logic when hitting a `[Message]` node.

---

## Technical Progress & Demonstrations

### MVP Execution Engine & Fibonacci Demo
To validate the Graph-Walk concept, we implemented the schema (`src/execution_engine/schema.py`) and the interpreter (`src/execution_engine/interpreter.py`). We successfully hardcoded a 25-node topological matrix representing a `while` loop that calculates the 10th Fibonacci number. The interpreter traversed the purely structural graph—evaluating the condition, updating memory states via data-dependency edges, and returning the correct output (`55`)—all without relying on a traditional compiler.

### Dataset Compression
We developed the parser (`scripts/dataset_prep/compress_ast.py`) to systematically distill the 22,452 Ruby ASTs into our Universal Motifs.
When run against complex, real-world examples (like the 144-node `structure` method from the AWS Ruby SDK), the parser:
1. Stripped away all 74 dimensions of Ruby syntax, mapping everything to the 6 Motifs.
2. Extracted exactly **50 literal values** (strings and method names like `"empty?"` and `"underscore"`) completely out of the graph and into the LLM's `literal_pool`.
3. Re-mapped the tree into 107 `DATA` routing edges and 36 `EXECUTION` path edges.

This proves we can mathematically separate logic from semantics on a massive scale, producing perfectly dense matrices for the Diffusion Transformer.

**[Hugging Face Hub Repository: timlawrenz/jubilant-palm-tree](https://huggingface.co/timlawrenz/jubilant-palm-tree)**

### Generative Model: Permuted Dense DiT & Flow Matching
We engineered a specialized Diffusion Transformer (DiT) architecture capable of native graph generation without spatial biases:
1. **Node Permutation & Cross-Hatch Injection**: The DataLoader randomly shuffles node ordering to mathematically destroy spatial bias. The 1D sequence of Motifs is embedded and "Cross-Hatched" (broadcasted across rows and columns) into the 2D noise matrix, giving the DiT 360-degree awareness of the structural nodes it is connecting.
2. **Axial Attention**: Instead of ViT square patches, the DiT uses Message-Passing Axial Attention. It evaluates outgoing edges via Row-Attention and incoming edge constraints via Column-Attention.
3. **Hybrid Flow Matching & Classification**: The DiT predicts continuous velocity vectors (via masked MSE) for topological routing, alongside categorical logits (via masked Cross-Entropy) to distinctly assign argument indices without continuous rounding collisions.
4. **Deterministic Inference**: Sampling is performed using a 20-step Euler ODE solver, followed by Sigmoid thresholding to snap the continuous field into topological 1s and 0s, and an argmax over the logit channels to distinctively assign data argument routing.

*(Note: The hyperparameters for this architecture—Effective Batch Size=16, LR=1e-4, Depth=12—were mathematically locked in via an extensive grid search. See the [Hyperparameter Ablation Study](docs/HYPERPARAMETER_ABLATION.md) for data and methodology).*

### The Validation Harness: 5 Laws of Physics
To deterministically grade the DiT's output (Syntactic Validity Rate), we implemented a static topological analyzer that enforces 5 absolute graph laws:
1. **Execution Out-Degree** (Valid branching limits per Motif)
2. **Data In-Degree** (Strict argument arity constraints)
3. **No Orphans** (BFS reachability)
4. **Acyclic Data Plane** (DFS paradox/cycle detection)
5. **Terminal Sink** (Escape hatch routing for infinite loops)

*For a detailed breakdown of the math and implementation, see [docs/neural-universal-machine-architecture.md](docs/neural-universal-machine-architecture.md).*

---

## Roadmap

This pivot redefines the immediate technical milestones of the project:

1. **Dataset Compression (Completed):** Write a parser to run over the 22,452 Ruby AST graphs, strip away the 74-dimensional Ruby syntax, and collapse the human logic into our 6 language-agnostic Universal Motifs. This distills the dataset into pure human problem-solving topology.
2. **Execution Engine MVP (Completed):** Write the Graph-Walk Interpreter and hardcode a simple mathematical graph (like a Fibonacci sequence) to prove execution natively inside the matrix structure.
3. **DiT Continuous Pre-Training (Completed):** Train the Diffusion Transformer on the compressed dataset to iteratively denoise empty matrices into valid Motif structures. The model successfully scaled to 128-node graphs using Optimal Transport Flow Matching. *(Note: The originally reported "100% SVR" was measured with a buggy validator — see PROJECT_LOG.md "Milestone 6: Validator Bug Discovery" for the corrected results.)*
4. **Differentiable Structural Loss Fine-Tuning (In Progress):** Fine-tune the DiT using differentiable versions of the `GraphValidator` Laws of Physics as direct loss terms (NOTEARS acyclic penalty, degree losses, density/orphan constraints), combined with reconstruction MSE and a KL divergence anchor to the pre-trained reference. *(Note: The originally planned PPO/REINFORCE approach was abandoned — see PROJECT_LOG.md for details.)*
5. **Semantic Integration:** Connect the LLM rendering bridge to populate the DiT scaffolds with executable literals. 

*The detailed technical specification for this phase is tracked via OpenSpec in `openspec/changes/pivot-to-execution-graphs/`.*# Neural Universal Machine: Architecture & Implementation

This document serves as the detailed technical specification and historical record of the architectural decisions made during the pivot to pure topological execution graphs.

## 1. The Core Challenge: Topology vs. Spatial Bias

When transitioning from text-based LLMs to a structural Diffusion Transformer (DiT) for graph generation, we encountered the **Permutation Invariance vs. Spatial Bias** trap.

Standard image-based DiTs (like Stable Diffusion 3 or Sora) use Vision Transformer (ViT) blocks. ViTs chop matrices into square patches and use 2D positional encodings. If applied to an adjacency matrix, the network learns that Node 4 connects to Node 5 because they are "next to each other" spatially. However, in a graph, node ordering is entirely arbitrary.

To solve this, we implemented **The Permuted Dense DiT**:
1. **No 2D Positional Encodings**: Coordinates get a symmetric `Node Embedding(i) + Node Embedding(j)` without spatial `(x, y)` math.
2. **Node Permutation Data Augmentation**: In the PyTorch `DataLoader`, every time a graph is fetched, its node order is randomly shuffled. The topological routing remains identical, but the visual matrix changes completely. This mathematically destroys spatial bias and forces the DiT to learn pure topological rules.

## 2. The 1D vs 2D Disconnect: "Cross-Hatch" Embedding Injection

The DiT operates on a 2D adjacency matrix, but the ingredients (the Motifs) are a 1D list (e.g., `[Boundary, Loop, State...]`). If the DiT only sees noise in the 2D grid, it doesn't know what types of nodes it is connecting.

We built the **InputConditioner (Cross-Hatch Injection)**:
1. Embed the 1D Motif tensor (values 0-6) into `[BATCH, 128, 16]`.
2. **Broadcast Rows**: Expand to `[BATCH, 16, 128, 128]` to represent Source Nodes.
3. **Broadcast Columns**: Transpose and expand to represent Target Nodes.
4. **Concatenate**: Fuse the 3-channel noisy adjacency grid with the 16-channel Source grid and 16-channel Target grid.

Every pixel `(i, j)` now has 35 channels, giving the DiT complete 360-degree awareness of the structural identities it is attempting to connect.

## 3. The Model: Message-Passing Axial Attention

Because we abandoned ViT square patches, we must process the matrix using **Axial (Row-Column) Attention**. This mimics Graph Message Passing but with the holistic visibility of a Transformer.

Inside the `GraphDiTBlock`:
1. **Row Attention (The "Outgoing" Perspective)**: Look across the rows `[B * H, W, C]`. The transformer evaluates all 128 potential outgoing connections simultaneously (e.g., "I am a Condition Motif. I must point to exactly two targets").
2. **Column Attention (The "Incoming" Perspective)**: Look down the columns `[B * W, H, C]`. The transformer evaluates incoming data dependencies (e.g., "I am a State Motif. I can only accept one incoming data edge").

## 4. The Training Objective: Hybrid Flow Matching & Classification

Instead of DDPM, we use **Conditional Flow Matching (Optimal Transport)** for the continuous graph physics, combined with **Categorical Classification** for discrete argument ordering.

The model outputs 6 channels per edge coordinate:
*   **Channels 0 & 1** (Presence, EdgeType): Trained via Optimal Transport to predict the continuous vector field velocity from noise to structure.
*   **Channels 2-5** (Logit0 - Logit3): Trained via Cross-Entropy to classify the specific `input_index` of the edge (resolving the "Continuous Index Channel" collision issue).

For the Continuous Channels (0 & 1):
*   **The Target**: $x_1$ (Clean adjacency)
*   **The Noise**: $x_0 \sim \mathcal{N}(0, I)$
*   **The Target Velocity**: $v_t = x_1 - x_0$

**Masked Loss**: Because our graphs vary in size (padded to 128x128), calculating standard MSE/CE would penalize the model for failing to perfectly denoise the empty void. We use a boolean `padding_mask` to zero out the error in the padded regions, averaging the loss ONLY over the valid node intersections.

## 5. Inference: Euler ODE Solver & The Judicial Constraint Solver

Inference is fully deterministic. We start at pure noise and follow the predicted vector field to $t=1$.
```python
for step in range(num_steps):
    t_val = step * dt
    velocity = model(x, t, motifs)
    x = x + (velocity * dt) # Euler Step
```

**Discretization**: The output of the DiT is a continuous probabilistic heat map. Naive global thresholding (e.g. `> 0.5`) causes catastrophic index collisions. To bridge the continuous-to-discrete gap, we pass the continuous matrix through the **Judicial Constraint Solver**.

Rather than relying on the DiT to perfectly zero out its own noise, the solver reads the probability heatmap and mathematically "snaps" the edges into legal bounds based on the Motif laws:
1. **Arity Snapping**: A `[Condition]` node must have exactly 2 outgoing edges. The solver isolates the row, selects the `Top-2` highest probability values, snaps them to `1`, and forces the rest to `0`.
2. **Index Conflict Resolution**: If two arguments on a `[Message]` node both probabilistically claim `input_index = 0`, the solver uses the categorical logits to distinctively enforce mutually exclusive routing (e.g. the edge with the highest magnitude logit wins the slot).

## 6. The Validation Harness: The 5 Laws of Physics

To measure true Syntactic Validity Rate (SVR), we built a deterministic PyTorch grader that runs 5 strict topological checks against the thresholded matrix:

1.  **Execution Out-Degree Laws**: E.g., `[Condition]` must have 0, 1, or 2 outgoing execution edges (True/False). *(Originally required exactly 0 or 2; relaxed to allow 1 for compressed motifs where one branch is outside the compression window.)*
2.  **Data In-Degree (Arity) Laws**: E.g., `[Condition]` and `[Loop]` must have at least 1 incoming data edge. `[State]` writes must have at least 1 incoming data edge. *(Originally required exactly 1; relaxed because real conditions evaluate multi-operand expressions.)*
3.  **No Orphans (Reachability)**: A Breadth-First Search confirms no disconnected logic islands.
4.  **Acyclic Data Plane**: A Depth-First Search confirms the data dependencies contain zero paradoxes/cycles.
5.  **Terminal Sink**: A reverse-BFS ensures no infinite loops exist without an escape path to a `[Boundary]`.

## 7. RLAIF: Differentiable Structural Loss Fine-Tuning

Flow Matching (MSE) is blind to graph topology. To push the 128-node graph Syntactic Validity Rate toward 100%, the model transitions from continuous generative pre-training to **Reinforcement Learning from AI Feedback (RLAIF)**.

> **Note (May 2026):** The originally planned PPO approach was abandoned because computing exact log-likelihoods of 20-step ODE trajectories is computationally intractable. Instead, we use **differentiable structural losses** computed directly on the continuous model output, combined with reconstruction MSE and a KL anchor to the pre-trained reference. See PROJECT_LOG.md for the full evolution.

The current approach uses three loss components:
*   **Reconstruction Loss (β=1.0):** MSE to the ground truth graph — anchors edge density and shape to the training distribution.
*   **Structural Loss (β=0.01):** Differentiable versions of the 5 Laws of Physics, including a NOTEARS acyclic penalty (tr(exp(A))-n), out-degree/in-degree losses, orphan detection, and edge density.
*   **KL Divergence Anchor (β=1.0):** MSE between active policy and frozen reference velocities — prevents catastrophic forgetting.

> **Errata (May 9, 2026):** Three bugs were discovered in the `GraphValidator` that invalidated all prior SVR measurements. See "Milestone 6: Validator Bug Discovery" in PROJECT_LOG.md. The `_check_acyclic_data` function was inverted (returned True for cyclic graphs), `_check_in_degree` was too strict (required exactly 1 instead of ≥1), and `_check_out_degree` disallowed valid CONDITION configurations. The training dataset validates at 83.2% SVR with the corrected validator.

## 8. The Workflow: Three Branches of Government

To construct perfectly legal code without an autoregressive token bottleneck, we divide labor strictly along model strengths:
*   **The Legislative Branch (Semantic LLM)**: Translates human intent into the "Bill of Materials" (the 1D array of Motifs) and populates the Literal Pool.
*   **The Executive Branch (Structural DiT)**: Takes the unrouted ingredients and predicts a continuous probability heat map for routing the `EXECUTION` and `DATA` edges.
*   **The Judicial Branch (Constraint Solver)**: Maps the continuous heat map into a mathematically perfect, executable discrete DAG by enforcing strict arity and collision laws.

## 9. Dataset / Structural Compression

**`scripts/dataset_prep/compress_ast.py`** extracts human syntax out of AST representations.
- Reduces massive node vocabularies (e.g., 73 types in Ruby) down to the 6 Motif primitives.
- Replaces concrete syntax with topological Data routing (e.g., assigning a variable points its data flow edge).
- Collects and manages strings/numbers as integers mapped into a `literal_pool`.
