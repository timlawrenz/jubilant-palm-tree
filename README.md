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

1. **Dataset Compression:** Write a parser to run over the 22,452 Ruby AST graphs, strip away the 74-dimensional Ruby syntax, and collapse the human logic into our 6 language-agnostic Universal Motifs. This distills the dataset into pure human problem-solving topology.
2. **Execution Engine MVP:** Write the Graph-Walk Interpreter and hardcode a simple mathematical graph (like a Fibonacci sequence) to prove execution natively inside the matrix structure.
3. **DiT Generation Model:** Train the Diffusion Transformer on the compressed dataset to iteratively denoise empty matrices into valid Motif structures.
4. **Semantic Integration:** Connect the LLM rendering bridge to populate the DiT scaffolds with executable literals. 

*The detailed technical specification for this phase is tracked via OpenSpec in `openspec/changes/pivot-to-execution-graphs/`.*