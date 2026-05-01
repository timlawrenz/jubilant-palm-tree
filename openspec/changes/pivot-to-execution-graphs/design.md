## Context
The prior phase of the research proved that GNN autoencoders achieved 81% accuracy on AST node type prediction but failed entirely to generate syntactically valid code. The primary culprit is the "literal value bottleneck," where traditional string and variable representations collapse into `UNKNOWN` tokens. The pivot directly replaces strings with raw topological structures designed natively for AI execution.

## Goals / Non-Goals
- Goals: Prove an AI can generate and execute pure topological graphs; build a virtual machine wrapper to map DAGs into Python/Ruby ASTs; utilize a GNN autoencoder for graph generation.
- Non-Goals: Generating human-readable code; building a production-ready compiler from scratch.

## Decisions
- Decision: Motif-Driven Hybrid Architecture.
  - Rationale: A pure structural GNN struggles with granular logic, but can easily predict macro-structures (Motifs) like if/else blocks or loops. The structural model outputs valid scaffolds, and a semantic LLM fills the "content".
- Decision: Diffusion Transformer (DiT) / Flow Matching for Structure.
  - Rationale: Single-pass GNNs fail at creating structure due to the "chicken and egg" problem of node/edge interdependence. Iterative denoising allows the graph to negotiate connections into a valid state.
- Decision: Direct Graph-Walk Interpretation.
  - Rationale: Instead of mapping to a native language AST, the MVP will execute the adjacency matrix directly via a minimal graph-walking script. This proves the graph itself is structurally sound code without relying on Ruby/Python VM baggage.

## Risks / Trade-offs
- Risk: The generated graph is entirely unreadable by humans.
  - Mitigation: The Validation Harness will return exact error nodes and buffer states, keeping debugging confined to machine-interpretable vectors.
- Risk: GNN generation might still create topologically valid but semantically meaningless graphs.
  - Mitigation: Loss functions must be updated to penalize semantic errors from vectorized execution feedback.

## Migration Plan
1. Archive all models and metric generation tools focused on syntax string matching (`validate_alignment_model.py`, `train_hierarchical.py`).
2. Scaffold the new execution engine.
3. Test MVP on simple algorithmic tasks (like array sorting) before any complex modeling.

## Open Questions
- How should the continuous loss function incorporate the vectorized feedback from the execution harness?
- Which target AST (Ruby or Python) offers the most robust manipulation bindings for the interpreter phase?
