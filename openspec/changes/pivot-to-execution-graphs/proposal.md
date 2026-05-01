# Change: Pivot to pure execution graph generation

## Why
Phase 4B of the research demonstrated that generating code via intermediate human-readable text fails due to the "literal value bottleneck." GNN autoencoders achieved 81% node type accuracy but 0% syntax validity. To bypass this translation bottleneck, the project must pivot to generating pure topological execution graphs directly, avoiding syntax strings, named variables, and control flow keywords.

## What Changes
- Replace text-based generation and validation with direct execution graph generation.
- Implement a "Graph-Walk" Interpreter (Execution Engine) that executes the AI-generated adjacency matrix directly by routing an execution pointer through the Motifs, bypassing native language ASTs entirely.
- Develop a GNN Autoencoder designed to output topological AST node structures explicitly (no strings or named variables, only memory tensor coordinates).
- Create a Validation Harness that strictly evaluates graphs for topological correctness (no unpermitted circular dependencies, arity checks) and runs them through vectorized unit tests without generating stack traces (producing only error node indices and memory buffer states).
- **BREAKING**: Abandons string-based metric validation (BLEU, string syntax checks) and traditional model generation approaches.

## Impact
- Affected specs: `execution-graph-generation` (New Capability)
- Affected code: New implementation scripts replacing `train_hierarchical.py`, `train_autoencoder.py`, and `validate_generation_*.py`.
