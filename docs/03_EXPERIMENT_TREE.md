# Research & Experiment Tree

A central, living list of ideas, concepts, and experiments. 
Status tags: `[TBD]`, `[ACTIVE]`, `[PAUSED]`, `[CONCLUDED]`

## 1. Constraint Solving & Generation
- **[ACTIVE] Decode-Time Constraint Solving**
  - *Concept:* Move hard-validity enforcement (acyclicity, exact degree) to a deterministic decode-time projector, applied *after* the DiT generates the continuous heatmap.
  - *Branch:* `exp/decode-time-solver`
- **[CONCLUDED] RLAIF Structural Penalty**
  - *Concept:* Differentiable penalties on continuous diffusion output to enforce discrete topological rules.
  - *Result:* Mode collapse (reward hacking). See `02_EXPERIMENTS_AND_RESULTS.md`.

## 2. Semantics & Execution
- **[TBD] Semantic Integration (The LLM Custodian)**
  - *Concept:* Map human intent into a Constant Pool, where the structural matrix interfaces with these literals via integer pointers.
  - *Branch:* TBD
- **[TBD] Execution Trace Reward**
  - *Concept:* RL based on functional correctness (executing the generated graph in the Graph-Walk Interpreter) rather than static topology penalties.
  - *Branch:* TBD
