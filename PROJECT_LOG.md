# Project Log: May 1, 2026

## Milestone 3: Validation & The Continuous/Discrete Representation Gap
**Status:** Core Hypothesis Proven / Architectural Refinement Required

### Summary of Findings:
The Phase 1 Curriculum training run (Flow Matching on 10-node graphs) successfully validated the core hypothesis of the Neural Universal Machine. The Diffusion Transformer (DiT) demonstrated the ability to natively learn complex topological rules (graph physics) from pure Gaussian noise, bypassing traditional text-based AST generation.

### Metric Analysis:
The validation harness revealed a distinct split in metric success, directly mapping to the multi-channel tensor architecture:

*   **Topological Success (`No_Orphan_Pass`, `Acyclic_Data_Pass`)**: The model achieved exceptional performance on Channel 0 (Presence) and Channel 1 (Edge Type). The model successfully learned to route Directed Acyclic Graphs, ensuring unbroken pathing and preventing data paradoxes.
*   **Categorical Failure (`Out_Degree_Pass`, `In_Degree_Pass`)**: The model struggled to satisfy strict integer constraints required for execution branching and argument ordering.

**Root Cause:** The failure stems from the "Continuous Index Channel." Channel 2 (`input_index`) is currently modeled as a continuous MSE regression target. During inference, continuous predictions (e.g., 0.49 and 0.51) are snapped via rounding to integers, causing frequent collisions (e.g., a `[Condition]` node routing two `True` branches and zero `False` branches). Continuous Flow Matching vectors cannot reliably enforce the mutual exclusivity required for discrete index assignment.

### Next Immediate Technical Step:

1.  **Architectural Refinement:** Transition Channel 2 (`input_index`) from a continuous MSE regression target to a Categorical Classification target.
2.  **Implementation:** The DiT output will be modified so that Channels 0 and 1 remain continuous targets for the Flow Matching vector field, while Channel 2 predicts logits across $K$ discrete index classes (e.g., 0, 1, 2, 3), trained via CrossEntropyLoss. This will allow the transformer's attention mechanism to natively distribute edge indices without rounding collisions.