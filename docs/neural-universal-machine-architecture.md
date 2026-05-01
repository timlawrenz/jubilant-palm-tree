# Neural Universal Machine: Architecture & Implementation

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

## 4. The Training Objective: Optimal Transport Flow Matching

Instead of DDPM, we use **Conditional Flow Matching (Optimal Transport)**.
Flow matching draws a mathematically straight line from pure noise to the clean graph, and trains the network to predict the constant velocity along that line.

*   **The Target**: $x_1$ (Clean 3-channel adjacency matrix)
*   **The Noise**: $x_0 \sim \mathcal{N}(0, I)$
*   **The Interpolation**: $x_t = (1-t)x_0 + tx_1$
*   **The Target Velocity**: $v_t = x_1 - x_0$

**Masked Loss**: Because our graphs vary in size (padded to 128x128), calculating standard MSE would penalize the model for failing to perfectly denoise the empty void. We use a boolean `padding_mask` to zero out the error in the padded regions, averaging the loss ONLY over the valid node intersections.

## 5. Inference: Euler ODE Solver & Discretization

Inference is fully deterministic. We start at pure noise and follow the predicted vector field to $t=1$.
```python
for step in range(num_steps):
    t_val = step * dt
    velocity = model(x, t, motifs)
    x = x + (velocity * dt) # Euler Step
```

**Thresholding**: The output of the DiT is continuous. To yield a runnable execution graph, we push the continuous probabilities through a `Sigmoid` and threshold at `0.5`, snapping the vector field back into rigid 1s and 0s. 

## 6. The Validation Harness: The 5 Laws of Physics

MSE loss only measures continuous proximity. To measure true Syntactic Validity Rate (SVR), we built a deterministic PyTorch grader that runs 5 strict topological checks against the thresholded matrix:

1.  **Execution Out-Degree Laws**: E.g., `[Condition]` must have exactly 2 outgoing execution edges (True/False).
2.  **Data In-Degree (Arity) Laws**: E.g., `[State]` writes must have exactly 1 incoming data edge.
3.  **No Orphans (Reachability)**: A Breadth-First Search confirms no disconnected logic islands.
4.  **Acyclic Data Plane**: A Depth-First Search confirms the data dependencies contain zero paradoxes/cycles.
5.  **Terminal Sink**: A reverse-BFS ensures no infinite loops exist without an escape path to a `[Boundary]`.

This acts as a passive metric during Flow Matching pre-training, and can be used as an active RLAIF reward signal during fine-tuning.

## 7. The Workflow: Two Branches of Government

To extract the 1D Motif list from the user's prompt without needing an autoregressive GNN, we use an LLM as the **Architect (Legislative Branch)**.
*   **The Semantic LLM**: Translates human intent into the "Bill of Materials" (the 1D array of Motifs) and populates the Literal Pool.
*   **The Structural DiT**: Takes the unrouted ingredients and acts as the **Plumber (Executive Branch)**, routing the complex `EXECUTION` and `DATA` edges.

This completely separates semantic reasoning (LLM strength) from rigorous multi-dimensional state management (DiT strength).
