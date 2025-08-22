# Phase 4 - AST Autoencoder for Code Generation

**Goal**: To build and train a GNN-based decoder that can reconstruct a Ruby method's AST from its learned embedding, validating the generative potential of the embeddings.

## Overview

Phase 4 represents a major advancement from complexity prediction to code generation. Building on the successful embeddings from Phase 3, this phase developed a complete autoencoder architecture. However, the initial implementation faced significant challenges, including numerical instability (`NaN` loss) and a failure to generate valid code. This phase documents the iterative process of identifying and fixing these critical issues.

## Phase 4 Issues Completed (14 Issues)

### Core Architecture Development
#### [Issue #34: Autoencoder Model Definition](https://github.com/timlawrenz/jubilant-palm-tree/issues/34)
- Created `ASTAutoencoder` combining `RubyComplexityGNN` (encoder) with a new `ASTDecoder`.
- **Result**: Functional autoencoder architecture.

#### [Issue #35 & #163: AST Reconstruction Loss Function](https://github.com/timlawrenz/jubilant-palm-tree/issues/35)
- **V1 (`ast_reconstruction_loss_comprehensive`)**: Implemented a loss function with node type and parent prediction. However, this version was numerically unstable, causing `NaN` loss for single-node graphs.
- **V2 (`ast_reconstruction_loss_improved`)**: Implemented a more robust, semantically-aware loss function with weighted components for type, edge, role, and name prediction, resolving the `NaN` issue and providing better training guidance.
- **Result**: A stable and more sophisticated loss function ready for training.

#### [Issue #36: Autoencoder Training Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/36)
- Built the complete training pipeline (`train_autoencoder.py`).
- **Result**: A script capable of training the decoder, which was instrumental in debugging the loss function and model output.

### Evaluation and Validation
#### [Issue #37 & #50: Evaluation Framework](https://github.com/timlawrenz/jubilant-palm-tree/issues/37)
- Created a consolidated evaluation script (`scripts/evaluate_model.py`) and notebook.
- Implemented a robust set of metrics: **Syntactic Validity**, **AST Isomorphism**, **Normalized BLEU Score**, and **Standard BLEU Score**.
- **Result**: A comprehensive "final exam" for the model, capable of assessing both structural and semantic correctness.

### Quality and Robustness Improvements
#### [Issue #159 & #162: Fixing NaN Loss and Model Output](https://github.com/timlawrenz/jubilant-palm-tree/issues/159)
- **Problem**: Training produced `NaN` loss, and the model generated empty or invalid ASTs.
- **Root Cause**: The loss function was numerically unstable for single-node graphs, and the `ASTDecoder` had a critical flaw in its graph construction logic.
- **Solution**:
    1.  Fixed the loss function to handle edge cases gracefully.
    2.  Corrected the `ASTDecoder` to ensure it can only predict parent nodes that exist within the current graph's context.
- **Result**: A stable training process and a decoder capable of generating structurally valid (though not yet semantically correct) ASTs.

## Technical Architecture

### ASTAutoencoder Architecture
```python
ASTAutoencoder(
  encoder: RubyComplexityGNN (FROZEN)
  decoder: ASTDecoder (TRAINABLE)
)
```

### Key Components
- **Frozen Encoder**: Pre-trained `RubyComplexityGNN` preserving learned representations.
- **Trainable Decoder**: The `ASTDecoder`, which was debugged and fixed to produce valid graph structures.
- **Improved Loss Function**: The `ast_reconstruction_loss_improved` provides stable and semantically-aware training signals.
- **Evaluation Tools**: A suite of scripts (`evaluate_model.py`, `pretty_print_ast.rb`, `check_syntax.rb`, `normalize_code.rb`) to rigorously assess model performance.

## Reconstruction Quality Results

### Initial State: Failure to Generate
Initial evaluations of the model trained with the `comprehensive` loss function yielded scores of zero across all metrics. Investigation revealed the model was not producing "garbage" output, but **no output at all**, because the decoder was generating invalid ASTs.

### Post-Optimization Results: Still Zero
After a massive effort to optimize the training pipeline—achieving a **~2500x speedup** in epoch time—the model was trained to full convergence. The final validation loss was **6.4203**. However, a full evaluation on the test set revealed that the model is still unable to generate syntactically valid code.

| Metric                     | Score |
| -------------------------- | ----- |
| Syntactic Validity (%)     | 0.0   |
| AST Isomorphism (%)        | 0.0   |
| Avg. Normalized BLEU Score | 0.0   |
| Avg. Standard BLEU Score   | 0.0   |

This is a critical result. It demonstrates that the current model architecture, even when trained efficiently and to convergence, is not powerful enough to learn the hard constraints of AST structure. The loss function, which only weakly penalizes incorrect edge counts, does not provide a strong enough signal for structural learning.

This outcome definitively proves that a more sophisticated approach is needed to enforce structural correctness.

## Next Steps: A New Experimental Plan

The massive performance gains from the optimization work are now a strategic advantage, enabling rapid experimentation. We will exhaust the potential of the current non-autoregressive approach before considering a shift in architecture.

### Experiment 1: Explicit Parent Prediction (High-Impact)

The most critical weakness of the current model is its lack of explicit structural prediction. This experiment will address that directly.

-   **Hypothesis:** Forcing the model to predict the parent of each node will provide a much stronger learning signal for AST structure than the current edge-counting heuristic.
-   **Implementation:**
    1.  **Modify `ASTDecoder`:** Add a new output head to predict `parent_logits` for each node.
    2.  **Upgrade Loss Function:** Add a **Parent Prediction Loss** component (cross-entropy) to `ast_reconstruction_loss_improved`.
-   **Expected Outcome:** A significant improvement in the `AST Isomorphism` score, hopefully bringing it above zero for the first time.

### Experiment 2: Increase Model Capacity and Expressiveness

If the parent prediction model still struggles, we can leverage our fast training times to increase the model's power.

-   **Hypothesis:** A larger or more expressive model will be better able to capture the complex patterns of code structure.
-   **Implementation (choose one or more):**
    -   **Go Deeper:** Increase the number of GNN layers in the decoder (e.g., from 3 to 5).
    -   **Go Wider:** Increase the hidden dimension size (e.g., from 64 to 128 or 256).
    -   **Use a More Powerful GNN Layer:** Replace `SAGEConv` with a **Graph Attention Network (GAT)** to allow the model to learn the relative importance of different nodes.
-   **Expected Outcome:** An improvement in the overall loss and evaluation metrics, assuming the model is not already overfitting.

This structured, iterative approach will allow us to systematically determine if a non-autoregressive model can be made to succeed at this task.
