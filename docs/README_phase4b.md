# Phase 4b - Hierarchical AST Generation (Coarse-to-Fine Approach)

**Goal**: To build and train a functional generative model that can construct a Ruby method's AST from a learned embedding using a hierarchical, top-down approach. This phase replaces the failed "one-shot" autoencoder from the original Phase 4.

**Status**: 🚧 **PLANNED**

## 1. Core Concept & Inspiration

The "one-shot" decoder in Phase 4 failed because it attempted to predict the entire, complex graph structure of an AST in a single forward pass. This approach proved incapable of learning the hard structural rules of code, resulting in a 0% success rate.

This new phase is directly inspired by the progressive, coarse-to-fine architecture of the `glowing-tribble` image generation project. We will adapt its core principles for the code domain:

-   **Progressive Generation**: Instead of generating a high-resolution image from a low-resolution one, we will generate a high-detail AST from a low-detail one (i.e., from the root downwards).
-   **Multi-Scale Guidance**: Instead of using a pyramid of DINOv2 image features, we will use a pyramid of simplified ASTs, where each level of the pyramid represents one additional level of depth in the tree.

The model will learn to generate an AST hierarchically, starting with the root and progressively adding children level by level, ensuring structural coherence at each step.

## 2. Proposed Architecture

The generation process will be a sequence of stages, where each stage is responsible for predicting one level of the AST.

```
                                     +-----------------+
                                     |  Text Embedding |
                                     +-----------------+
                                             |
                                             v
+------------------------------------------------------------------------------------------+
| Stage 0: Predict Root Node                                                               |
|   - GNN_0(embedding) -> AST_Level_0 (e.g., the 'DEF' node)                                 |
+------------------------------------------------------------------------------------------+
                                             |
                                             v
+------------------------------------------------------------------------------------------+
| Stage 1: Predict Children of Root                                                        |
|   - GNN_1(AST_Level_0) -> AST_Level_1 (e.g., 'ARGS', 'BLOCK' nodes connected to root)      |
+------------------------------------------------------------------------------------------+
                                             |
                                             v
+------------------------------------------------------------------------------------------+
| Stage 2: Predict Children at Depth 2                                                     |
|   - GNN_2(AST_Level_1) -> AST_Level_2 (e.g., nodes for arguments and method body)          |
+------------------------------------------------------------------------------------------+
                                             |
                                             v
                                           (...)
                                             |
                                             v
+------------------------------------------------------------------------------------------+
| Final Stage: Predict Leaf Nodes                                                          |
|   - GNN_N(AST_Level_N-1) -> Complete AST (with variable names, literals, etc.)             |
+------------------------------------------------------------------------------------------+

```

## 3. Execution Plan

### Task 1: Data Preparation - Create the "AST Pyramid" Dataset

-   **Objective**: Process the existing dataset to create the multi-level ground-truth data needed for training.
-   **Action**: Create a new script, `scripts/create_ast_level_dataset.py`.
-   **Process**:
    1.  Read each method's AST from `dataset/train.jsonl` and `dataset/validation.jsonl`.
    2.  For each AST, decompose it into a series of "level-graphs".
    3.  `level_0.json` would contain just the root node of each method.
    4.  `level_1.json` would contain the root and its immediate children.
    5.  ...and so on, up to the maximum depth of the ASTs in our dataset.
-   **Output**: A new directory, `dataset/hierarchical`, containing the training data for each level of the AST generation process.

### Task 2: Model Implementation - The Hierarchical Decoder

-   **Objective**: Implement the new staged decoder architecture.
-   **Action**: Define a new model, `HierarchicalASTDecoder`, in `src/models.py`.
-   **Architecture**:
    -   The model will be a container for a sequence of smaller GNN blocks (e.g., `GNN_Level_0`, `GNN_Level_1`, etc.).
    -   The `forward` pass will iteratively call each GNN block, taking the graph from the previous stage as input and predicting the nodes/edges for the next level.

### Task 3: Training Pipeline

-   **Objective**: Create a training script for the new hierarchical model.
-   **Action**: Create a new script, `train_hierarchical.py`.
-   **Process**:
    1.  The training loop will iterate through the levels of the AST.
    2.  At each level `i`, it will use the `HierarchicalASTDecoder` to predict the nodes and edges for that level.
    3.  The loss will be calculated by comparing the predicted `AST_Level_i` to the ground-truth `level_i.json` data.
    4.  The total loss will be a sum of the losses from each generation stage.

### Task 4: Inference and Evaluation

-   **Objective**: Build a script to generate code and evaluate the new model's performance.
-   **Action**: Create a new generation script, `generate_hierarchical.py`.
-   **Process**:
    -   The script will take a text prompt, get its embedding, and feed it into the trained `HierarchicalASTDecoder`.
    -   It will run the full, multi-stage generation process to produce a final AST.
    -   The final AST will be converted to Ruby code and evaluated using the same metrics as before: Syntactic Validity, AST Isomorphism, and BLEU score.

## 4. Expected Outcome

This phase will be considered a success if the `HierarchicalASTDecoder` can consistently generate **syntactically valid** ASTs for simple and moderately complex Ruby methods. Achieving a non-zero score on Syntactic Validity and AST Isomorphism will represent a monumental step forward from the complete failure of the original Phase 4 decoder.
