# Phase 5 - Aligning Text and Code Embeddings

**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.

**Status**: 🚧 **IN PROGRESS**

## Overview

Phase 5 was focused on creating multimodal embeddings that align natural language descriptions of Ruby methods with their structural AST representations. This was intended to enable text-to-code and code-to-text retrieval.

While a complete dual-encoder architecture was successfully implemented and trained, the resulting model proved to be **non-functional**. The evaluation showed that it was unable to reliably align text and code embeddings, with performance metrics near random chance.

This document details the technical architecture and implementation that was attempted.

## Final Assessment

The text-code alignment model is **completely non-functional**.

- **Performance**: The model's ability to retrieve the correct code for a given text description (and vice-versa) is near zero. The final evaluation showed **Recall@10 scores of only 2-3%**, which is barely above random chance.
- **Misleading Metrics**: The "43.5% loss improvement" observed during training proved to be a **misleading vanity metric**. It had no correlation with the model's actual performance on the downstream retrieval task.

The model failed to achieve its primary goal. The alignment between the text and code embedding spaces was not successfully learned.

## Technical Implementation Details

### AlignmentModel Architecture

The `AlignmentModel` class implements a dual-encoder architecture designed to align text descriptions with code embeddings in a shared 64-dimensional space.

```
Text Description → Text Encoder → Projection Head → 64D Embedding
                                      ↓ (Alignment)
Ruby AST → Code Encoder (frozen) → 64D Embedding
```

- **Code Encoder (Frozen)**: A pre-trained `RubyComplexityGNN` is used to generate code embeddings. It is frozen to preserve the representations learned in earlier phases.
- **Text Encoder**: A `SentenceTransformer` model (`all-MiniLM-L6-v2`) is used to generate high-quality text embeddings from method names and docstrings.
- **Projection Head**: A simple linear layer projects the higher-dimensional text embeddings down to the shared 64-dimensional space.

### Paired Dataset

A paired dataset was created containing `(code, text)` pairs for training.
- The dataset contains over 155,000 Ruby methods.
- Descriptions are sourced from method names, docstrings, and RSpec test descriptions.
- A custom data loader (`PairedDataset`) was implemented to randomly sample one description for each method during training.

### Contrastive Loss

The model was trained using a contrastive loss function (InfoNCE) to minimize the distance between corresponding `(code, text)` pairs and maximize the distance between non-corresponding pairs in the embedding space.

### Training Loop

A full training pipeline (`train_alignment.py`) was implemented, which included:
- Loading the paired dataset.
- Initializing the `AlignmentModel`.
- A standard training/validation loop.
- Checkpointing to save the model with the best validation loss.
- Early stopping to prevent overfitting.

While the training loop ran successfully and the loss metric decreased, this did not translate into a functional model, as confirmed by the final evaluation.