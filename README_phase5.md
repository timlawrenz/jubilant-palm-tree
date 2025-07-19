# Phase 5 - Aligning Text and Code Embeddings

**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.

## Overview

Phase 5 is focused on creating multimodal embeddings that align natural language descriptions of Ruby methods with their structural AST representations. This phase would enable text-to-code and code-to-text generation capabilities by bridging the gap between natural language and code structure.

## Current Status

**Phase 5 is currently in planning phase.** We will create specific tickets for implementation as we work on this phase, and this README will be updated with detailed progress and results.

## Foundation Available

The successful completion of Phases 1-4 has established all necessary technical foundations:

- ✅ **High-quality AST dataset** from Phase 1 (1,896 Ruby methods)
- ✅ **Trained GNN encoder** producing 64D embeddings from Phase 2  
- ✅ **Validated embedding quality** through complexity prediction in Phase 3
- ✅ **Proven generative capability** through AST reconstruction in Phase 4

## Target Architecture

The envisioned approach would extend the successful autoencoder from Phase 4:

```
Text Description → Text Encoder → 64D Embedding
                                      ↓ (Alignment Loss)
Ruby AST → GNN Encoder (frozen) → 64D Embedding
```

### Key Components
- **Text Encoder**: Transformer-based encoder for method descriptions
- **Embedding Alignment**: Contrastive learning to align text and code embeddings  
- **Shared Embedding Space**: 64-dimensional space from Phase 4 as target
- **Frozen GNN**: Preserve learned AST representations from previous phases

## Implementation Approach

As we progress through Phase 5, this section will be updated with:

1. **Specific GitHub issues** created for each implementation milestone
2. **Detailed technical specifications** for each component
3. **Progress updates** and results achieved
4. **Lessons learned** and technical challenges encountered

## Next Steps

Phase 5 implementation will begin with creating detailed tickets for:

- Data collection and preparation for text-code pairs
- Text encoder architecture design and implementation
- Contrastive learning framework development
- Evaluation methodology establishment

This README will be continuously updated as work progresses on Phase 5.