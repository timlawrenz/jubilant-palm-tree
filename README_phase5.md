# Phase 5 - Aligning Text and Code Embeddings

**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.

## Overview

Phase 5 is focused on creating multimodal embeddings that align natural language descriptions of Ruby methods with their structural AST representations. This phase would enable text-to-code and code-to-text generation capabilities by bridging the gap between natural language and code structure.

## Current Status

**Phase 5 data preparation is complete.** The paired dataset now includes method name descriptions for all 155,949 Ruby methods, with additional docstring descriptions where available.

## Foundation Available

The successful completion of Phases 1-4 has established all necessary technical foundations:

- ✅ **High-quality AST dataset** from Phase 1 (155,949 Ruby methods)
- ✅ **Trained GNN encoder** producing 64D embeddings from Phase 2  
- ✅ **Validated embedding quality** through complexity prediction in Phase 3
- ✅ **Proven generative capability** through AST reconstruction in Phase 4
- ✅ **Complete paired dataset** with method name descriptions for all methods (155,949 entries)

## Paired Dataset Details

The `scripts/05_create_paired_dataset.rb` script now generates comprehensive paired data:

### Dataset Statistics
- **Total methods**: 155,949 (100% of processed methods)
- **Methods with both descriptions**: 10,028 (method name + docstring)
- **Methods with method name only**: 145,921 (method name description)
- **Output file**: `./dataset/paired_data.jsonl`

### Description Sources
Each method entry includes a `descriptions` array with:

1. **Method Name Description** (all methods): 
   - Source: `"method_name"`
   - Transforms snake_case method names to natural language descriptions
   - Example: `get_user` → `"gets user"`, `calculate_total_price` → `"calculates total price"`

2. **Docstring Description** (when available):
   - Source: `"docstring"`
   - Extracted from RDoc/YARD comments in the source code
   - Example: `"Calculates the total price including tax and discounts"`

### Example Entry Structure
```json
{
  "id": "unique_method_id",
  "repo_name": "repository_name",
  "file_path": "./path/to/file.rb",
  "method_name": "calculate_total_price",
  "method_source": "def calculate_total_price...",
  "ast_json": "{...}",
  "descriptions": [
    {
      "source": "method_name",
      "text": "calculates total price"
    },
    {
      "source": "docstring",
      "text": "Calculates the total price including tax and discounts"
    }
  ]
}
```

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

- Text encoder architecture design and implementation
- Contrastive learning framework development
- Evaluation methodology establishment
- Training pipeline setup using the prepared paired dataset

This README will be continuously updated as work progresses on Phase 5.