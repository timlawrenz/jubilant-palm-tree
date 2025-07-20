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

The `scripts/05_create_paired_dataset.rb` script now generates comprehensive paired data with support for three types of descriptions:

### Dataset Statistics
- **Total methods**: 155,949 (100% of processed methods)
- **Methods with method name descriptions**: 155,949 (100%)
- **Methods with docstring descriptions**: 10,028 (when available)
- **Methods with test descriptions**: Variable (depends on availability of RSpec files)
- **Output file**: `./dataset/paired_data.jsonl`

### Description Sources
Each method entry includes a `descriptions` array with up to three types of descriptions:

1. **Method Name Description** (all methods): 
   - Source: `"method_name"`
   - Transforms snake_case method names to natural language descriptions
   - Example: `get_user` → `"gets user"`, `calculate_total_price` → `"calculates total price"`

2. **Docstring Description** (when available):
   - Source: `"docstring"`
   - Extracted from RDoc/YARD comments in the source code
   - Example: `"Calculates the total price including tax and discounts"`

3. **Test Description** (when available):
   - Source: `"test_description"`
   - Extracted from RSpec `it` block descriptions that test the method
   - Example: `"calculates the total correctly"`, `"returns the proper value"`
   - Uses heuristics to link test descriptions to specific methods being tested

### Test Description Extraction

The script implements a sophisticated approach to extract meaningful descriptions from RSpec test files:

#### RSpec File Discovery
- Automatically scans for files ending in `_spec.rb`
- Maps implementation files to corresponding spec files using common Rails conventions:
  - `app/models/user.rb` → `spec/models/user_spec.rb`
  - `lib/calculator.rb` → `spec/lib/calculator_spec.rb` or `spec/calculator_spec.rb`

#### Test-to-Method Linking
Uses multiple heuristics to identify which test descriptions apply to specific methods:

1. **Description Pattern Matching**: Checks if the method name appears in the test description
2. **Method Call Analysis**: Parses the test body to find method calls on test subjects
3. **Subject Identification**: Recognizes common RSpec patterns like `subject.method_name` or `described_class.method_name`

#### RSpec DSL Parsing
- Parses RSpec files using the Ruby AST parser
- Extracts descriptions from `it "description"` blocks
- Filters out common RSpec helper methods to focus on the actual method under test

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
    },
    {
      "source": "test_description",
      "text": "calculates the total correctly"
    },
    {
      "source": "test_description",
      "text": "includes tax in the calculation"
    }
  ]
}
```

### Implementation Notes

- **Graceful Degradation**: The script works even when RSpec files are not available, ensuring all methods still receive method name and docstring descriptions
- **Performance Optimization**: Uses caching to avoid re-parsing the same spec files multiple times
- **Error Handling**: Continues processing even if individual spec files cannot be parsed
- **Flexible File Mapping**: Supports various project structures and file organization patterns

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

## Updated Data Loader Implementation

**Status: ✅ COMPLETE** - The data loader has been successfully updated to handle the paired data format.

### New PairedDataset Class

A new `PairedDataset` class has been implemented in `src/data_processing.py` with the following capabilities:

- **Reads paired_data.jsonl**: Loads the complete paired dataset with 155,949 method entries
- **Random description sampling**: For each method access, randomly selects one description from the available descriptions array
- **Graph-text pairs**: Returns tuples of `(graph_data, text_description)` instead of the previous format
- **Reproducible sampling**: Supports optional seed parameter for consistent results during testing

### Key Features

- **Multiple description sources**: Handles method name descriptions, docstring descriptions, and test descriptions
- **Graceful fallback**: Uses method name as fallback if no descriptions are available
- **Memory efficient**: Loads data once and samples descriptions on-demand
- **Compatible interface**: Integrates seamlessly with existing graph processing infrastructure

### Usage Example

```python
from src.data_processing import PairedDataset, create_paired_data_loaders

# Create dataset and loader
loader = create_paired_data_loaders(
    paired_data_path="dataset/paired_data.jsonl",
    batch_size=32,
    shuffle=True,
    seed=42  # Optional for reproducible sampling
)

# Iterate through batches
for batched_graphs, text_descriptions in loader:
    # batched_graphs: Dictionary with graph data (x, edge_index, batch, etc.)
    # text_descriptions: List of strings, one per graph in the batch
    print(f"Batch size: {len(text_descriptions)}")
    print(f"Sample description: {text_descriptions[0]}")
```

### Data Format

Each iteration yields:
- **Batched graph data**: Standard graph batch format with node features, edge indices, and batch assignments
- **Text descriptions**: List of strings corresponding to each graph in the batch

The loader successfully handles the complete dataset and yields batches of (graph, text) pairs as required.

## Next Steps

Phase 5 implementation will begin with creating detailed tickets for:

- Text encoder architecture design and implementation
- Contrastive learning framework development
- Evaluation methodology establishment
- Training pipeline setup using the prepared paired dataset

This README will be continuously updated as work progresses on Phase 5.