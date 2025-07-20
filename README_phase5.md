# Phase 5 - Aligning Text and Code Embeddings

**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.

## Overview

Phase 5 is focused on creating multimodal embeddings that align natural language descriptions of Ruby methods with their structural AST representations. This phase would enable text-to-code and code-to-text generation capabilities by bridging the gap between natural language and code structure.

## Current Status

**Phase 5 data preparation is complete.** The paired dataset now includes method name descriptions for all 155,949 Ruby methods, with additional docstring descriptions where available.

**✅ NEW: AlignmentModel Implementation Complete** - The dual-encoder model for aligning text and code embeddings is now implemented and tested.

## AlignmentModel Architecture

The `AlignmentModel` class implements a dual-encoder architecture that aligns text descriptions with code embeddings in a shared 64-dimensional space.

### Architecture Components

```
Text Description → Text Encoder → Projection Head → 64D Embedding
                                      ↓ (Alignment)
Ruby AST → Code Encoder (frozen) → 64D Embedding
```

#### 1. Code Encoder (Frozen)
- **Model**: Pre-trained, frozen `RubyComplexityGNN` without the prediction head
- **Input**: PyTorch Geometric Data object containing Ruby AST
- **Output**: 64-dimensional code embeddings
- **Status**: Frozen to preserve learned representations from previous phases

#### 2. Text Encoder  
- **Primary**: `SentenceTransformer` (all-MiniLM-L6-v2) for high-quality text embeddings
- **Fallback**: Custom `SimpleTextEncoder` using character-level features and LSTM for offline environments
- **Input**: List of text descriptions (method names, docstrings, test descriptions)
- **Output**: Text embeddings (384D for SentenceTransformer, configurable for SimpleTextEncoder)

#### 3. Projection Head
- **Model**: Linear layer (`torch.nn.Linear`)
- **Purpose**: Projects text embeddings to the same 64-dimensional space as code embeddings
- **Input**: Text encoder output (384D)
- **Output**: 64-dimensional aligned text embeddings

### Model Implementation

The `AlignmentModel` is implemented in `src/models.py` with the following key features:

```python
class AlignmentModel(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, 
                 text_model_name: str = 'all-MiniLM-L6-v2',
                 code_encoder_weights_path: str = None):
        # Initialize frozen code encoder
        # Initialize text encoder (with fallback support)
        # Create projection head for alignment
        
    def forward(self, data: Data, texts: list) -> dict:
        # Returns: {'code_embeddings': ..., 'text_embeddings': ...}
```

### Key Features

1. **Flexible Text Encoding**: Automatically falls back to SimpleTextEncoder if SentenceTransformers is unavailable
2. **Frozen Code Encoder**: Preserves learned AST representations from previous training phases
3. **Batch Processing**: Efficiently handles batches of graphs and texts
4. **Device Compatibility**: Works on both CPU and GPU
5. **Pre-trained Weights**: Supports loading pre-trained code encoder weights

### Usage Example

```python
from src.models import AlignmentModel
from torch_geometric.data import Data

# Initialize model
model = AlignmentModel(
    input_dim=74,  # Node feature dimension from dataset
    hidden_dim=64,
    code_encoder_weights_path="best_model.pt"  # Optional pre-trained weights
)

# Forward pass
graph_data = Data(x=node_features, edge_index=edges, batch=batch_indices)
texts = ["calculate total price", "process user data"]

outputs = model(graph_data, texts)
code_embeddings = outputs['code_embeddings']  # Shape: (batch_size, 64)
text_embeddings = outputs['text_embeddings']  # Shape: (batch_size, 64)

# Compute alignment loss (e.g., cosine similarity)
similarity = F.cosine_similarity(code_embeddings, text_embeddings, dim=1)
```

### Testing and Validation

Comprehensive testing has been implemented in `test_alignment_model.py` covering:

1. **Model Initialization**: Verifies proper setup of dual-encoder architecture
2. **Individual Encoders**: Tests code and text encoding separately
3. **Forward Pass**: Validates complete model forward pass
4. **Batch Processing**: Confirms handling of multiple graphs and texts
5. **Embedding Similarity**: Tests cosine similarity computation for alignment
6. **Device Compatibility**: Ensures CPU/GPU compatibility
7. **Real Data Processing**: Validates performance on actual dataset

All tests pass successfully, confirming the model meets the specified requirements.

### Definition of Done ✅

The AlignmentModel implementation satisfies all requirements from the issue:

- ✅ **Dual-encoder architecture**: Code encoder + Text encoder + Projection head
- ✅ **Frozen code encoder**: Uses existing RubyComplexityGNN without prediction head
- ✅ **Text encoder**: Sentence-transformers model (all-MiniLM-L6-v2) with fallback
- ✅ **Projection head**: Linear layer mapping text embeddings to 64D space
- ✅ **Forward pass capability**: Takes batch of graphs and texts, returns aligned embeddings
- ✅ **Same dimensionality**: Both encoders output 64-dimensional embeddings
- ✅ **Comprehensive testing**: All functionality validated with test suite

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

### Testing and Validation

Comprehensive testing has been implemented in `test_paired_dataset.py` with the following test cases:

1. **Dataset Loading**: Verifies successful loading of paired_data.jsonl (155,949 samples)
2. **Item Access**: Validates (graph_data, text_description) tuple returns
3. **Batch Collation**: Tests proper batching of multiple graph-text pairs
4. **DataLoader Functionality**: Confirms end-to-end batch iteration
5. **Description Variety**: Verifies random sampling from multiple descriptions

All tests pass successfully, confirming the data loader meets the specified requirements.

### Definition of Done ✅

- ✅ **Dataset reads paired_data.jsonl**: PairedDataset successfully loads the new format
- ✅ **Random description sampling**: Each access randomly selects from available descriptions
- ✅ **Returns (graph, text) tuples**: __getitem__ method returns correct format
- ✅ **Successful batch yielding**: DataLoader yields batches of (graph, text) pairs

The data loader implementation is complete and ready for Phase 5 training pipeline integration.

## Next Steps

Phase 5 implementation will begin with creating detailed tickets for:

- Text encoder architecture design and implementation
- Contrastive learning framework development
- Evaluation methodology establishment
- Training pipeline setup using the prepared paired dataset

This README will be continuously updated as work progresses on Phase 5.