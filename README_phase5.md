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
    code_encoder_weights_path="models/best_model.pt"  # Optional pre-trained weights
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

**✅ NEW: Improved RSpec Parsing with Custom Formatter** - The script now uses a custom RSpec formatter instead of manual AST parsing for more accurate test description extraction.

The script implements a sophisticated approach to extract meaningful descriptions from RSpec test files using two complementary methods:

#### Primary Method: Custom RSpec Formatter
- **Implementation**: `CustomRSpecFormatter` class that hooks into RSpec's event system
- **Events**: Listens for `example_group_started`, `example_group_finished`, `example_started`, and `dump_summary`
- **Dry-run Mode**: Runs RSpec with `--dry-run` flag to collect metadata without executing test code
- **Contextual Descriptions**: Builds full contextual sentences by combining describe/context blocks with it descriptions
- **Example Output**: "User, with a valid profile, can log in successfully" instead of just "can log in successfully"

#### Fallback Method: Manual AST Parsing
- **Compatibility**: Falls back to original Ruby AST parsing if RSpec formatter fails
- **Robustness**: Ensures the script continues working even with malformed spec files
- **Legacy Support**: Maintains compatibility with non-standard RSpec files

#### Implementation Details

**Custom RSpec Formatter**:
```ruby
class CustomRSpecFormatter
  # Hooks into RSpec's event system
  RSpec::Core::Formatters.register self, 
    :example_group_started, 
    :example_group_finished,
    :example_started,
    :dump_summary

  def example_group_started(notification)
    # Builds context stack from describe/context blocks
  end

  def example_started(notification)
    # Combines context with example description
    # Creates: "#{context_parts.join(', ')}, #{example_description}"
  end
end
```

**Integration in Main Script**:
```ruby
def extract_test_descriptions_using_rspec_formatter(spec_file, target_method_name)
  # Configure RSpec for dry-run mode
  RSpec.configure { |config| config.dry_run = true }
  
  # Run RSpec with custom formatter
  exit_code = RSpec::Core::Runner.run([spec_file, '--dry-run'])
  
  # Extract relevant descriptions using existing heuristics
  formatter.collected_examples.select do |example|
    test_likely_targets_method_from_rspec(example, target_method_name)
  end
end
```

#### RSpec File Discovery
- Automatically scans for files ending in `_spec.rb`
- Maps implementation files to corresponding spec files using common Rails conventions:
  - `app/models/user.rb` → `spec/models/user_spec.rb`
  - `lib/calculator.rb` → `spec/lib/calculator_spec.rb` or `spec/calculator_spec.rb`

#### Advanced Test-to-Method Linking
The system uses multiple sophisticated heuristics to identify which test descriptions apply to specific methods:

**RSpec Formatter Heuristics** (Primary):
1. **Contextual Description Matching**: Analyzes full contextual descriptions from RSpec formatter
2. **Described Class Integration**: Uses RSpec's `described_class` metadata when available
3. **Method Name Pattern Recognition**: Enhanced fuzzy matching with context awareness
4. **Context Stack Analysis**: Examines describe/context hierarchy for method relevance

**AST Parsing Heuristics** (Fallback):
1. **Description Pattern Matching**: Checks if the method name appears in the test description
2. **Method Call Analysis**: Parses the test body to find method calls on test subjects
3. **Subject Identification**: Recognizes common RSpec patterns like `subject.method_name` or `described_class.method_name`
4. **Behavioral Pattern Matching**: Identifies tests based on behavioral descriptions and patterns

#### Benefits of the New Approach
- **Higher Accuracy**: RSpec formatter provides better context than manual AST parsing
- **Richer Descriptions**: Full contextual sentences instead of isolated test descriptions
- **Better Maintainability**: Uses RSpec's official event system instead of custom AST traversal
- **Robust Fallback**: Maintains compatibility with edge cases through dual approach
- **Performance**: RSpec dry-run mode is more efficient than manual parsing for complex specs

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
      "text": "User, #calculate_total_price, calculates the total correctly"
    },
    {
      "source": "test_description", 
      "text": "User, #calculate_total_price, includes tax in the calculation"
    },
    {
      "source": "test_description",
      "text": "User, with premium account, calculates total price with discount"
    }
  ]
}
```

### Implementation Notes

- **Dual Approach**: Primary RSpec formatter with AST parsing fallback ensures robustness
- **Graceful Degradation**: The script works even when RSpec files are not available, ensuring all methods still receive method name and docstring descriptions  
- **Performance Optimization**: Uses caching to avoid re-parsing the same spec files multiple times
- **Error Handling**: Continues processing even if individual spec files cannot be parsed
- **Flexible File Mapping**: Supports various project structures and file organization patterns
- **Enhanced Context**: RSpec formatter provides richer contextual information than manual parsing

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

### Dataset Organization

The paired dataset is split into separate training and validation sets for proper model evaluation:

- **Training set**: `dataset/train_paired_data.jsonl` (140,354 samples, 90%)
- **Validation set**: `dataset/validation_paired_data.jsonl` (15,595 samples, 10%)
- **Original dataset**: `dataset/paired_data.jsonl` (155,949 total samples)

This ensures validation metrics are reliable and not overly optimistic from data leakage.

To recreate the split or use a different ratio, run:
```bash
python scripts/split_paired_dataset.py
```

### Usage Example

```python
from src.data_processing import PairedDataset, create_paired_data_loaders

# Create dataset and loader
loader = create_paired_data_loaders(
    paired_data_path="dataset/train_paired_data.jsonl",
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

## Contrastive Loss Functions ✅

**Status: ✅ COMPLETE** - Contrastive loss functions for code-text alignment have been successfully implemented.

### Implementation

The contrastive loss functions are implemented in `src/loss.py` and provide multiple approaches for training aligned embeddings between code and text representations.

### Available Loss Functions

#### 1. InfoNCE Loss
```python
from src.loss import info_nce_loss

loss = info_nce_loss(code_embeddings, text_embeddings, temperature=0.07)
```

**Purpose**: Information Noise Contrastive Estimation (InfoNCE) loss for contrastive learning
**Method**: Uses cross-entropy between similarity scores and correct indices
**Features**:
- Temperature-scaled cosine similarities 
- Symmetric loss (code-to-text and text-to-code)
- Standard choice for contrastive multimodal learning

#### 2. Cosine Embedding Loss
```python
from src.loss import cosine_embedding_loss

loss = cosine_embedding_loss(code_embeddings, text_embeddings, margin=0.2)
```

**Purpose**: Simple cosine similarity-based contrastive loss
**Method**: Encourages high similarity for positive pairs, low similarity for negative pairs
**Features**:
- MSE loss for positive pairs (target similarity = 1.0)
- Margin-based loss for negative pairs
- Handles single-sample batches gracefully

#### 3. Simple Contrastive Loss
```python
from src.loss import simple_contrastive_loss

loss = simple_contrastive_loss(code_embeddings, text_embeddings, temperature=0.1)
```

**Purpose**: Straightforward contrastive loss maximizing cosine similarity
**Method**: Negative mean cosine similarity between positive pairs
**Features**:
- Simplest implementation
- Direct optimization of similarity
- Temperature scaling for gradient control

### Usage Example

```python
import torch
from src.models import AlignmentModel
from src.loss import info_nce_loss
from torch_geometric.data import Data

# Initialize alignment model
model = AlignmentModel(input_dim=74, hidden_dim=64)

# Forward pass
graph_data = Data(x=node_features, edge_index=edges, batch=batch_indices)
texts = ["calculate total price", "process user data"]

outputs = model(graph_data, texts)
code_embeddings = outputs['code_embeddings']  # Shape: (batch_size, 64)
text_embeddings = outputs['text_embeddings']  # Shape: (batch_size, 64)

# Compute contrastive loss
loss = info_nce_loss(code_embeddings, text_embeddings)

# Backpropagation
loss.backward()
```

### Key Features

1. **Multiple Loss Options**: InfoNCE, cosine embedding, and simple contrastive losses
2. **Robust Implementation**: Handles various batch sizes and embedding dimensions
3. **Gradient Flow**: All functions support gradient computation for training
4. **Temperature Scaling**: Configurable temperature parameters for fine-tuning
5. **Batch Processing**: Efficient computation for batched embeddings
6. **Device Compatibility**: Works on both CPU and GPU

### Testing and Validation

Comprehensive testing has been implemented in `test_contrastive_loss.py` covering:

1. **Identical Embeddings**: Verifies low loss for perfect alignment
2. **Random Embeddings**: Confirms higher loss for poor alignment  
3. **Opposite Embeddings**: Tests maximum loss for anti-aligned embeddings
4. **Gradient Flow**: Validates gradient computation through loss functions
5. **Batch Size Robustness**: Tests with various batch sizes (1, 2, 8, 16)
6. **Embedding Dimensions**: Validates across different dimensions (16, 32, 64, 128, 256)
7. **Temperature Scaling**: Confirms proper temperature parameter behavior

All tests pass successfully, confirming the loss functions meet the specified requirements.

### Definition of Done ✅

The contrastive loss implementation satisfies all requirements from issue #76:

- ✅ **Contrastive loss function**: InfoNCE loss implemented as primary option
- ✅ **Alternative approaches**: Cosine embedding loss and simple contrastive loss provided
- ✅ **Code/text embeddings input**: Functions take two tensors of embeddings
- ✅ **Scalar loss output**: All functions return single scalar loss values
- ✅ **Similarity-based logic**: Low loss for close pairs, high loss for distant pairs
- ✅ **Comprehensive testing**: All functionality validated with test suite

The implementation is ready for integration into the Phase 5 training pipeline.

## Alignment Training Loop ✅

**Status: ✅ COMPLETE** - The alignment training loop has been successfully implemented and tested.

### Implementation

The alignment training loop is implemented in `train_alignment.py` and provides a complete pipeline for training the dual-encoder alignment model using contrastive learning.

### Training Script Features

#### 1. Complete Training Pipeline
```bash
# Run full alignment training
python train_alignment.py

# Run demo training (faster, for testing)
python demo_train_alignment.py
```

#### 2. Key Components
- **Data Loading**: Uses `PairedDataset` to load graph-text pairs from separate train/validation splits
- **Model Initialization**: Creates `AlignmentModel` with frozen code encoder and trainable text projection head
- **Contrastive Learning**: Uses InfoNCE loss to align code and text embeddings
- **Training Loop**: Implements proper training/validation with progress monitoring
- **Model Checkpointing**: Saves best model weights based on validation loss
- **Early Stopping**: Prevents overfitting with configurable patience

#### 3. Training Configuration
```python
# Configurable hyperparameters
batch_size = 8          # Batch size for training
learning_rate = 1e-3    # Learning rate for text projection head
num_epochs = 20         # Maximum training epochs
patience = 5            # Early stopping patience
temperature = 0.07      # InfoNCE temperature parameter
```

#### 4. Model Architecture Details
- **Frozen Code Encoder**: Pre-trained `RubyComplexityGNN` (351,553 total params)
- **Trainable Components**: Only text projection head (338,368 params)
- **Embedding Dimension**: 64D shared space for both code and text
- **Loss Function**: InfoNCE contrastive loss with temperature scaling

### Training Results

#### Demo Training Performance
The demo training script demonstrates successful convergence:

```
🏁 Demo Training Summary
Initial loss (epoch 1): 1.3897
Final loss (epoch 10): 0.7855
✅ Loss decreased by: 0.6042
✅ Relative improvement: 43.5%

Loss progression:
  Epoch  1: 1.3897
  Epoch  2: 1.2987
  Epoch  3: 1.2377
  Epoch  4: 1.1619
  Epoch  5: 1.0982
  Epoch  6: 1.0376
  Epoch  7: 0.9552
  Epoch  8: 0.8710
  Epoch  9: 0.8524
  Epoch 10: 0.7855
```

#### Key Metrics Monitored
- **Training Loss**: InfoNCE contrastive loss (decreases over time)
- **Validation Loss**: Early stopping based on validation performance
- **Positive Pair Similarity**: Cosine similarity between aligned code-text pairs
- **Negative Pair Similarity**: Similarity between misaligned pairs
- **Alignment Gap**: Difference between positive and negative similarities

### Usage Example

```python
# Load trained alignment model
from src.models import AlignmentModel
import torch

# Initialize model with saved weights
model = AlignmentModel(
    input_dim=74,
    hidden_dim=64,
    code_encoder_weights_path="models/best_model.pt"
)

# Load trained alignment weights
checkpoint = torch.load("models/best_alignment_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])

# Encode code and text
code_embedding = model.encode_code(graph_data)      # (batch_size, 64)
text_embedding = model.encode_text(["description"]) # (batch_size, 64)

# Compute similarity
similarity = F.cosine_similarity(code_embedding, text_embedding, dim=1)
```

### Testing and Validation

Comprehensive testing implemented in `test_train_alignment.py`:

1. **Pipeline Testing**: Verifies complete training pipeline functionality
2. **Forward/Backward Pass**: Tests gradient flow through trainable components
3. **Loss Calculation**: Validates InfoNCE loss computation
4. **Model Saving**: Confirms checkpoint saving and loading
5. **Training Steps**: Demonstrates actual training iterations

All tests pass successfully, confirming the training implementation meets specifications.

### Definition of Done ✅

The alignment training implementation satisfies all requirements:

- ✅ **Training Script**: `train_alignment.py` successfully trains the alignment model
- ✅ **Paired Dataset Loading**: Uses `PairedDataset` to load (graph, text) pairs
- ✅ **AlignmentModel Initialization**: Frozen code encoder + trainable text projection head
- ✅ **Contrastive Loss**: InfoNCE loss for code-text alignment
- ✅ **Decreasing Training Loss**: Demonstrated 43.5% improvement over 10 epochs
- ✅ **Best Weight Saving**: Saves model with best validation performance
- ✅ **Progress Monitoring**: Tracks training metrics and alignment quality

### Output Files

- `models/best_alignment_model.pt`: Best model weights based on validation loss
- `models/demo_alignment_model.pt`: Demo training results for testing
- Training logs with epoch-by-epoch loss progression and metrics

Phase 5 alignment training implementation is complete and ready for production use.