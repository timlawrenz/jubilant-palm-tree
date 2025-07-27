# Phase 2 - Model Setup & Training

**Goal**: To build, train, and benchmark the GNN model for complexity prediction.

## Overview

Phase 2 focused on implementing the core Graph Neural Network architecture and training pipeline for Ruby code complexity prediction. This phase established the foundation for learning meaningful structural representations from Abstract Syntax Trees (ASTs).

## Phase 2 Issues Completed

### [Issue #5: Python Environment Setup](https://github.com/timlawrenz/jubilant-palm-tree/issues/5)
- Set up Python environment and dependencies for PyTorch Geometric
- Configured CUDA support for GPU acceleration
- Established development environment for GNN training

### [Issue #9: Data Ingestion & Graph Conversion](https://github.com/timlawrenz/jubilant-palm-tree/issues/9)
- Implemented data loading from Phase 1 JSONL files
- Created AST to PyTorch Geometric graph conversion
- Built node feature encoding for Ruby AST nodes (74-dimensional one-hot encoding)
- Established train/validation/test data loaders

### [Issue #10: GNN Model Definition](https://github.com/timlawrenz/jubilant-palm-tree/issues/10)
- Designed `RubyComplexityGNN` architecture using SAGE convolution layers
- Implemented 3-layer GNN with 64-dimensional hidden representations
- Added global mean pooling for graph-level embeddings
- Created complexity prediction head with regression output

### [Issue #11: Training & Validation Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/11)
- Built complete training pipeline with Adam optimizer
- Implemented learning rate scheduling and early stopping
- Added model checkpointing for best validation loss
- Created comprehensive logging and progress tracking

### [Issue #12: Heuristic Benchmark Implementation](https://github.com/timlawrenz/jubilant-palm-tree/issues/12)
- Implemented baseline complexity prediction using simple heuristics
- Created keyword-based complexity estimation for comparison
- Established baseline metrics (MAE: 4.46) for model evaluation

## Technical Architecture

### Model Architecture
```python
RubyComplexityGNN(
  convs: ModuleList of SAGEConv layers (3 layers)
  hidden_dim: 64
  num_layers: 3
  dropout: 0.1
  pool: Global mean pooling
  predictor: Linear layer for complexity regression
)
```

### Key Features
- **Node Features**: 74-dimensional one-hot encoding for Ruby AST node types
- **Graph Convolution**: SAGE (GraphSAINT) convolution for neighbor aggregation
- **Global Pooling**: Mean pooling to create graph-level representations
- **Regression Head**: Single linear layer for complexity prediction
- **Regularization**: Dropout (0.1) and L2 weight decay

### Training Configuration
- **Optimizer**: Adam with learning rate 0.001
- **Batch Size**: 32 graphs per batch
- **Epochs**: 10 (Phase 2), later extended to 100 (Phase 3)
- **Device**: CUDA-enabled for GPU acceleration
- **Loss Function**: Mean Squared Error (MSE)

## Results Achieved

### Model Performance
- **Training**: Successfully converged with decreasing loss
- **Validation**: Achieved stable validation metrics
- **Generalization**: Model learned meaningful structural patterns
- **Baseline Comparison**: Prepared foundation for Phase 3 evaluation

### Technical Validation
- ✅ Complete data pipeline from JSONL to PyTorch Geometric graphs
- ✅ Successful GNN training with stable convergence
- ✅ Model checkpointing and state persistence
- ✅ Heuristic baseline established for comparison
- ✅ GPU acceleration working properly

### Dataset Statistics
- **Training Set**: 1,516 Ruby methods (80%)
- **Validation Set**: 189 Ruby methods (10%)
- **Test Set**: 189 Ruby methods (10%)
- **Node Types**: 74 unique AST node types encoded
- **Complexity Range**: 2.0 to 100.0 cyclomatic complexity

## File Structure Created

```
src/
├── models.py              # RubyComplexityGNN model definition
├── data_processing.py     # Graph conversion and data loading
└── __init__.py           # Package initialization

train.py                  # Main training script
requirements.txt          # Python dependencies
```

## Next Steps to Phase 3

Phase 2 successfully established the GNN training foundation. The trained model (`models/best_model.pt`) is ready for comprehensive evaluation in Phase 3, where:

1. Final performance metrics will be calculated
2. Comparison against heuristic baseline will be performed
3. Embedding visualization will reveal learned patterns
4. Complete project validation will be achieved

## Technical Dependencies

### Python Packages
- torch >= 1.9.0
- torch-geometric >= 2.0.0
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0

### Hardware Requirements
- CUDA-compatible GPU (recommended)
- 8GB+ RAM for dataset processing
- 4GB+ GPU memory for training

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Monitor training progress
# Training logs show epoch progress, losses, and model checkpointing
```

The successful completion of Phase 2 validated that Graph Neural Networks can learn from Ruby AST structure and established the foundation for complexity prediction, setting up Phase 3 for comprehensive model evaluation and analysis.