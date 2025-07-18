# jubilant-palm-tree

## Overview

This project explores the potential of Graph Neural Networks (GNNs) to understand and generate Ruby code through Abstract Syntax Tree (AST) analysis.

**Current Phase**: Building a GNN-based decoder that can reconstruct a Ruby method's AST from learned embeddings, validating the generative potential of structural code representations.

## Phase 1 Results (Completed)

✅ **Successfully demonstrated that GNNs can predict Ruby code complexity**
- Trained GNN model achieved MAE of 4.27 (beating heuristic baseline of 4.46)
- Learned meaningful 64-dimensional embeddings that cluster methods by complexity
- Complete dataset of 1,896 Ruby methods from 8 open-source projects
- Full documentation available in [README_phase1.md](README_phase1.md)

## Phase 2 Goals (Current)

🎯 **GNN-based AST Decoder Development** ✅
- ✅ Built ASTAutoencoder that combines existing RubyComplexityGNN (encoder) with new ASTDecoder
- ✅ Encoder can extract meaningful 64-dimensional embeddings from Ruby method ASTs
- ✅ Decoder reconstructs AST structure from embeddings with configurable target nodes
- ✅ Complete forward pass: AST_in → embedding → AST_out implemented and tested
- ✅ Support for frozen encoder weights to preserve pre-trained representations
- ✅ **AST Reconstruction Loss Function** - Custom loss combining node type prediction and edge structure comparison
- 🔄 Training pipeline for autoencoder optimization (future work)
- 🔄 Advanced metrics for AST reconstruction quality (future work)

## Quick Start

### Prerequisites
- Ruby 2.7+ and Python 3.8+
- See [README_phase1.md](README_phase1.md) for complete setup instructions

### Phase 1 Dataset & Models (Available)
```bash
# The project includes:
./dataset/               # 1,896 processed Ruby methods (train/val/test splits)
./src/models.py         # Trained GNN models for complexity prediction
best_model.pt           # Pre-trained model with learned embeddings
```

### New Autoencoder Architecture ✨
```python
# Complete AST reconstruction pipeline
from src.models import ASTAutoencoder
from src.loss import ast_reconstruction_loss, ast_reconstruction_loss_simple

# Initialize autoencoder
autoencoder = ASTAutoencoder(
    encoder_input_dim=74,      # Ruby AST node feature dimension
    node_output_dim=74,        # Reconstructed node features
    hidden_dim=64,             # Embedding dimension
    freeze_encoder=True,       # Freeze pre-trained weights
    encoder_weights_path="best_model.pt"
)

# Forward pass: AST → embedding → reconstructed AST
result = autoencoder(ast_data)
embedding = result['embedding']           # (batch_size, 64)
reconstruction = result['reconstruction'] # Full AST structure

# Compute reconstruction loss for training
loss = ast_reconstruction_loss_simple(ast_data, reconstruction)
# Or use the full loss with edge prediction:
# loss = ast_reconstruction_loss(ast_data, reconstruction, node_weight=1.0, edge_weight=0.5)
```

### Development Setup
```bash
# Ruby dependencies
bundle install

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify setup
python test_dataset.py
python test_gnn_models.py
python test_autoencoder.py
python test_loss.py           # Test new loss functions
```

### AST Reconstruction Loss Functions ⚡

The project now includes specialized loss functions for training the AST autoencoder:

```python
from src.loss import ast_reconstruction_loss, ast_reconstruction_loss_simple

# Simple node-type prediction loss (recommended)
loss = ast_reconstruction_loss_simple(original_ast, reconstructed_ast)

# Full loss with node + edge prediction
loss = ast_reconstruction_loss(
    original_ast, 
    reconstructed_ast,
    node_weight=1.0,    # Weight for node type loss
    edge_weight=0.5     # Weight for edge prediction loss
)
```

**Loss Components:**
- **Node Type Loss**: Cross-entropy loss for predicting correct AST node types from 74-dimensional one-hot encoded features
- **Edge Prediction Loss**: Simplified loss for graph connectivity (compares edge counts and structure)
- **Batch Support**: Handles variable-sized graphs in batched training
- **Gradient Flow**: Optimized for training decoder parameters

**Demo Usage:**
```bash
python demo_loss.py          # Interactive demonstration
```

## Project Structure

```
jubilant-palm-tree/
├── README_phase1.md          # Complete Phase 1 documentation
├── dataset/                  # ML-ready Ruby method dataset
├── src/                      # GNN models and training code
│   ├── models.py            # ASTAutoencoder, RubyComplexityGNN, ASTDecoder
│   ├── loss.py              # AST reconstruction loss functions
│   └── data_processing.py   # Dataset loading and preprocessing
├── scripts/                  # Data extraction pipeline
├── notebooks/                # Analysis and visualization
└── requirements.txt          # Python dependencies
```

## Next Steps

1. **Decoder Architecture Design**: Design GNN decoder that maps embeddings to AST structure
2. **Training Pipeline**: Implement training loop for embedding → AST reconstruction
3. **Validation Metrics**: Develop metrics for AST reconstruction accuracy
4. **Code Generation**: Validate that reconstructed ASTs produce syntactically correct Ruby code

## Documentation

- **Phase 1 (Complete)**: See [README_phase1.md](README_phase1.md) for comprehensive documentation of data pipeline, GNN training, and complexity prediction results
- **Current Development**: This README tracks Phase 2 progress

---

*Building on the successful validation that GNNs can learn meaningful code structure representations, we now explore their generative potential for automated code synthesis.*