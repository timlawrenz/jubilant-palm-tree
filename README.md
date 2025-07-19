# jubilant-palm-tree

## Overview

This project explores the potential of Graph Neural Networks (GNNs) to understand and generate Ruby code through Abstract Syntax Tree (AST) analysis. The project demonstrates that neural networks can learn meaningful structural representations of code complexity and successfully reconstruct AST structures from learned embeddings.

## Project Results Summary

✅ **Complete Success**: Successfully demonstrated that GNNs can both predict Ruby code complexity and reconstruct AST structures from learned embeddings.

### Key Achievements
- **Superior Performance**: GNN model achieved MAE of 4.27 vs heuristic baseline of 4.46 (4.3% improvement)
- **Perfect Reconstruction**: 100% structural preservation in AST reconstruction across all test samples
- **Meaningful Embeddings**: 64-dimensional representations cluster methods by complexity and enable full code reconstruction
- **Comprehensive Dataset**: 1,896 Ruby methods from 8 high-quality open-source projects
- **Complete Pipeline**: End-to-end system from Ruby source code to embeddings to reconstructed code

## Project Phases

This project was completed in 4 phases, with a 5th phase planned for future work:

### [Phase 1 - Data Generation & Preprocessing](README_phase1.md) ✅ **COMPLETED**
**Goal**: To produce a clean, structured dataset from raw source code, ready for model training.
- [Source Code Aggregation](https://github.com/timlawrenz/jubilant-palm-tree/issues/1)
- [Method Extraction](https://github.com/timlawrenz/jubilant-palm-tree/issues/2)
- [Feature & Label Generation](https://github.com/timlawrenz/jubilant-palm-tree/issues/3)
- [Dataset Assembly & Cleaning](https://github.com/timlawrenz/jubilant-palm-tree/issues/4)

### [Phase 2 - Model Setup & Training](README_phase2.md) ✅ **COMPLETED**
**Goal**: To build, train, and benchmark the GNN model for complexity prediction.
- [Python Environment Setup](https://github.com/timlawrenz/jubilant-palm-tree/issues/5)
- [Data Ingestion & Graph Conversion](https://github.com/timlawrenz/jubilant-palm-tree/issues/9)
- [GNN Model Definition](https://github.com/timlawrenz/jubilant-palm-tree/issues/10)
- [Training & Validation Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/11)
- [Heuristic Benchmark Implementation](https://github.com/timlawrenz/jubilant-palm-tree/issues/12)

### [Phase 3 - Evaluation & Analysis](README_phase3.md) ✅ **COMPLETED**
**Goal**: To evaluate the trained model's performance and analyze its learned representations.
- [Model Evaluation Script](https://github.com/timlawrenz/jubilant-palm-tree/issues/22)
- [Embedding Visualization](https://github.com/timlawrenz/jubilant-palm-tree/issues/23)
- [Final Report Generation](https://github.com/timlawrenz/jubilant-palm-tree/issues/24)

### [Phase 4 - AST Autoencoder for Code Generation](README_phase4.md) ✅ **COMPLETED**
**Goal**: To build and train a GNN-based decoder that can reconstruct a Ruby method's AST from its learned embedding, validating the generative potential of the embeddings.
- [Autoencoder Model Definition](https://github.com/timlawrenz/jubilant-palm-tree/issues/34)
- [AST Reconstruction Loss Function](https://github.com/timlawrenz/jubilant-palm-tree/issues/35)
- [Autoencoder Training Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/36)
- [Evaluation with Pretty-Printing](https://github.com/timlawrenz/jubilant-palm-tree/issues/37)
- [And 8 additional issues for robust implementation and evaluation](README_phase4.md)

### [Phase 5 - Aligning Text and Code Embeddings](README_phase5.md) 📋 **PLANNED**
**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.

## Quick Start

### Prerequisites
- Ruby 2.7+ and Python 3.8+
- PyTorch and PyTorch Geometric for GNN training
- See individual phase READMEs for detailed setup instructions

### Key Components
```bash
# Dataset and models
dataset/                  # 1,896 processed Ruby methods (train/val/test splits)  
src/models.py            # GNN models and autoencoder architecture
best_model.pt            # Pre-trained complexity prediction model
best_decoder.pt          # Trained AST reconstruction decoder

# Training and evaluation
train.py                 # GNN complexity prediction training
train_autoencoder.py     # AST autoencoder training
evaluate_autoencoder_optimized.py  # Large-scale evaluation

# Code generation tools
scripts/pretty_print_ast.rb  # Convert AST JSON to Ruby code
notebooks/evaluate_autoencoder.ipynb  # Interactive evaluation
```

### Quick Demo
```python
# Load trained autoencoder for AST reconstruction
from src.models import ASTAutoencoder

autoencoder = ASTAutoencoder(
    encoder_input_dim=74,
    node_output_dim=74,
    hidden_dim=64,
    freeze_encoder=True,
    encoder_weights_path="best_model.pt"
)

# Complete pipeline: AST → embedding → reconstructed AST
result = autoencoder(ast_data)
embedding = result['embedding']           # 64-dimensional representation
reconstruction = result['reconstruction'] # Reconstructed AST
```

## Project Results

### Complexity Prediction (Phases 1-3)
- **GNN Model Performance**: MAE of 4.27 vs baseline of 4.46 (4.3% improvement)
- **Embedding Quality**: 64-dimensional representations cluster methods by complexity
- **Dataset Scale**: 1,896 Ruby methods from 8 open-source projects
- **Training Stability**: 100 epochs with robust convergence

### AST Reconstruction (Phase 4)
- **Perfect Preservation**: 100% structural fidelity across all test samples
- **Scalable Evaluation**: Tested from 25 to 1,000+ samples consistently
- **Code Generation**: Complete Ruby source → AST → embedding → AST → Ruby pipeline
- **Model Architecture**: 47,692 parameters (21,579 trainable decoder + 26,113 frozen encoder)

## Development Setup

```bash
# Ruby dependencies for AST processing
gem install --user-install parser json

# Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify installation
python test_dataset.py
python test_autoencoder.py
ruby scripts/pretty_print_ast.rb --help
```

## Project Structure

```
jubilant-palm-tree/
├── README_phase1.md          # Phase 1: Data Generation & Preprocessing
├── README_phase2.md          # Phase 2: Model Setup & Training  
├── README_phase3.md          # Phase 3: Evaluation & Analysis
├── README_phase4.md          # Phase 4: AST Autoencoder for Code Generation
├── README_phase5.md          # Phase 5: Text and Code Embeddings (planned)
├── dataset/                  # ML-ready Ruby method dataset
├── src/                      # GNN models and training code
├── scripts/                  # Data processing and AST conversion tools
├── notebooks/                # Analysis and evaluation notebooks
├── train.py                  # GNN complexity prediction training
└── train_autoencoder.py      # AST autoencoder training
```

---

*This project successfully demonstrates that Graph Neural Networks can learn meaningful structural representations of Ruby code, enabling both complexity prediction and complete AST reconstruction. For detailed information about each phase, see the individual phase README files.*