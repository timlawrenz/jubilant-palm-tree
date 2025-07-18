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

🎯 **GNN-based AST Decoder Development**
- Build and train a generative GNN model that can reconstruct Ruby method ASTs
- Validate that learned embeddings contain sufficient structural information for code generation
- Develop decoder architecture that maps embeddings → syntactically correct Ruby ASTs
- Establish foundation for advanced code generation capabilities

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
```

## Project Structure

```
jubilant-palm-tree/
├── README_phase1.md          # Complete Phase 1 documentation
├── dataset/                  # ML-ready Ruby method dataset
├── src/                      # GNN models and training code
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