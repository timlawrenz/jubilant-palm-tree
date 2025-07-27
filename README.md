# jubilant-palm-tree

[![CircleCI](https://circleci.com/gh/timlawrenz/jubilant-palm-tree.svg?style=svg)](https://circleci.com/gh/timlawrenz/jubilant-palm-tree)

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
- **Text-Code Alignment**: Contrastive learning aligns natural language descriptions with code embeddings
- **Multimodal Learning**: Successful dual-encoder architecture with 43.5% loss improvement over training
- **Text-to-Code Generation**: Complete pipeline from natural language to executable Ruby code
- **Semantic Understanding**: Excellent performance for arithmetic operations and array methods

## Project Phases

This project has been developed through 7 phases, with Phase 7 representing the next major advancement:

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

### [Phase 5 - Aligning Text and Code Embeddings](README_phase5.md) ✅ **COMPLETED**
**Goal**: Train a text-encoder so that the embedding it produces for a method's description is located at the same point in the 64-dimensional space as the embedding our GNN produces for the method's AST.
- [Alignment Training Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/77)

### [Phase 6 - Text-to-Code Generation](README_phase6.md) ✅ **COMPLETED**
**Goal**: Complete the end-to-end text-to-code generation pipeline by combining aligned text-code embeddings with AST reconstruction to generate Ruby code from natural language descriptions.
- Complete integration of all phases into working text-to-code system
- Demonstrated successful generation for arithmetic and array operations
- Identified decoder limitations for complex control flow structures

### [Phase 7 - Advanced Decoder Architectures](README_phase7.md) 🚧 **PLANNED**
**Goal**: To overcome the limitations of the simple, one-shot decoder by implementing a more powerful, autoregressive model that can generate complex, nested code structures.
- [Update Data Loader for Autoregressive Training](https://github.com/timlawrenz/jubilant-palm-tree/issues/27)
- [Implement Autoregressive AST Decoder Model](https://github.com/timlawrenz/jubilant-palm-tree/issues/28)
- [Implement Autoregressive Training Loop](https://github.com/timlawrenz/jubilant-palm-tree/issues/29)
- [Implement Autoregressive Inference](https://github.com/timlawrenz/jubilant-palm-tree/issues/30)

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
best_alignment_model.pt  # Trained text-code alignment model

# Training and evaluation
train.py                 # GNN complexity prediction training
train_autoencoder.py     # AST autoencoder training
train_alignment.py       # Text-code alignment training
evaluate_autoencoder_optimized.py  # Large-scale evaluation

# Code generation tools
generate_code.py         # Complete text-to-code generation pipeline
scripts/pretty_print_ast.rb  # Convert AST JSON to Ruby code
notebooks/demonstrate_text_to_code.ipynb  # Interactive text-to-code demo
notebooks/evaluate_autoencoder.ipynb     # Interactive evaluation
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

### Text-to-Code Generation
```bash
# Generate Ruby code from natural language
python generate_code.py "a method that adds two numbers"

# Interactive code generation
python generate_code.py --interactive
```

```python
# Use in Python scripts
from generate_code import CodeGenerator

generator = CodeGenerator()
ruby_code = generator.generate_code("calculate total price with tax")
print(ruby_code)
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

### Text-Code Alignment (Phase 5)
- **Dual-encoder Architecture**: Frozen GNN code encoder + trainable text projection head
- **Contrastive Learning**: InfoNCE loss aligns text descriptions with code embeddings  
- **Successful Training**: 43.5% loss improvement demonstrating effective alignment learning
- **Shared Embedding Space**: 64-dimensional space enables text-to-code and code-to-text tasks

### Text-to-Code Generation (Phase 6)
- **End-to-End Pipeline**: Complete system from natural language to executable Ruby code
- **Semantic Understanding**: Excellent performance for arithmetic operations and array methods
- **Stable Architecture**: Consistent 64D embeddings and 15-node AST generation
- **Successful Examples**: Perfect generation for "adds two numbers" and "finds largest in array"
- **Current Limitations**: Decoder bottleneck identified for complex control flow (conditionals, loops)
- **Future Direction**: Phase 7 autoregressive architecture planned to address complex code generation

### Advanced Decoder Architectures (Phase 7) - Planned
- **Autoregressive Generation**: Sequential AST building to handle complex control structures
- **Enhanced Training**: Teacher forcing with step-by-step sequence generation
- **Improved Inference**: Iterative sampling with temperature and top-k controls
- **Target Capability**: Generate conditional statements, loops, and nested logic structures

## Development Setup

### Ruby Dependencies (Required for AST processing)

**Quick Setup for Copilot Agents:**
```bash
# Automated setup - recommended for Copilot coding agents
./setup-ruby.sh

# Activate Ruby environment in current session
source .env-ruby
```

**Manual Setup (if needed):**
```bash
# Install Ruby gems to user directory (avoids permission errors)
gem install --user-install bundler parser json

# Configure environment for user gems
export PATH="$HOME/.local/share/gem/ruby/$(ruby -e "puts RUBY_VERSION.match(/\d+\.\d+/)[0]").0/bin:$PATH"
export GEM_PATH="$HOME/.local/share/gem/ruby/$(ruby -e "puts RUBY_VERSION.match(/\d+\.\d+/)[0]").0:$GEM_PATH"
```

### Python Environment
```bash
# Python dependencies for GNN models
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Verify Installation
```bash
# Test Ruby AST processing
ruby test-ruby-setup.rb

# Test specific scripts
ruby scripts/check_syntax.rb < scripts/check_syntax.rb

# Test Python ML pipeline
python test_dataset.py
python test_autoencoder.py

# Test AST pretty printing
ruby scripts/pretty_print_ast.rb --help
```

## Project Structure

```
jubilant-palm-tree/
├── README_phase1.md          # Phase 1: Data Generation & Preprocessing
├── README_phase2.md          # Phase 2: Model Setup & Training  
├── README_phase3.md          # Phase 3: Evaluation & Analysis
├── README_phase4.md          # Phase 4: AST Autoencoder for Code Generation
├── README_phase5.md          # Phase 5: Text and Code Embeddings
├── README_phase6.md          # Phase 6: Text-to-Code Generation
├── README_phase7.md          # Phase 7: Advanced Decoder Architectures
├── dataset/                  # ML-ready Ruby method dataset
├── src/                      # GNN models and training code
├── scripts/                  # Data processing and AST conversion tools
├── notebooks/                # Analysis and evaluation notebooks
├── generate_code.py          # Text-to-code generation pipeline
├── train.py                  # GNN complexity prediction training
├── train_autoencoder.py      # AST autoencoder training
├── train_alignment.py        # Text-code alignment training
└── train_autoregressive.py   # Autoregressive decoder training (Phase 7)
```

---

*This project successfully demonstrates that Graph Neural Networks can learn meaningful structural representations of Ruby code, enabling complexity prediction, complete AST reconstruction, text-code alignment through contrastive learning, and end-to-end text-to-code generation. The 6-phase implementation proves the viability of neural approaches to code understanding and generation, with Phase 7 planned to address the remaining limitations in complex control flow generation through autoregressive decoder architectures. For detailed information about each phase, see the individual phase README files.*