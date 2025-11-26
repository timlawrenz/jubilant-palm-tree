# Documentation Directory

This directory contains comprehensive documentation for the jubilant-palm-tree research project.

## 🔴 Project Status: DISCONTINUED

This project has been discontinued after demonstrating that graph-based approaches fundamentally fail at code generation. All findings are preserved for the research community.

## Key Documents

### Failure Analysis (Start Here)
- **[HIERARCHICAL_FAILURE_ANALYSIS.md](HIERARCHICAL_FAILURE_ANALYSIS.md)**: Complete analysis of why GNNs fail at code generation
- **[RESEARCH_DATA_AVAILABILITY.md](RESEARCH_DATA_AVAILABILITY.md)**: How to use this data for your research

### Phase Documentation
1. **[README_phase1.md](README_phase1.md)** - Data Generation & Preprocessing ✅
2. **[README_phase2.md](README_phase2.md)** - Model Setup & Training ✅
3. **[README_phase3.md](README_phase3.md)** - Evaluation & Analysis ✅
4. **[README_phase4.md](README_phase4.md)** - AST Autoencoder (Failed) ❌
5. **[README_phase4b.md](README_phase4b.md)** - Hierarchical Decoder (Failed) ❌
6. **[README_phase5.md](README_phase5.md)** - Text-Code Alignment (Failed) ❌
7. **[README_phase6.md](README_phase6.md)** - Text-to-Code Generation (Failed) ❌
8. **[README_phase7.md](README_phase7.md)** - Autoregressive Decoder (Not Implemented)

### Technical Guides
- **[CPU_OPTIMIZATION_SUMMARY.md](CPU_OPTIMIZATION_SUMMARY.md)** - Performance optimization notes
- **[MEMORY_OPTIMIZATION_GUIDE.md](MEMORY_OPTIMIZATION_GUIDE.md)** - Memory management for training
- **[README_assessment.md](README_assessment.md)** - Assessment methodology

### Visualizations
- **hierarchical_training_analysis.png** - Loss progression over 100 epochs
- **hierarchical_failure_analysis.png** - 6-panel diagnostic visualization

## Key Findings

**What Worked**: GNN-based code complexity prediction (26.6% improvement)  
**What Failed**: All generative models (0% syntactic validity)  
**Why**: Graph-based approaches incompatible with sequential code generation

See [HIERARCHICAL_FAILURE_ANALYSIS.md](HIERARCHICAL_FAILURE_ANALYSIS.md) for complete details.

## License

All materials in this directory are released under CC0 1.0 Universal (Public Domain).  
See [../LICENSE](../LICENSE) for details.
