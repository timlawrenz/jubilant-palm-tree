# Research Data Availability

## Purpose

This repository preserves a complete failed experiment in using Graph Neural Networks for code generation. All materials are released to the public domain (CC0 license) so other researchers can learn from this failure without repeating it.

## What's Available

### Documentation
- **[HIERARCHICAL_FAILURE_ANALYSIS.md](HIERARCHICAL_FAILURE_ANALYSIS.md)**: Complete analysis of why the approach failed
- **Phase READMEs**: Detailed documentation for each experimental phase
- **Training logs**: Full training history (9.7MB, 100 epochs)
- **Validation results**: Complete evaluation metrics

### Code & Models
- **Source code**: All training, evaluation, and data processing code
- **Trained models**: 20 hierarchical decoder levels (~850K parameters)
- **Training scripts**: Reproducible training pipelines
- **Evaluation scripts**: Validation and analysis tools

### Data
- **Dataset**: 218,000+ Ruby methods from 42 open-source projects
- **AST representations**: Hierarchical tree structures at 20 depth levels
- **Embeddings**: Pre-computed code and text embeddings

### Visualizations
- **[hierarchical_training_analysis.png](hierarchical_training_analysis.png)**: Loss progression showing model did train
- **[hierarchical_failure_analysis.png](hierarchical_failure_analysis.png)**: 6-panel diagnostic visualization

## Key Findings for Researchers

### ✅ Use These Approaches for Code Generation
- Transformer-based autoregressive models (GPT-style)
- Pre-trained embeddings (CodeBERT, CodeT5)
- Cross-entropy loss on tokens
- Teacher forcing during training

### ❌ Don't Use These Approaches
- Graph Neural Networks for generation (good for understanding, bad for generation)
- Hierarchical independence (breaks semantic coherence)
- MSE loss on discrete features
- Small custom embeddings (64D insufficient)

## How to Use This Data

### For Academic Papers
Cite this as a negative result demonstrating architectural limitations:

```bibtex
@dataset{lawrenz2025gnn_failure,
  title = {Graph Neural Networks for Ruby Code Generation: A Failed Experiment},
  author = {Lawrenz, Tim},
  year = {2025},
  license = {CC0-1.0},
  url = {https://github.com/timlawrenz/jubilant-palm-tree},
  note = {Complete experimental record showing GNNs fail at code generation (0\% validity) despite succeeding at code understanding (26.6\% improvement)}
}
```

### For Course Material
Use as case study in:
- ML architecture selection
- Negative results in research
- Code generation approaches
- Graph vs sequence modeling

### For Replication Studies
All code and data are available to:
- Verify our findings
- Try alternative approaches
- Compare with transformer baselines
- Extend to other languages

## License: CC0 1.0 Universal (Public Domain)

**No rights reserved.** You can:
- ✅ Use for any purpose (commercial or academic)
- ✅ Modify without attribution
- ✅ Redistribute freely
- ✅ Build upon this work

See [LICENSE](../LICENSE) for full legal text.

## Quick Navigation

- **Start here**: [HIERARCHICAL_FAILURE_ANALYSIS.md](HIERARCHICAL_FAILURE_ANALYSIS.md)
- **Main README**: [../README.md](../README.md)
- **Phase documentation**: Individual README_phase*.md files
- **Training code**: `../train_hierarchical.py`
- **Models**: `../models/hierarchical/`

## Questions?

This is a discontinued project, but the complete record is preserved for educational purposes. All findings are documented in the analysis files.

---

**Note**: This is a negative result. The experiment failed, and we're sharing why so others don't repeat the same mistakes. That's valuable scientific contribution.
