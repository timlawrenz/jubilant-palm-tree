# Project Discontinuation Notice

**Date**: November 26, 2025  
**Project**: jubilant-palm-tree (GNN-based Ruby Code Generation)  
**Status**: DISCONTINUED

## Summary

This research project has been discontinued after comprehensive analysis revealed that Graph Neural Networks (GNNs) are fundamentally unsuitable for code generation tasks, despite their success in code understanding.

## What We Learned

### Successful Components ✅
- **Code Complexity Prediction**: 26.6% improvement over heuristics (MAE 4.77)
- **Large-scale AST Processing**: Successfully handled 218,000+ Ruby methods
- **Structural Understanding**: GNNs effectively learn code patterns

### Failed Components ❌
- **Hierarchical AST Decoder**: 0% syntactic validity after 100 training epochs
- **Text-Code Alignment**: Near-random performance (2-3% Recall@10)
- **Code Generation Pipeline**: Unable to generate any valid code

## Root Cause

**Code generation is a sequence modeling problem, not a graph problem.**

While ASTs have graph structure, generating code requires:
- Sequential dependencies (autoregressive generation)
- Temporal context (attention mechanisms)
- Language modeling (token-by-token prediction)

GNNs excel at reasoning over fixed graphs but cannot:
- Generate sequential structures
- Maintain semantic coherence across generation steps
- Learn syntax rules through graph convolutions

## Why We're Sharing This

**Negative results are valuable.** By documenting this failure comprehensively, we hope to:
- Save other researchers from repeating this mistake
- Demonstrate the importance of architecture selection
- Contribute to understanding of GNN limitations
- Provide a cautionary tale with complete data

## What's Available

All experimental materials released under CC0 (Public Domain):

- ✅ Complete failure analysis documentation
- ✅ All source code and training scripts
- ✅ Trained models (850K parameters across 20 levels)
- ✅ Training logs (9.7MB, 100 epochs)
- ✅ Validation results and visualizations
- ✅ Full dataset (218K+ methods)

## For Future Researchers

**If you're working on code generation:**

Do use:
- Transformer-based models (GPT, CodeT5, StarCoder)
- Pre-trained embeddings (CodeBERT, GraphCodeBERT)
- Autoregressive token prediction
- Cross-entropy loss

Don't use:
- GNNs for generation (use them for understanding/analysis)
- Hierarchical graph decoders
- MSE loss on discrete features
- Small custom embeddings without pre-training

## Documentation

- **Main README**: [README.md](README.md) - Updated with findings
- **Failure Analysis**: [docs/HIERARCHICAL_FAILURE_ANALYSIS.md](docs/HIERARCHICAL_FAILURE_ANALYSIS.md)
- **Research Data**: [docs/RESEARCH_DATA_AVAILABILITY.md](docs/RESEARCH_DATA_AVAILABILITY.md)
- **Citation**: [CITATION.cff](CITATION.cff)

## License

All materials released under [CC0 1.0 Universal](LICENSE) (Public Domain).  
No rights reserved. Use freely for any purpose.

## Contact

This is a discontinued project. No further development is planned.  
The repository remains available as an educational resource.

---

*"Science is built up of facts, as a house is built of stones; but an accumulation of facts is no more a science than a heap of stones is a house." - Henri Poincaré*

*This project accumulated facts about what doesn't work. That's science too.*
