# Fresh Validation Results - November 19, 2025

This document contains the results of a comprehensive validation of all models in the jubilant-palm-tree project.

## Executive Summary

**✅ Phases 1-3 (Complexity Prediction): FUNCTIONAL**
- The GNN complexity prediction model works and significantly outperforms baselines

**❌ Phase 4b (Hierarchical AST Decoder): NON-FUNCTIONAL**
- 5 levels trained but 0% syntactic validity on all metrics

**❌ Phase 5 (Text-Code Alignment): NON-FUNCTIONAL**  
- Retrieval performance at near-random levels (0.3-0.4% Recall@10)

**❌ Phase 6 (Text-to-Code Generation): NON-FUNCTIONAL**
- Generates trivial/meaningless code regardless of prompt

---

## Phase 1-3: GNN Complexity Prediction Model ✅

**Model**: `models/best_model.pt`  
**Test Dataset**: 22,453 Ruby methods  
**Status**: **FUNCTIONAL**

### Performance Metrics

| Metric | Value |
|--------|-------|
| **GNN MAE** | **4.64** |
| **Simple Heuristic MAE** | 8.69 |
| **Improvement** | **46.6% better than heuristic** |

### Notes
- Previous assessment showed MAE of 4.77 vs heuristic baseline of 6.50 (26.6% improvement)
- Current test shows even better performance: 4.64 vs 8.69 (46.6% improvement)
- The model successfully learns structural representations of Ruby code complexity
- **Conclusion**: Core foundational model is solid and working as intended

---

## Phase 4b: Hierarchical AST Decoder ❌

**Model Directory**: `models/hierarchical/`  
**Levels Trained**: 5 (level_0 through level_4)  
**Test Samples**: 100 Ruby methods  
**Status**: **NON-FUNCTIONAL**

### Performance Metrics

| Metric | Score |
|--------|-------|
| **Syntactic Validity (%)** | **0.0** |
| **AST Isomorphism (%)** | **0.0** |
| **Normalized BLEU Score** | **0.0** |
| **Standard BLEU Score** | **0.0** |

### Analysis
- The hierarchical approach was intended to replace the failed "one-shot" autoencoder from Phase 4
- 5 levels have been trained (models exist for levels 0-4)
- Despite the hierarchical/coarse-to-fine approach, the model still produces **zero valid ASTs**
- The model appears to generate trivial single-node ASTs that don't represent valid Ruby code
- Training may have stopped at level 4 (only 5 levels vs. planned 20+ levels)

### Potential Issues
1. **Insufficient training depth**: Only 5 levels trained, many Ruby methods need deeper trees
2. **Training methodology**: The level-by-level training may not be learning proper structure
3. **Loss function**: May not be penalizing structural invalidity enough
4. **Data representation**: The AST pyramid dataset may not capture enough context

---

## Phase 5: Text-Code Alignment Model ❌

**Model**: `models/best_alignment_model.pt`  
**Test Dataset**: 28,122 paired (text, code) samples  
**Status**: **NON-FUNCTIONAL**

### Performance Metrics

#### Text-to-Code Retrieval
| Metric | Score |
|--------|-------|
| **Recall@1** | **0.03%** |
| **Recall@5** | **0.16%** |
| **Recall@10** | **0.34%** |

#### Code-to-Text Retrieval
| Metric | Score |
|--------|-------|
| **Recall@1** | **0.04%** |
| **Recall@5** | **0.23%** |
| **Recall@10** | **0.42%** |

### Analysis
- Performance is **near random chance** (random baseline would be ~0.004% Recall@1, 0.04% Recall@10)
- The model is only marginally better than random selection
- Previous validation showed 2.22% and 2.96% for Recall@10; current results are **even worse**
- The contrastive loss training appears to have failed to align the embedding spaces

### Potential Issues
1. **Frozen code encoder**: Using frozen GNN may be too constraining
2. **Text descriptions**: Method names and docstrings may not provide enough semantic signal
3. **Projection head**: Simple linear projection may be insufficient to bridge the modalities
4. **Training data quality**: Paired descriptions may not be semantically aligned with code
5. **Embedding dimension**: 64-dim may be too small for meaningful text-code alignment

---

## Phase 6: Text-to-Code Generation Pipeline ❌

**Models Used**:
- Alignment Model: `models/samples/best_alignment_model.pt`
- Autoregressive Decoder: `models/samples/best_autoregressive_decoder.pt`

**Test Prompt**: `"a method that adds two numbers"`  
**Status**: **NON-FUNCTIONAL**

### Generated Output

```ruby
def a_method_that_adds_t
nil
end
```

### Analysis
- The pipeline **fails to generate meaningful code**
- Output is syntactically incomplete (method name truncated)
- Body is trivial (`nil`) with no logic
- Same pattern observed previously: single-node AST generation
- The model appears to ignore the semantic content of the prompt entirely

### Root Causes
This failure is **expected** because it depends on two broken upstream components:
1. **Broken alignment model** (Phase 5): Cannot convert text to meaningful code embeddings
2. **Broken decoder** (Phase 4/4b): Cannot convert embeddings to valid ASTs

Even if one component worked, the other would prevent end-to-end success.

---

## Comparison with Previous Assessment

| Phase | Previous Status | Current Status | Change |
|-------|----------------|----------------|--------|
| 1-3 (Complexity) | ✅ MAE 4.77 | ✅ MAE 4.64 | ⬆️ Slightly improved |
| 4 (Autoencoder) | ❌ 0% validity | N/A (superseded) | - |
| 4b (Hierarchical) | 🚧 In progress | ❌ 0% validity | ⬇️ Trained but failed |
| 5 (Alignment) | ❌ 2-3% Recall@10 | ❌ 0.3-0.4% Recall@10 | ⬇️ Worse |
| 6 (Generation) | ❌ Trivial output | ❌ Trivial output | ➡️ No change |

---

## Recommendations

### Immediate Actions

1. **Update README.md**: Change status of Phases 4b, 5, and 6 from "COMPLETED" or "IN PROGRESS" to accurately reflect non-functional status

2. **Investigate Hierarchical Decoder**:
   - Why did training stop at level 4?
   - Review loss curves and training logs
   - Consider architectural changes (attention mechanisms, better node/edge prediction heads)

3. **Rethink Alignment Strategy**:
   - Current approach is not working
   - Consider unfreezing code encoder
   - Explore better text encoders or fine-tuning strategies
   - Investigate data quality issues in paired dataset

### Strategic Decisions

**Option A: Debug Current Approach**
- Incrementally fix each phase's issues
- High effort, uncertain payoff
- May require fundamental architectural changes

**Option B: Pivot to Simpler Targets**
- Focus on very simple code patterns first (single expressions, basic arithmetic)
- Validate the approach works at small scale before scaling
- Build confidence with incremental successes

**Option C: Leverage Pre-trained Models**
- Consider using existing code generation models (CodeT5, CodeGen, StarCoder)
- Fine-tune on Ruby code
- Focus on the complexity prediction aspect where this project excels

### Next Steps for Validation

1. **Detailed Loss Analysis**: Plot training curves for all models to identify where learning failed
2. **Manual Inspection**: Examine specific failure cases to understand what the models are learning
3. **Ablation Studies**: Test individual components in isolation
4. **Data Quality Check**: Verify the paired dataset contains meaningful semantic alignments

---

## Validation Artifacts

All validation results have been saved:
- `validation_results_complexity.txt` - Complexity model evaluation
- `validation_results_alignment.txt` - Alignment model evaluation  
- `validation_results_hierarchical.txt` - Hierarchical decoder evaluation
- `validation_results_generation.txt` - End-to-end generation test
- `VALIDATION_RESULTS_2025-11-19.md` - This summary document

---

## Conclusion

The fresh validation confirms the previous assessment findings:

- **Phase 1-3**: Working well, solid foundation
- **Phases 4-6**: All non-functional despite implementation efforts

The project has successfully demonstrated that GNNs can learn code complexity but has **not yet achieved functional code generation capabilities**. Significant architectural and methodological changes will be required to make the generative models work.
