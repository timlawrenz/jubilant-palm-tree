# Validation Results - 2025-11-20

## Phase 4b: Hierarchical AST Decoder (COMPLETE TRAINING)

**Date**: 2025-11-20 13:10 UTC  
**Training Status**: 100% complete (20/20 levels)  
**Training Duration**: ~8 hours (2025-11-19 22:47 - 2025-11-20 07:03 UTC)

---

## Summary

The Hierarchical AST Decoder has been fully trained across all 20 levels, but **completely fails to generate valid code**. Despite achieving convergence during training (final loss: -70.2750), the model produces syntactically invalid and semantically nonsensical output.

---

## Quantitative Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Syntactic Validity** | 0.0% | >80% | ❌ **FAILED** |
| **AST Isomorphism** | 0.0% | >50% | ❌ **FAILED** |
| **Normalized BLEU** | 0.00 | >0.30 | ❌ **FAILED** |
| **Standard BLEU** | 0.00 | >0.20 | ❌ **FAILED** |

**Evaluation Details:**
- Samples Evaluated: 100
- Model Path: `models/hierarchical/`
- Levels Loaded: 20/20 ✓

---

## Qualitative Analysis

### Example Generation

**Prompt**: "a function that adds two numbers"

**Generated AST Pattern**:
```
def
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
├── casgn (10x range children each)
└── casgn (no children)
```

**Observations**:
1. **Structural plausibility**: Root node is `def` (method definition) - semantically appropriate
2. **Repetitive patterns**: Generates exactly 10 `casgn` children, each with 10 `range` children
3. **No semantic meaning**: `casgn` (constant assignment) and `range` are unrelated to "adding two numbers"
4. **Generates maximum nodes**: Always produces max allowed nodes per level (10)
5. **No variation**: Same pattern regardless of input prompt

### Problem Diagnosis

**Why the model fails:**

1. **Spawn probability is broken**: Model always generates max children (10) per node
   - The diagonal adjacency predictor returns high values for all nodes
   - No learned sparsity in tree structure

2. **Node type prediction is stuck**: 
   - Level 1 always predicts `casgn` (constant assignment)
   - Level 2 always predicts `range`
   - No diversity or context-awareness

3. **Text embedding is ignored**:
   - Same output regardless of input prompt
   - The text-to-code alignment is not working
   - Embeddings may not contain enough semantic information

4. **Training loss is misleading**:
   - Final loss: -70.2750 (looks like convergence)
   - But the model hasn't learned meaningful generation
   - Loss may be dominated by other terms (adjacency, features) rather than validity

---

## Architecture Issues

### 1. **Hierarchical GNN Approach**
- Each level uses GCN/SAGE convolutions
- But there's no graph structure at level 0 (single node)
- GNN may not be appropriate for tree generation from single embedding

### 2. **Level-by-Level Independence**
- Each level's generator is trained independently
- No end-to-end gradient flow through the full generation process
- Each level optimizes its own loss, not final code validity

### 3. **Text Embedding Bottleneck**
- Single 64-dim embedding must encode all semantic information
- No attention mechanism to focus on different aspects
- No recurrent state to maintain context across levels

### 4. **Missing Components**
- No copy mechanism to reuse tokens
- No pointer network to handle references
- No grammar constraints to enforce valid structures
- No teacher forcing during generation

---

## Training Analysis

### Model Weights Check

Sample of model parameters across levels:

| Level | Parameters | Mean | Std | Range |
|-------|------------|------|-----|-------|
| 0 | 34,378 | -0.0005 | 0.1165 | [-0.49, 0.48] |
| 5 | 42,570 | 0.0008 | 0.0679 | [-0.16, 0.17] |
| 10 | 42,570 | 0.0000 | 0.0681 | [-0.17, 0.18] |
| 15 | 42,570 | 0.0004 | 0.0681 | [-0.17, 0.18] |
| 19 | 42,570 | 0.0005 | 0.0679 | [-0.17, 0.18] |

**Observations**:
- Weights are not all zeros (model was trained)
- Standard deviation is reasonable (~0.07-0.12)
- Higher levels have similar statistics → may indicate later levels aren't learning much new

---

## Comparison with Previous Results

### Phase 4b Progress

| Training % | Syntactic Validity | Notes |
|------------|-------------------|-------|
| 25% (5/20 levels) | 0% | Nov 19, partial training |
| 100% (20/20 levels) | 0% | Nov 20, **NO IMPROVEMENT** |

**Conclusion**: More training levels did not help. The architecture itself is the problem.

---

## Lessons Learned

1. **Graph-based generation is hard**: Hierarchical GNN approach may not be suitable for code generation from text
2. **Independent level training doesn't work**: Need end-to-end training for the full generation pipeline
3. **Text embeddings alone are insufficient**: Need richer representations or attention mechanisms
4. **Training loss ≠ generation quality**: Model can minimize loss while producing garbage

---

## Recommended Actions

### 🚨 **IMMEDIATE PIVOT REQUIRED**

The hierarchical GNN approach has **fundamentally failed** after full training. Recommended next steps:

### Option A: Autoregressive Transformer (RECOMMENDED)
- **Why**: Proven architecture for code generation (GPT, CodeT5, Codex)
- **Approach**: Sequence-to-sequence transformer with teacher forcing
- **Implementation**: Can use existing frameworks (Hugging Face Transformers)
- **Timeline**: 2-3 days to implement and train
- **Success probability**: High (proven approach)

### Option B: Simplify Problem First
- **Why**: Validate approach can work on easier tasks
- **Approach**: 
  - Only generate simple expressions (no functions)
  - Limit to 10 most common node types
  - Depth-1 trees only
- **Timeline**: 1-2 days
- **Success probability**: Medium (may reveal fixable issues)

### Option C: Use Pre-trained Models
- **Why**: Fastest path to working solution
- **Approach**: Fine-tune CodeT5/CodeGen on Ruby
- **Timeline**: 1 day to setup and run
- **Success probability**: Very High (proven models)

### Option D: Fix Architecture (NOT RECOMMENDED)
- **Why**: Uncertain payoff, high effort
- **Issues to fix**:
  - Add attention mechanisms
  - Implement teacher forcing
  - Add grammar constraints
  - Use pre-trained code embeddings
  - End-to-end training
- **Timeline**: 1-2 weeks
- **Success probability**: Low-Medium

---

## Files Generated

- `models/hierarchical/hierarchical_decoder_level_[0-19].pt` - 20 trained models
- `models/hierarchical/training_checkpoint.pt` - Final checkpoint
- `validation_results_hierarchical.txt` - Detailed validation output
- `generate_hierarchical.py` - Fixed generation script (was crashing, now works but produces nonsense)

---

## Next Session Checklist

- [ ] Decide on pivot strategy (Option A recommended)
- [ ] Update PROJECT_STATUS.md with decision
- [ ] If pivoting to autoregressive: Review `docs/README_phase7.md`
- [ ] If trying to fix: Create detailed architecture redesign plan
- [ ] If simplifying: Create reduced dataset with simple expressions only

---

**Status**: Phase 4b is complete but non-functional. Architecture fundamentally unsuitable for this task. **Pivot required.**
