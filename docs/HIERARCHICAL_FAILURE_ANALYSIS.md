# Hierarchical Decoder Failure Analysis

**Date**: 2025-11-26  
**Model**: Hierarchical AST Decoder (Phase 4b)  
**Status**: ❌ COMPLETE FAILURE - 0% Validity After Full Training

---

## Executive Summary

The hierarchical decoder approach has **fundamentally failed** despite successful training completion. After training all 20 levels (8 hours, 100 epochs total), the model achieves:
- **0% syntactic validity** (0/100 samples)
- **0% AST isomorphism** 
- **0.0 BLEU score**

**Root Cause**: Architecture mismatch between graph-based approach and sequential nature of code generation.

---

## Training Metrics Analysis

### ✅ Training DID Complete Successfully

**Loss Progression**:
- Initial loss (Level 0, Epoch 1): -3.46
- Final loss (Level 19, Epoch 5): -70.28
- Convergence pattern: Smooth decrease, plateaus at ~-70

**Loss by Level** (final epoch):
```
Level  0: -56.13  ← Starting point
Level  1: -72.46  ← Best loss achieved
Level  2: -67.53
...
Level 10: -70.27  ← Convergence
Level 19: -70.28  ← Stable
```

**Key Finding**: Loss decreased consistently, suggesting the model IS learning... but learning the WRONG thing.

### ✅ Weights Changed Significantly

Comparing Level 0 → Level 19:
- `gnn.bias`: 99.26% change
- `gnn.lin.weight`: ~85% change (estimate)
- `node_predictor`: ~70% change (estimate)
- No NaN or extreme values detected

**Conclusion**: Training worked mechanically. The problem is architectural.

---

## Validation Results

### What the Model Generates

From validation (100 samples):
- **Syntactically valid code**: 0 samples
- **Semantically meaningful ASTs**: 0 samples
- **Pattern observed**: Repetitive, nonsensical structures

Example output:
```ruby
def
  casgn
    range
    range
    range
    ... (10 times)
  casgn
    range
    range
    ... (10 times)
  ... (10 casgn nodes)
```

This structure is:
- ✅ Structurally valid (has parent-child relationships)
- ❌ Semantically nonsensical (casgn doesn't have range children)
- ❌ Ignores input text description
- ❌ Repetitive pattern (model collapsed to fixed output)

---

## Architecture Analysis

### Model Structure (Per Level)

```
Input: Text embedding (64D)
  ↓
GCNConv Layer (64→128 or 128→128)
  ↓
Node Predictor: Linear(128→74)  ← No softmax!
Adjacency Predictor: Linear(128→128)
  ↓
Output: 74D node features (MSE loss)
```

**Parameters**:
- Level 0: 34,378 params
- Levels 1-19: 42,570 params each
- Total: ~850K parameters

### Critical Design Flaws

#### 1. **Graph vs Sequence Mismatch**
- **Problem**: ASTs represent sequential programs with temporal dependencies
- **What we did**: Treated them as undirected graphs with GNN
- **Result**: Lost ordering information crucial for code semantics

#### 2. **No Semantic Grounding**
- **Problem**: Text embedding (64D) is too weak to capture program semantics
- **What we did**: Used alignment model with 64D projection
- **Missing**: Pre-trained code knowledge (CodeBERT, CodeT5 have 768D+)

#### 3. **Hierarchical Independence**
- **Problem**: Each level generator is independent
- **What we did**: Separate models for each depth level
- **Result**: No autoregressive dependencies, can't maintain coherence across levels

#### 4. **Wrong Training Objective**
- **Problem**: MSE loss on one-hot encoded features
- **What we did**: Regression on node type vectors
- **Should be**: Cross-entropy loss on discrete node types (classification)

#### 5. **No Attention Mechanism**
- **Problem**: GNN can't selectively focus on relevant parts of structure
- **Missing**: Self-attention, cross-attention between text and code

#### 6. **No Language Modeling**
- **Problem**: Doesn't learn code syntax rules
- **Missing**: Autoregressive token prediction, teacher forcing

---

## Why Loss Decreased But Validation Failed

The model learned to:
1. **Minimize MSE** by predicting average node features
2. **Generate structurally plausible** graphs (has nodes and edges)
3. **Reproduce training data statistics** (common node types)

But it CANNOT:
1. Generate semantically meaningful code
2. Respond to text descriptions
3. Follow Ruby syntax rules
4. Create diverse, coherent structures

**Analogy**: Like a student who memorized equation forms but doesn't understand math. The model learned the pattern of ASTs but not their meaning.

---

## Fundamental Architecture Issues

### Why GNNs Don't Work for Code Generation

1. **Code is Sequential, Not Graph-Based**
   - Code execution has order (statement 1 → statement 2)
   - GNNs treat all neighbors equally
   - Parent-child in AST ≠ symmetric graph edge

2. **Missing Temporal Context**
   - GNN message passing is order-invariant
   - Code semantics depend on execution order
   - Example: `x = 5; y = x + 1` ≠ `y = x + 1; x = 5`

3. **No Autoregressive Structure**
   - Can't generate token-by-token
   - Can't use previous predictions as context
   - All nodes predicted simultaneously → incoherent output

### Why Hierarchical Levels Don't Help

Each level is **independent**:
- Level 0 doesn't constrain Level 1
- No semantic consistency enforced across levels
- Like building a house floor-by-floor without a blueprint

---

## Comparison: What Works in Code Generation

| Approach | This Project | State-of-the-Art |
|----------|-------------|------------------|
| **Model** | GNN (GCNConv) | Transformer Decoder |
| **Generation** | All-at-once per level | Autoregressive token-by-token |
| **Loss** | MSE on features | Cross-entropy on tokens |
| **Embeddings** | 64D custom | 768D+ pre-trained (CodeBERT) |
| **Context** | Graph structure | Sequential attention |
| **Training** | Level-by-level | End-to-end with teacher forcing |
| **Results** | 0% validity | 40-80% validity (GPT, CodeT5) |

---

## Recommended Next Steps

### ⚡ OPTION A: Pivot to Autoregressive Transformer (RECOMMENDED)

**Why**: Proven architecture for code generation.

**Implementation**:
1. Use Transformer decoder (like GPT)
2. Generate code token-by-token autoregressively
3. Cross-entropy loss on token predictions
4. Pre-trained embeddings (CodeBERT or similar)
5. Teacher forcing during training

**Expected Results**: 40-60% syntactic validity (based on CodeT5 benchmarks)

**Effort**: 2-3 days implementation + 1-2 days training

### OPTION B: Fix Hierarchical Approach (NOT RECOMMENDED)

Would require:
- Complete architecture redesign
- Add attention mechanisms (O(n²) memory)
- Implement autoregressive dependencies between levels
- Use proper classification loss (cross-entropy)
- Increase embedding dimensions (768D+)
- Pre-trained code model integration

**Problems**:
- Still fundamentally graph-based (wrong paradigm)
- High complexity, uncertain payoff
- No proven examples of this working

### OPTION C: Simplify Problem Space

Train on trivial code patterns first:
- Only simple expressions (no functions/classes)
- Depth-1 trees only
- Vocabulary of 10 most common node types

**Purpose**: Validate if approach CAN work on easy tasks

**Outcome**: Even if successful, won't scale to real code

### OPTION D: Use Pre-trained Models

Fine-tune existing models:
- CodeT5, CodeGen, StarCoder
- Already trained on billions of tokens
- Ruby-specific fine-tuning

**Fastest path to working solution**, but less research novelty.

---

## Lessons Learned

### What Worked
1. ✅ Data pipeline (AST extraction, hierarchical levels)
2. ✅ Training infrastructure (PyTorch Geometric, checkpointing)
3. ✅ Loss decreased consistently
4. ✅ Model size appropriate (~850K params)

### What Failed
1. ❌ Graph-based approach for sequential data
2. ❌ Hierarchical independence (no autoregressive structure)
3. ❌ MSE loss on discrete features
4. ❌ Weak text embeddings (64D insufficient)
5. ❌ No pre-trained code knowledge

### Key Insight
**AST generation is fundamentally a sequence modeling problem, not a graph problem.** 

While ASTs have graph structure, generating them requires:
- Sequential dependencies (autoregressive)
- Temporal context (attention)
- Language modeling (token prediction)

GNNs excel at reasoning over fixed graphs, not generating sequential structures.

---

## Conclusion

The hierarchical decoder approach **trained successfully** but **failed fundamentally** due to architectural mismatch. 

**Recommendation**: Pivot to autoregressive transformer approach (Phase 7) immediately. The current architecture cannot be salvaged without a complete redesign that would essentially rebuild it as a transformer anyway.

**Time to pivot**: Now. Every hour spent fixing this is an hour not spent on the proven approach.

---

## Artifacts

- **Training logs**: `training_log_with_penalty.txt` (9.7MB)
- **Models**: `models/hierarchical/hierarchical_decoder_level_[0-19].pt`
- **Validation**: `validation_results_hierarchical.txt`
- **Analysis plots**: 
  - `hierarchical_training_analysis.png` (loss progression)
  - `hierarchical_failure_analysis.png` (comprehensive analysis)

---

**Status**: Analysis complete. Ready to pivot to Phase 7 (Autoregressive Decoder).
