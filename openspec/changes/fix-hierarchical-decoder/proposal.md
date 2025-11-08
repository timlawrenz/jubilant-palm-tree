# Fix Hierarchical Decoder Architecture and Inference

## Why

The current HierarchicalASTDecoder implementation is incomplete and non-functional:
- The model uses simple linear layers instead of proper GNN layers that can process graph structures
- The model has no `generate()` method for inference, making it impossible to actually construct ASTs during evaluation
- Evaluation shows 0% syntactic validity because the model cannot produce valid ASTs from embeddings

This critical flaw prevents Phase 4b from succeeding and blocks progress toward functional text-to-code generation.

## What Changes

- **Implement proper `generate()` method** for HierarchicalASTDecoder that constructs ASTs level-by-level during inference
- **Replace linear layers with GNN layers** in the level generators to properly process graph structures at each level
- **Add tree construction logic** that maintains parent-child relationships and builds valid AST structures
- **Re-train the corrected model** with proper architecture to learn hierarchical AST generation
- **Re-evaluate** to verify the model can now generate syntactically valid Ruby code

## Impact

- **Affected specs**: `hierarchical-ast-generation` (new spec to be created)
- **Affected code**: 
  - `src/models.py` - HierarchicalASTDecoder class
  - `train_hierarchical.py` - May need updates for new architecture
  - `scripts/evaluate_hierarchical_model.py` - Uses generate() method
  - Models: `models/hierarchical/hierarchical_decoder_level_*.pt` - Need retraining

## Breaking Changes

None - this fixes a non-functional feature, doesn't break existing working functionality.

## Dependencies

- Existing trained alignment model (`models/best_alignment_model.pt`) for text embeddings
- Hierarchical dataset (`dataset/hierarchical/`) already prepared
- Ruby AST processing scripts (`scripts/pretty_print_ast.rb`, `scripts/check_syntax.rb`)

## Success Criteria

- HierarchicalASTDecoder has a functional `generate()` method
- Model uses proper GNN layers (not just linear layers) for graph processing
- Evaluation achieves >0% syntactic validity (any valid Ruby code generated is progress)
- Model can complete full inference pipeline: text prompt → embedding → hierarchical AST → Ruby code
