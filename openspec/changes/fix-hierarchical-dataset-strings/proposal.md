# Fix Hierarchical Dataset String Node Handling

## Why

The hierarchical AST dataset created by `scripts/create_ast_level_dataset.py` has a critical flaw that causes 50% of nodes to be encoded as "unknown" (type 73), making the trained model useless:

**Problem**: String children in AST nodes (method names, literals, etc.) are being treated as structural AST nodes and encoded as "unknown" because they have no `type` field.

**Evidence**:
- Evaluation shows trained model predicts type_73 ("unknown") for almost all nodes
- Analysis reveals 50% of training data nodes are type_73
- Raw AST example: `{"type": "def", "children": ["method_name_string"]}` 
- The string `"method_name_string"` becomes a graph node with type="unknown"

**Impact**: The hierarchical decoder learned correctly from the data, but the data itself is malformed. Generated ASTs consist entirely of "unknown" nodes, producing 0% syntactic validity.

## What Changes

Update `scripts/create_ast_level_dataset.py` to properly handle string children:

**Option 1: Filter out string nodes** (Simplest)
- Skip string children when building graph representation
- Only include dict nodes with `type` field as graph nodes
- Strings become edge labels or node attributes, not separate nodes

**Option 2: Add proper node types for strings** (More complete)
- Create specific node types: `method_name`, `variable_name`, `string_literal`, etc.
- Map strings to appropriate types based on parent context
- Expand `ASTNodeEncoder` vocabulary to include these types

**Recommended**: Option 1 for immediate fix, consider Option 2 for future enhancement.

## Impact

### Breaking Changes
- **Dataset format changes**: Hierarchical JSONL files will have different node counts
- **Requires retraining**: All 20 levels of hierarchical models must be retrained
- **Training time**: ~5-6 hours to retrain all levels

### Affected Files
- `scripts/create_ast_level_dataset.py` - Dataset creation logic
- `dataset/hierarchical/` - All level files need regeneration
- `models/hierarchical/` - All trained models invalidated, need retraining
- `src/data_processing.py` - May need updates to handle new format

### Not Affected
- Alignment model (unchanged)
- Base GNN encoder (unchanged)
- Evaluation scripts (work with new format automatically)

## Dependencies

- Existing paired data (`dataset/train_paired_data.jsonl`, etc.)
- Ruby parser for AST processing
- Sufficient disk space for regenerated dataset (~same size as current)

## Success Criteria

1. **Dataset quality**: <5% unknown nodes in hierarchical training data (down from 50%)
2. **Model learning**: Trained model predicts diverse node types (not 90%+ unknown)
3. **Generation quality**: >0% syntactic validity on evaluation (any valid Ruby code)
4. **Structural correctness**: Generated ASTs have proper tree structure with recognized node types

## Rollback Plan

If new approach fails:
1. Keep old dataset in `dataset/hierarchical.backup/`
2. Old trained models in `models/hierarchical.backup/`
3. Can revert dataset generation script via git

## Timeline Estimate

1. Fix dataset creation script: 1 hour
2. Regenerate dataset: 10-20 minutes
3. Retrain all 20 levels: 5-6 hours
4. Evaluation: 5 minutes
5. **Total**: ~7 hours (mostly unattended training)
