# Design: String Node Filtering in Hierarchical Dataset

## Problem Analysis

### Current Behavior
```python
# Input AST
{
  "type": "def",
  "children": ["method_name", {"type": "args", "children": []}]
}

# Current output: Creates 3 graph nodes
# Node 0: type=def (valid)
# Node 1: type=unknown (STRING - problem!)
# Node 2: type=args (valid)
```

### Root Cause
The `build_partial_ast()` function recursively includes all children, regardless of type:
- Dict children with `"type"` field → valid AST nodes
- String children → no `"type"` field → encoded as unknown
- ~50% of children are strings (method names, variable names, literals)

## Solution Design

### Approach: Filter String Nodes

**Key insight**: Strings in AST children are **data**, not **structure**. They should not become separate graph nodes.

### Implementation Strategy

**Location**: `scripts/create_ast_level_dataset.py`

**Function to modify**: `build_partial_ast()`

**Changes**:
```python
def build_partial_ast(original_node, max_depth, current_depth=0):
    # ... existing code ...
    
    if 'children' in original_node and original_node['children']:
        for child in original_node['children']:
            # ADDED: Only process dict children with 'type' field
            if isinstance(child, dict) and 'type' in child:
                partial_child = build_partial_ast(child, max_depth, current_depth + 1)
                if partial_child:
                    new_node['children'].append(partial_child)
            # String children are silently skipped
            # They can be recovered from original AST if needed for generation
    
    return new_node
```

### Alternative Considered: Add String Node Types

**Rejected because**:
1. Requires extending `ASTNodeEncoder` vocabulary
2. Requires context-aware type mapping (method name vs variable name)
3. Adds complexity without clear benefit
4. Can be added later if needed

### Data Flow Impact

**Before filtering**:
```
AST JSON → Graph with 100 nodes (50 unknown) → Training data
```

**After filtering**:
```
AST JSON → Graph with 50 nodes (all typed) → Training data
```

**Benefits**:
- 50% reduction in node count → faster training
- 0% unknown nodes (ideally) → meaningful learning
- Cleaner graph structure → better GNN performance

## Validation Strategy

### Dataset Quality Checks

1. **Unknown node percentage**: 
   ```python
   unknown_count / total_nodes < 0.05  # Allow 5% for truly unknown types
   ```

2. **Node count reduction**:
   ```python
   new_node_count ≈ old_node_count * 0.5  # Expect ~50% reduction
   ```

3. **AST structure preservation**:
   ```python
   # Verify all dict nodes with 'type' are preserved
   # Verify parent-child relationships intact
   ```

### Training Quality Checks

1. **Type diversity**: Model should predict 20+ different types (not just one)
2. **Loss convergence**: Training loss should decrease at each level
3. **Generation quality**: Syntactic validity >0% on evaluation

## Backward Compatibility

**Breaking change**: Yes - dataset format changes

**Migration path**:
1. Backup old dataset
2. Regenerate with new script
3. Retrain models
4. No code changes needed in consumers (format compatible)

## Error Handling

**Edge cases**:
1. **Empty children after filtering**: Return node with empty children list (valid)
2. **All children are strings**: Node becomes leaf (valid)  
3. **Nested string values**: Already handled by `isinstance(child, dict)` check

## Performance Considerations

**Dataset generation**:
- Same O(n) complexity
- Slightly faster (fewer nodes to process)

**Training**:
- Faster (50% fewer nodes per graph)
- Better convergence (cleaner signal)

**Inference**:
- Same speed
- Better quality (meaningful predictions)

## Testing Plan

1. **Unit test**: Filter logic with mock AST data
2. **Integration test**: Generate full level file and verify statistics
3. **End-to-end test**: Train 1 level, verify >0% non-unknown predictions
4. **Regression test**: Compare old vs new on same source ASTs

## Rollback Triggers

Rollback if any of:
1. Unknown nodes >10% in new dataset
2. Training fails to converge (loss doesn't decrease)
3. Generated ASTs have structural errors (not just wrong types)
4. Evaluation still shows 0% validity after retraining

## Future Enhancements

Consider for Phase 4c or later:
1. Add specific string types (method_name, var_name, literal)
2. Store string values as node attributes
3. Use strings during generation to create realistic names
4. Add string embedding layer for better name generation
