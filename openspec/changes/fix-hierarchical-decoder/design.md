# Technical Design: Hierarchical Decoder Fix

## Context

The HierarchicalASTDecoder was intended to generate ASTs level-by-level using a coarse-to-fine approach inspired by the glowing-tribble progressive image generation architecture. However, the current implementation has critical flaws:

1. **Simplistic Architecture**: Uses only `torch.nn.Linear` layers instead of proper Graph Neural Networks
2. **Missing Inference**: No `generate()` method exists for actually constructing ASTs from embeddings
3. **No Tree Building**: No logic to maintain parent-child relationships or construct hierarchical structures
4. **Training-Only Design**: The forward() method only works for training (comparing predictions to ground truth), not for generating new ASTs

## Goals

- Make HierarchicalASTDecoder capable of actual AST generation during inference
- Use proper GNN layers to process graph structures at each level
- Build valid hierarchical tree structures with correct parent-child relationships
- Achieve >0% syntactic validity (any improvement from current 0% is success)

## Non-Goals

- Achieving high accuracy (>80% validity) in this fix - that's a tuning/training concern
- Changing the overall hierarchical/coarse-to-fine approach
- Modifying the training data format or dataset preparation
- Replacing the model with an autoregressive architecture (that's Phase 7)

## Decisions

### 1. GNN Layer Choice

**Decision**: Use `GCNConv` (Graph Convolutional Network) layers for level generators

**Rationale**:
- GCNConv is simpler and more stable than GAT or SAGE for first implementation
- Already proven to work in the encoder (RubyComplexityGNN)
- Can process variable-sized graphs naturally
- Efficient for the small graph sizes at early AST levels

**Alternatives Considered**:
- `SAGEConv`: More complex, no clear benefit for this use case
- `GATConv`: Attention mechanism adds complexity, may overfit on small graphs
- Keep linear layers: Cannot process graph structures, fundamental flaw

### 2. Generation Algorithm

**Decision**: Iterative level-by-level generation with explicit parent tracking

**Algorithm**:
```python
def generate(embedding, max_levels):
    # Level 0: Generate root node
    current_nodes = [generate_root(embedding)]
    
    # Levels 1-N: For each level, generate children for all nodes from previous level
    for level in range(1, max_levels):
        next_nodes = []
        for parent_node in current_nodes:
            children = generate_children(parent_node, level)
            next_nodes.extend(children)
        current_nodes = next_nodes
        
        # Stop if no nodes generated
        if len(current_nodes) == 0:
            break
    
    return construct_ast_tree(all_nodes)
```

**Rationale**:
- Maintains hierarchical structure naturally
- Allows model to decide when to stop generating (0 children = leaf node)
- Explicit parent-child tracking prevents malformed trees
- Matches the training paradigm (level-by-level ground truth)

**Alternatives Considered**:
- Breadth-first generation: Harder to track parent-child relationships
- Fixed-depth generation: May generate too many or too few levels
- Recursive generation: Risk of stack overflow, harder to debug

### 3. Node Spawning Strategy

**Decision**: Each node at level N can spawn 0-10 children at level N+1

**Rationale**:
- Ruby ASTs typically have 0-10 children per node (method bodies, argument lists, etc.)
- Model predicts adjacency matrix indicating which nodes connect
- 0 children = leaf node (terminates that branch)
- Max 10 prevents combinatorial explosion

**Implementation**:
- Adjacency predictor outputs NxN matrix where N = num_nodes_at_current_level
- Threshold sigmoid output (>0.5) to determine parent-child edges
- Each edge represents a child spawned by that parent

### 4. Tree Construction Format

**Decision**: Generate AST in the same JSON format expected by Ruby pretty printer

**Format**:
```json
{
  "type": "def",
  "children": [
    {"type": "args", "children": [...]},
    {"type": "block", "children": [...]}
  ]
}
```

**Rationale**:
- Maintains compatibility with existing Ruby scripts
- No conversion layer needed
- Can directly evaluate with check_syntax.rb
- Matches training data format

### 5. Model Architecture Update

**Current (Broken)**:
```python
level_gnn = torch.nn.Linear(input_dim, hidden_dim)
```

**New (Fixed)**:
```python
level_gnn = GCNConv(input_dim, hidden_dim)
```

**Impact**: 
- Requires graph input (node features + edge_index) instead of flat tensor
- Can aggregate information from parent nodes and siblings
- Learns structural patterns in AST trees

## Risks / Trade-offs

### Risk: Model may still generate invalid ASTs after fix
**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**: 
- This fix addresses architectural flaws, not training quality
- If still getting 0% validity, investigate training data quality, loss functions, or hyperparameters
- Collect debugging data: what node types are being predicted? Are edges being formed?

### Risk: GNN layers may be slower than linear layers
**Likelihood**: High  
**Impact**: Low  
**Mitigation**:
- AST levels are small (root has 1 node, next level ~2-5 nodes)
- GNN overhead is minimal for small graphs
- Correctness is more important than speed for this research project

### Risk: Tree construction may be complex and buggy
**Likelihood**: Medium  
**Impact**: Medium  
**Mitigation**:
- Write comprehensive unit tests for generate() method
- Add logging/debugging output to track node generation
- Start with simple examples (single-line methods) to verify correctness

## Migration Plan

1. **Update model class** in src/models.py (backward compatible - new generate() method)
2. **Re-train models** - old checkpoints are incompatible due to architecture change
3. **Run evaluation** - use new generate() method
4. **Iterate** if needed based on results

**Rollback**: Keep old model code in git history if needed to reference

## Open Questions

1. Should we limit max tree depth to prevent infinite generation?
   - **Proposed**: Yes, use max_levels parameter (default 20 to match training)

2. What if a level predicts no nodes/edges?
   - **Proposed**: Treat as termination signal, stop generation

3. How to handle invalid node type predictions?
   - **Proposed**: Map to "unknown" type, let Ruby validator flag it

4. Should we add beam search or sampling for generation?
   - **Proposed**: Not in this fix - use greedy generation (argmax), add sampling later if needed
