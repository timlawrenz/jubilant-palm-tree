# Implementation Tasks

## 1. Architecture Fix
- [x] 1.1 Replace linear layers with proper GNN layers (GCNConv or SAGEConv) in level generators
- [x] 1.2 Add graph processing capabilities to handle node features and edge connections
- [x] 1.3 Update forward() method to work with graph structures instead of flat tensors
- [x] 1.4 Add proper node type prediction and edge prediction heads

## 2. Inference Implementation
- [x] 2.1 Implement generate() method that takes an embedding and constructs AST level-by-level
- [x] 2.2 Add tree construction logic that maintains parent-child relationships
- [x] 2.3 Implement node spawning logic that generates child nodes for each parent at each level
- [x] 2.4 Add termination criteria to stop generation at appropriate depth
- [x] 2.5 Convert generated graph structure to AST JSON format compatible with Ruby pretty printer

## 3. Training Updates
- [x] 3.1 Verify training script works with new architecture
- [x] 3.2 Update data loading if needed for new GNN layers
- [x] 3.3 Adjust loss functions if architecture changes require it
- [ ] 3.4 Re-train all 20 level models with corrected architecture

## 4. Evaluation
- [ ] 4.1 Run evaluation script with newly trained models
- [ ] 4.2 Verify generate() method is called correctly
- [ ] 4.3 Check that generated ASTs have proper structure (root, children, hierarchy)
- [ ] 4.4 Measure syntactic validity, AST isomorphism, and BLEU scores
- [ ] 4.5 Document results and compare to baseline (current 0%)

## 5. Documentation
- [x] 5.1 Update model docstrings to reflect new architecture
- [x] 5.2 Document generate() method parameters and return format
- [ ] 5.3 Add usage examples for inference
- [ ] 5.4 Update README_phase4b.md with findings
