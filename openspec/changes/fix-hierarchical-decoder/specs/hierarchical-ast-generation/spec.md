# Hierarchical AST Generation Specification

## ADDED Requirements

### Requirement: Level-by-Level AST Generation
The HierarchicalASTDecoder SHALL generate Abstract Syntax Trees progressively from root to leaves using a coarse-to-fine approach.

#### Scenario: Generate simple method AST
- **GIVEN** a text embedding for "a method that returns sum"
- **WHEN** generate() is called with the embedding
- **THEN** the decoder produces an AST with root node type "def" and at least one child node

#### Scenario: Hierarchical structure maintained
- **GIVEN** a text embedding for any Ruby method
- **WHEN** generate() constructs the AST level-by-level
- **THEN** each node's children are generated only after the node itself exists
- **AND** all parent-child relationships are valid

### Requirement: Graph Neural Network Processing
The HierarchicalASTDecoder level generators SHALL use proper Graph Neural Network layers (GCNConv or equivalent) to process AST structures.

#### Scenario: Process graph input at each level
- **GIVEN** a partial AST graph at level N with node features and edges
- **WHEN** the level N generator processes the graph
- **THEN** it outputs predictions for level N+1 nodes based on graph structure
- **AND** it aggregates information from parent and sibling nodes

#### Scenario: Handle variable graph sizes
- **GIVEN** different ASTs with varying numbers of nodes per level
- **WHEN** GNN layers process these graphs
- **THEN** they handle variable batch sizes correctly
- **AND** they produce appropriate outputs regardless of input graph size

### Requirement: Inference Generate Method
The HierarchicalASTDecoder SHALL provide a generate() method that constructs complete ASTs from embeddings.

#### Scenario: Generate from embedding
- **GIVEN** a 64-dimensional text embedding
- **WHEN** generate(embedding, max_levels=20) is called
- **THEN** the method returns a complete AST structure in JSON format
- **AND** the AST is compatible with Ruby pretty printer scripts

#### Scenario: Terminate generation appropriately
- **GIVEN** a text embedding for a simple method
- **WHEN** generate() constructs the AST
- **THEN** generation stops when no more children are predicted
- **OR** when max_levels depth is reached

#### Scenario: Output valid JSON structure
- **GIVEN** any text embedding
- **WHEN** generate() completes
- **THEN** the output is valid JSON with "type" and "children" fields
- **AND** the structure is a valid tree (no cycles, single root)

### Requirement: Node Spawning and Edge Prediction
The decoder SHALL predict which nodes spawn children and how they connect at each level.

#### Scenario: Predict child nodes for parent
- **GIVEN** a node at level N (e.g., a "def" node)
- **WHEN** the level N+1 generator processes it
- **THEN** it predicts 0-10 child node types (e.g., "args", "block")
- **AND** it creates edges from parent to each predicted child

#### Scenario: Leaf node generation
- **GIVEN** a node that should be a leaf (e.g., a literal value)
- **WHEN** the next level generator processes it
- **THEN** it predicts 0 children for that node
- **AND** that branch of the tree terminates

### Requirement: Training Compatibility
The HierarchicalASTDecoder forward() method SHALL work with hierarchical training data.

#### Scenario: Level-specific training
- **GIVEN** ground truth AST data for level N
- **WHEN** forward(input, target_level=N) is called during training
- **THEN** it produces predictions for level N nodes and edges
- **AND** loss can be computed against ground truth

#### Scenario: Support all 20 levels
- **GIVEN** hierarchical dataset with levels 0-19
- **WHEN** training iterates through each level
- **THEN** each level generator is trained independently
- **AND** level N generator receives output from level N-1 as input

### Requirement: Syntactic Validity Improvement
The fixed HierarchicalASTDecoder SHALL generate syntactically valid Ruby code at a rate greater than 0%.

#### Scenario: Generate valid Ruby syntax
- **GIVEN** a trained HierarchicalASTDecoder
- **WHEN** it generates ASTs for 100 test samples
- **THEN** at least 1 sample produces syntactically valid Ruby code
- **AND** syntactic validity percentage is measurably greater than 0%

#### Scenario: Measurable evaluation metrics
- **GIVEN** generated Ruby code from the decoder
- **WHEN** evaluated with check_syntax.rb
- **THEN** syntactic validity can be measured
- **AND** other metrics (AST isomorphism, BLEU score) can be calculated

### Requirement: Tree Structure Integrity
Generated ASTs SHALL maintain valid tree properties.

#### Scenario: Single root node
- **GIVEN** any generated AST
- **WHEN** tree structure is validated
- **THEN** the AST has exactly one root node
- **AND** all other nodes are descendants of the root

#### Scenario: No cycles in tree
- **GIVEN** any generated AST
- **WHEN** traversing parent-child relationships
- **THEN** no node is its own ancestor
- **AND** no cycles exist in the tree structure

#### Scenario: All nodes have valid types
- **GIVEN** any generated AST
- **WHEN** checking node types
- **THEN** all node types are from the known set of 74 Ruby AST node types
- **OR** nodes are marked as "unknown" if type cannot be determined
