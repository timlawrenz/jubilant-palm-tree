# Project Context

## Purpose
This project explores the potential of Graph Neural Networks (GNNs) to understand and generate Ruby code through Abstract Syntax Tree (AST) analysis. The project demonstrates that neural networks can learn meaningful structural representations of code complexity and successfully reconstruct AST structures from learned embeddings.

**Key Goals:**
- Build GNN models that understand Ruby code structure via AST analysis
- Predict code complexity (cyclomatic complexity) with >26% improvement over heuristic baselines
- Reconstruct complete AST structures from learned embeddings
- Align text descriptions with code embeddings for text-to-code generation
- Generate Ruby code from natural language descriptions

**Current Status:**
- ✅ Phase 1-3: Foundational GNN for complexity prediction (MAE 4.77 vs baseline 6.50)
- 🚧 Phase 4b-6: Generative models (hierarchical decoder, text-code alignment, text-to-code) in progress
- 🚧 Phase 7: Advanced autoregressive decoder architectures planned

## Tech Stack
**Core Languages:**
- Python 3.8+ (ML pipeline, GNN models, training)
- Ruby 2.7+ (AST processing, data extraction, code generation)

**ML/AI Frameworks:**
- PyTorch (neural network training and inference)
- PyTorch Geometric (graph neural network operations)
- Sentence Transformers (text embeddings)

**Ruby Libraries:**
- `parser` gem (AST processing and analysis)
- `json` gem (data serialization)
- `bundler` (dependency management)

**Development Tools:**
- Git (version control)
- CircleCI (continuous integration)
- pytest (Python testing)
- Jupyter notebooks (analysis and visualization)

## Project Conventions

### Code Style
**Python:**
- Follow PEP 8 conventions
- Use descriptive variable names (e.g., `complexity_score`, `ast_json`)
- Minimize comments - code should be self-documenting
- Only comment code that needs clarification

**Ruby:**
- Follow Ruby community style guide
- Use snake_case for variables and methods
- Use `parser` gem for all AST operations
- Maintain idempotent script operations

**File Naming:**
- Training scripts: `train_*.py` (e.g., `train_alignment.py`, `train_hierarchical.py`)
- Validation scripts: `validate_*.py`
- Demo scripts: Place in `demos/` directory
- Examples: Place in `examples/` directory
- Test files: `test_*.py` or `*_test.py` in `tests/` directory

### Architecture Patterns
**Data Pipeline:**
```
Raw Ruby Code → AST Extraction → Feature Engineering → Graph Conversion → GNN Model
```

**Model Architecture:**
- **Encoder**: GNN (Graph Attention Networks) converts AST graphs to 64-dimensional embeddings
- **Complexity Predictor**: MLP head on encoder for cyclomatic complexity prediction
- **Hierarchical Decoder**: Top-down AST reconstruction from embeddings
- **Text Encoder**: Sentence Transformer for natural language descriptions
- **Alignment**: Contrastive learning to align text and code embeddings

**Data Format:**
- Dataset: JSONL format with one JSON object per line
- Each entry contains: `repo_name`, `file_path`, `start_line`, `raw_source`, `complexity_score`, `ast_json`, `id` (UUID)
- Train/validation/test split: 80/10/10

**Model Checkpoints:**
- Saved in `models/` directory
- Naming convention: `{model_type}_best.pth` (e.g., `complexity_model_best.pth`)

### Testing Strategy
**Unit Tests:**
- Python tests in `tests/` directory using pytest
- Key test files: `test_dataset.py`, `test_autoencoder.py`
- Run with: `python tests/test_*.py`

**Integration Tests:**
- Example usage demonstrations in `examples/example_usage.py`
- Demo scripts in `demos/` directory verify end-to-end pipelines

**Validation:**
- Ruby AST processing: `ruby test-ruby-setup.rb`
- Syntax checking: `ruby scripts/check_syntax.rb`
- AST pretty printing: `ruby scripts/pretty_print_ast.rb`

**CI/CD:**
- CircleCI runs automated tests on commits
- Uses sample datasets for quick test execution (`test_sample.jsonl`, etc.)

### Git Workflow
**Repository Structure:**
- Main branch: `main` (protected)
- Development happens in feature branches
- CircleCI badge indicates build status

**Commit Conventions:**
- Clear, descriptive commit messages
- Reference GitHub issues when applicable
- Track progress through 7 project phases

## Domain Context
**Abstract Syntax Trees (ASTs):**
- Ruby code is parsed into tree structures representing program semantics
- 73 distinct Ruby AST node types identified in dataset
- AST nodes serialized to JSON format for ML processing
- Methods average ~48 AST nodes each

**Cyclomatic Complexity:**
- Metric measuring code complexity based on control flow paths
- Range: 2.0 to 100.0 in filtered dataset
- GNN achieves MAE of 4.77 (26.6% better than heuristic baseline of 6.50)
- Used as primary training objective for encoder

**Graph Neural Networks:**
- ASTs naturally map to graph structures (nodes = AST nodes, edges = parent-child relationships)
- GAT (Graph Attention Network) layers learn structural representations
- Embeddings capture code semantics in 64-dimensional space
- Learned embeddings enable both discriminative (complexity) and generative (AST reconstruction) tasks

**Dataset Characteristics:**
- 218,000+ Ruby methods from 42 open-source projects
- Core repositories: Rails, Sinatra, Forem, Mastodon, Discourse, Fastlane, Spree, Liquid
- Final filtered dataset: 1,896 methods (77.8% retention rate)
- Quality filters: complexity 2.0-100.0, valid AST, complete metadata

**Text-Code Alignment:**
- Contrastive learning aligns natural language descriptions with code embeddings
- Goal: Same point in 64D embedding space for description and corresponding code
- Currently in progress (Recall@10 ~2-3%, near random chance)

## Important Constraints
**Technical Constraints:**
- Ruby 2.7+ required for AST processing compatibility
- Python 3.8+ required for PyTorch Geometric
- GNN models require GPU for efficient training (CPU fallback available)
- 64-dimensional embedding space fixed across all models for compatibility

**Data Constraints:**
- Dataset limited to Ruby language only
- Complexity range filtered to 2.0-100.0 (excludes trivial and extremely complex code)
- AST must be valid and parseable by Ruby parser gem
- Train/validation/test split must remain consistent (80/10/10)

**Model Constraints:**
- Encoder architecture frozen after Phase 2 training
- Embedding dimension (64) must remain constant for alignment and generation
- Hierarchical decoder operates top-down (cannot generate siblings before parent)
- Text encoder uses fixed Sentence Transformer architecture

**Performance Targets:**
- Complexity prediction: MAE < 6.50 (heuristic baseline)
- AST reconstruction: Target >80% syntactic validity
- Text-code alignment: Target Recall@10 >50%
- Code generation: Target syntactically valid Ruby code

## External Dependencies
**Ruby Gem Dependencies:**
- `parser` (~3.x): AST parsing and analysis - CRITICAL for all Ruby processing
- `json`: Data serialization
- `bundler`: Dependency management

**Python Package Dependencies:**
- `torch` (PyTorch): Neural network framework
- `torch-geometric`: Graph neural network operations
- `sentence-transformers`: Text embedding models
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- `pytest`: Testing framework

**External Resources:**
- Open-source Ruby repositories (cloned to `./repos/` - excluded from git)
- Pre-trained Sentence Transformer models (downloaded on first use)
- CircleCI for continuous integration

**Development Setup:**
- Ruby gems installed to user directory to avoid permission issues
- Virtual environment (`venv/`) for Python dependencies
- Environment setup scripts: `setup-ruby.sh`, `.env-ruby`
- Sample datasets for CI testing (avoid full dataset download)
