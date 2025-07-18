# jubilant-palm-tree

## Overview

This project is an experiment to determine if a Graph Neural Network (GNN) can learn to understand the structural complexity of Ruby code. The central hypothesis is that by training a GNN on the Abstract Syntax Tree (AST) of thousands of Ruby methods, it can learn to accurately predict a standard complexity metric (Cyclomatic Complexity) without being explicitly taught the rules of the language.

The ultimate goal is not just to predict complexity, but to validate a methodology. If successful, the learned structural embeddings can serve as a foundation for a more advanced generative model capable of writing syntactically correct and logically coherent Ruby code.

## Project Status

### Completed Work

**✅ Phase 1: Data Generation & Preprocessing** 

The complete data collection, processing, and dataset assembly phase has been successfully implemented:

1. **Source Code Aggregation** (`scripts/01_clone_repos.sh`)
   - Automated cloning of 8 high-quality Ruby repositories
   - Target repositories: Rails, Sinatra, Forem, Mastodon, Discourse, Fastlane, Spree, Liquid
   - Robust error handling and idempotent operation
   - Repositories stored in excluded `./repos/` directory

2. **Method Extraction** (`scripts/02_extract_methods.rb`)
   - Comprehensive Ruby method extraction using the `parser` gem
   - Recursive scanning of all `.rb` files in cloned repositories
   - AST-based method detection (both instance `:def` and class `:defs` methods)
   - Structured JSON output with method metadata: repository name, file path, line numbers, raw source code
   - Successfully tested on real repositories (2,437+ methods extracted from 291+ files)

3. **AST Processing & Complexity Analysis** (`scripts/03_process_methods.rb`)
   - Comprehensive AST analysis using Ruby parser gem
   - Cyclomatic complexity calculation for each extracted method
   - Enhanced data structure with JSON-serialized AST representations
   - Output to structured JSONL format in `./output/processed_methods.jsonl`

4. **Dataset Assembly & Cleaning** (`scripts/04_assemble_dataset.rb`)
   - Final dataset filtering by complexity range (2.0 ≤ complexity ≤ 100.0)
   - Removal of edge cases: too simple (<2.0) or too complex (>100.0) methods
   - Unique UUID assignment to each method entry
   - Automated train/validation/test splitting (80/10/10)
   - Clean JSONL output files ready for machine learning training
   - Final dataset: 1,896 methods from 2,437 original extractions (77.8% retention)

5. **Project Infrastructure**
   - Comprehensive documentation with methodology explanation
   - Proper `.gitignore` configuration excluding cloned repositories
   - Complete dataset pipeline from source code to ML-ready format

**✅ Phase 2: Data Ingestion & Graph Conversion**

A complete data ingestion pipeline has been implemented to convert Ruby AST data into graph objects:

1. **AST Graph Conversion** (`src/data_processing.py`)
   - `ASTNodeEncoder` class for mapping Ruby AST node types to feature vectors
   - `ASTGraphConverter` class for parsing AST JSON and creating graph representations
   - Support for 73 common Ruby AST node types with one-hot encoding
   - Robust error handling for malformed AST data
   - Parent-child relationship extraction to edge indices

2. **Custom Dataset Implementation**
   - `RubyASTDataset` class compatible with PyTorch Dataset interface
   - JSONL file loading with automatic graph conversion
   - Individual sample access with graph data and metadata
   - Feature dimension: 74 (73 node types + 1 unknown)

3. **Batch Processing & DataLoader**
   - `collate_graphs` function for batching multiple graph samples
   - `SimpleDataLoader` class as PyTorch DataLoader replacement
   - Automatic node offset calculation for proper batching
   - Support for shuffling and custom batch sizes
   - `create_data_loaders` convenience function for train/validation setup

4. **PyTorch Geometric Compatibility**
   - Direct compatibility with `torch_geometric.data.Data` objects
   - Easy conversion to PyTorch tensors when libraries are available
   - Batch format compatible with `torch_geometric.data.Batch`
   - Ready for GNN model training with `RubyComplexityGNN` model

5. **Comprehensive Testing & Validation**
   - Complete test suite validating all functionality (`test_dataset.py`)
   - Example usage script demonstrating all features (`example_usage.py`)
   - Verified successful loading and collation of graph batches
   - All tests passing: dataset loading, item access, batch collation, DataLoader simulation

**✅ Phase 3: GNN Model Implementation**

A complete Graph Neural Network architecture has been implemented for Ruby complexity prediction:

1. **GNN Model Definition** (`src/models.py`)
   - `RubyComplexityGNN` class as torch.nn.Module
   - Support for both SAGEConv and GCNConv layers for message passing
   - Configurable number of layers (2-4 layers typical)
   - Global mean pooling layer for graph-level embeddings
   - Final linear regression head outputting single complexity score
   - Configurable dropout for regularization

2. **Model Architecture Features**
   - Input dimension: 74 (Ruby AST node features)
   - Hidden dimensions: 32-128 (configurable)
   - Layer types: GCN (Graph Convolutional) or SAGE (GraphSAGE)
   - Global pooling: `global_mean_pool` for graph-level representation
   - Output: Single regression value for complexity prediction
   - Parameter counts: 3K-118K depending on configuration

3. **DataLoader Integration**
   - Fully compatible with DataLoader from Phase 2
   - Processes batched graph data efficiently
   - Handles variable graph sizes in each batch
   - Automatic batch index management for PyTorch Geometric

4. **Model Validation & Testing**
   - Comprehensive test suite (`test_gnn_models.py`)
   - Example usage demonstrations (`example_gnn_usage.py`)
   - Verified single sample and batch processing
   - All model configurations tested and working
   - Error handling for invalid configurations

### Current Dataset

The project now contains a complete, cleaned, and ML-ready dataset in the `./dataset/` directory:

**Final Dataset Statistics:**
- **Train set**: `train.jsonl` - 1,517 method entries
- **Validation set**: `validation.jsonl` - 190 method entries  
- **Test set**: `test.jsonl` - 189 method entries
- **Total**: 1,896 filtered methods (from 2,437 original extractions)
- **Complexity range**: 2.0 to 96.1 (filtered from broader range)
- **Data format**: JSONL with enhanced structure including AST and complexity data

**Graph Conversion Statistics:**
- **Feature dimension**: 74 (73 Ruby AST node types + 1 unknown)
- **Average nodes per graph**: ~48 nodes
- **Average edges per graph**: ~47 edges
- **Node types supported**: 73 common Ruby AST constructs
- **Graph format**: Compatible with PyTorch Geometric

**Legacy datasets** in `./output/`:
- `processed_methods.jsonl` - Complete processed dataset before splitting
- `methods.json` - Original extracted methods (superseded)
- `sinatra_methods.json` - Sinatra framework subset (superseded)

Each final dataset entry includes:
```json
{
  "repo_name": "liquid",
  "file_path": "./repos/liquid/lib/liquid/variable.rb", 
  "start_line": 62,
  "raw_source": "def strict_parse(markup)\n  @filters = []\n  # ... method implementation\nend",
  "complexity_score": 22.4,
  "ast_json": "{\"type\":\"def\",\"children\":[...]}",
  "id": "9167fdae-f91d-49e4-ab6b-d32a5a878748"
}
```

**Graph Data Format:**
When processed through the `RubyASTDataset`, each entry becomes:
```python
{
  "x": [[1.0, 0.0, ...], ...],        # Node features (74-dim one-hot)
  "edge_index": [[0, 1, ...], [...]], # Edge connectivity
  "y": [22.4],                        # Target complexity score
  "num_nodes": 48,                    # Number of nodes in graph
  "id": "9167fdae-...",               # Unique identifier
  "repo_name": "liquid",              # Source repository
  "file_path": "./repos/liquid/..."   # Source file
}
```

## Getting Started

### Prerequisites

- Ruby (version 2.7+ recommended)
- Python (version 3.8+ recommended)
- Git
- `parser` gem for Ruby AST processing
- `jq` (optional, for viewing JSON data)

### Setup & Usage

#### Phase 1: Ruby Data Extraction

1. **Clone the repository:**
   ```bash
   git clone https://github.com/timlawrenz/jubilant-palm-tree.git
   cd jubilant-palm-tree
   ```

2. **Install Ruby dependencies:**
   
   Using Bundler (recommended):
   ```bash
   bundle install
   ```
   
   Or install manually:
   ```bash
   gem install parser
   ```

3. **Collect source code repositories:**
   ```bash
   ./scripts/01_clone_repos.sh
   ```

4. **Extract method definitions:**
   ```bash
   ruby scripts/02_extract_methods.rb
   ```

5. **Process methods and calculate complexity:**
   ```bash
   ruby scripts/03_process_methods.rb
   ```

6. **Assemble final dataset:**
   ```bash
   ruby scripts/04_assemble_dataset.rb
   ```

7. **View final dataset:**
   ```bash
   ls dataset/
   wc -l dataset/*.jsonl
   head -n 1 dataset/train.jsonl | jq .
   ```

#### Phase 2: Python Environment & Graph Processing

8. **Set up Python virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

9. **Install Python dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

10. **Verify installation:**
    ```bash
    python -c "import torch, torch_geometric, pandas; print('✅ All libraries installed successfully')"
    ```

11. **Test the data ingestion pipeline:**
    ```bash
    python test_dataset.py
    ```

12. **Run example usage:**
    ```bash
    python example_usage.py
    ```

13. **Use the dataset in your code:**
    ```python
    # Basic usage
    from src.data_processing import RubyASTDataset, create_data_loaders
    
    # Create datasets
    train_dataset = RubyASTDataset("dataset/train.jsonl")
    sample = train_dataset[0]  # Get first sample
    
    # Create data loaders for training
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=32
    )
    
    # Process batches
    for batch in train_loader:
        # batch contains: x, edge_index, y, batch, metadata
        # Convert to PyTorch tensors when PyTorch is available
        pass
    ```

14. **Test the GNN models:**
    ```bash
    python test_gnn_models.py
    ```

15. **Run GNN model examples:**
    ```bash
    python example_gnn_usage.py
    ```

17. **Explore the data with Jupyter:**
    ```bash
    source venv/bin/activate  # Ensure virtual environment is active
    jupyter notebook notebooks/01_data_exploration.ipynb
    ```
    ```python
    # Import required modules
    from src.data_processing import RubyASTDataset, create_data_loaders
    from src.models import RubyComplexityGNN
    import torch
    from torch_geometric.data import Data
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=32
    )
    
    # Create GNN model (choose GCN or SAGE)
    model = RubyComplexityGNN(
        input_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='SAGE',  # or 'GCN'
        dropout=0.1
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    # Process a batch
    for batch in train_loader:
        # Convert to PyTorch tensors
        x = torch.tensor(batch['x'], dtype=torch.float)
        edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)
        y = torch.tensor(batch['y'], dtype=torch.float)
        batch_idx = torch.tensor(batch['batch'], dtype=torch.long)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Forward pass
        predictions = model(data)
        loss = criterion(predictions.squeeze(), y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        break  # Process one batch for example
    ```

## Project Structure

```
jubilant-palm-tree/
├── scripts/                      # Data extraction and processing scripts
│   ├── 01_clone_repos.sh         # Repository cloning automation
│   ├── 02_extract_methods.rb     # Method extraction from Ruby files
│   ├── 03_process_methods.rb     # AST processing and complexity calculation
│   └── 04_assemble_dataset.rb    # Dataset filtering, cleaning, and splitting
├── dataset/                      # Final ML-ready dataset
│   ├── train.jsonl              # Training set (1,517 entries)
│   ├── validation.jsonl         # Validation set (190 entries)
│   └── test.jsonl               # Test set (189 entries)
├── src/                         # Python source code for GNN training
│   ├── __init__.py              # Package initialization
│   ├── data_processing.py       # Data loading, AST conversion, and Dataset class
│   └── models.py                # PyTorch Geometric GNN model implementations (GCN & SAGE)
├── notebooks/                   # Jupyter notebooks for analysis
│   └── 01_data_exploration.ipynb # Data exploration and visualization
├── output/                      # Intermediate processing files
│   ├── processed_methods.jsonl  # Complete processed dataset
│   ├── methods.json            # Original extracted methods
│   └── sinatra_methods.json    # Sinatra-specific subset
├── test_dataset.py             # Comprehensive test suite for data pipeline
├── test_gnn_models.py          # Comprehensive test suite for GNN models
├── example_usage.py            # Example usage of dataset and DataLoader
├── example_gnn_usage.py        # Advanced GNN model demonstration and training setup
├── repos/                      # Cloned repositories (excluded from git)
├── Gemfile                     # Ruby dependency management
├── venv/                       # Python virtual environment (excluded from git)
├── requirements.txt            # Python dependencies for GNN training
└── README.md                   # Project documentation
```

## Future Development Suggestions

To improve the development workflow and project maintainability, consider implementing:

### 1. Dependency Management
- **✅ Created a Gemfile** to manage Ruby dependencies
- **✅ Created requirements.txt** for Python ML dependencies
- Current Ruby dependencies: `parser` gem for AST processing
- Current Python dependencies: `torch`, `torch_geometric`, `pandas`, `tqdm`, and supporting libraries
- Future: Add development and testing dependencies as needed

### 2. Testing Infrastructure
- Add a testing framework (RSpec or Minitest)
- Create unit tests for the method extraction logic
- Add integration tests for the complete pipeline
- Implement test fixtures with known Ruby code samples

### 3. Build Automation
- **Create a Rakefile** for common development tasks:
  ```ruby
  task :setup => [:clone_repos, :install_deps]
  task :extract => :setup do
    ruby 'scripts/02_extract_methods.rb'
  end
  task :test do
    ruby 'test/run_tests.rb'
  end
  ```

### 4. Data Validation & Quality
- Add method complexity calculation (cyclomatic complexity)
- Implement data validation for extracted methods
- Add statistics generation for the dataset
- Create data quality reports

### 5. Development Environment
- Add linting configuration (RuboCop)
- Create development setup documentation
- Add CI/CD pipeline configuration
- Implement code formatting standards

### 6. Enhanced Documentation
- Add API documentation for the extraction classes
- Create developer contribution guidelines
- Document the AST processing methodology
- Add examples of using the extracted data

These improvements would provide a solid foundation for collaborative development and ensure code quality as the project scales to implement the GNN training phases.

## Next Steps

With Phases 1, 2, and 3 complete, the project is ready to move forward with:
- **Phase 4**: GNN Model Training & Hyperparameter Tuning  
- **Phase 5**: Complexity Prediction Validation & Evaluation
- **Phase 6**: Model Optimization & Production Deployment

### Immediate Next Actions

1. **Implement full training loop** using the provided DataLoader and GNN models
2. **Add model evaluation metrics** (MAE, RMSE, R²) for complexity prediction accuracy
3. **Hyperparameter tuning** for optimal model performance between GCN and SAGE architectures
4. **Cross-validation** to ensure robust performance across different Ruby codebases
5. **Model comparison** between different architectures and layer configurations

### Python GNN Development Environment

The project now includes a comprehensive Python environment for Graph Neural Network development:

**Core Libraries Available:**
- `torch` (2.7.1+): Deep learning framework
- `torch_geometric` (2.6.1+): Graph neural network extensions
- `pandas` (2.3.1): Data manipulation and analysis  
- `tqdm` (4.67.1): Progress bars for training loops

**Development Tools:**
- `jupyter`: Interactive notebook environment
- `matplotlib` & `seaborn`: Data visualization
- `scikit-learn`: Additional ML utilities
- `numpy`: Numerical computing

**Ready-to-Use Components:**
- `src/data_processing.py`: Complete data ingestion and graph conversion pipeline
- `src/models.py`: PyTorch Geometric GNN model implementations (GCN and SAGE)
- `test_dataset.py`: Comprehensive data pipeline testing suite
- `test_gnn_models.py`: Complete GNN model validation and testing
- `example_usage.py`: Basic usage examples and PyTorch integration guide
- `example_gnn_usage.py`: Advanced GNN model demonstration and training setup
- `notebooks/01_data_exploration.ipynb`: Data exploration and visualization notebook

**Data Pipeline & Model Status:**
✅ **COMPLETE**: DataLoader successfully loads and collates batches of graph objects from training dataset
✅ **COMPLETE**: GNN models (GCN and SAGE) process batched data and output complexity predictions
✅ **COMPLETE**: Full integration between data pipeline and GNN models verified

The Python environment can process the complete Ruby method dataset (1,896 methods across train/validation/test splits) and includes working GNN models ready for training and evaluation.
