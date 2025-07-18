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

### Current Dataset

The project now contains a complete, cleaned, and ML-ready dataset in the `./dataset/` directory:

**Final Dataset Statistics:**
- **Train set**: `train.jsonl` - 1,517 method entries
- **Validation set**: `validation.jsonl` - 190 method entries  
- **Test set**: `test.jsonl` - 189 method entries
- **Total**: 1,896 filtered methods (from 2,437 original extractions)
- **Complexity range**: 2.0 to 96.1 (filtered from broader range)
- **Data format**: JSONL with enhanced structure including AST and complexity data

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

## Getting Started

### Prerequisites

- Ruby (version 2.7+ recommended)
- Git
- `parser` gem for Ruby AST processing
- `jq` (optional, for viewing JSON data)

### Setup & Usage

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

## Project Structure

```
jubilant-palm-tree/
├── scripts/
│   ├── 01_clone_repos.sh         # Repository cloning automation
│   ├── 02_extract_methods.rb     # Method extraction from Ruby files
│   ├── 03_process_methods.rb     # AST processing and complexity calculation
│   └── 04_assemble_dataset.rb    # Dataset filtering, cleaning, and splitting
├── dataset/                      # Final ML-ready dataset
│   ├── train.jsonl              # Training set (1,517 entries)
│   ├── validation.jsonl         # Validation set (190 entries)
│   └── test.jsonl               # Test set (189 entries)
├── output/                       # Intermediate processing files
│   ├── processed_methods.jsonl   # Complete processed dataset
│   ├── methods.json             # Original extracted methods
│   └── sinatra_methods.json     # Sinatra-specific subset
├── repos/                        # Cloned repositories (excluded from git)
├── Gemfile                       # Ruby dependency management
└── README.md                     # Project documentation
```

## Future Development Suggestions

To improve the development workflow and project maintainability, consider implementing:

### 1. Dependency Management
- **✅ Created a Gemfile** to manage Ruby dependencies
- Current dependencies: `parser` gem for AST processing
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

With Phase 1 complete and the ML-ready dataset assembled, the project is ready to move forward with:
- **Phase 2**: GNN Model Architecture Design & Implementation
- **Phase 3**: Model Training & Hyperparameter Tuning  
- **Phase 4**: Complexity Prediction Validation & Evaluation
- **Phase 5**: Advanced Generative Model Development