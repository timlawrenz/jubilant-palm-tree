# jubilant-palm-tree

## Overview

This project is an experiment to determine if a Graph Neural Network (GNN) can learn to understand the structural complexity of Ruby code. The central hypothesis is that by training a GNN on the Abstract Syntax Tree (AST) of thousands of Ruby methods, it can learn to accurately predict a standard complexity metric (Cyclomatic Complexity) without being explicitly taught the rules of the language.

The ultimate goal is not just to predict complexity, but to validate a methodology. If successful, the learned structural embeddings can serve as a foundation for a more advanced generative model capable of writing syntactically correct and logically coherent Ruby code.

## Project Status

### Completed Work

**✅ Phase 1: Data Generation & Preprocessing** 

The initial data collection and preprocessing phase has been successfully implemented:

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

3. **Project Infrastructure**
   - Comprehensive documentation with methodology explanation
   - Proper `.gitignore` configuration excluding cloned repositories
   - Output directory structure for generated datasets

### Current Dataset

The project currently contains extracted method data in `./output/`:
- `methods.json` - Complete dataset of extracted Ruby methods
- `sinatra_methods.json` - Subset focusing on Sinatra framework methods

Each method entry includes:
```json
{
  "repo_name": "sinatra",
  "file_path": "./repos/sinatra/lib/sinatra/base.rb", 
  "start_line": 36,
  "raw_source": "def accept\n  @env['sinatra.accept'] ||= if @env.include?('HTTP_ACCEPT')\n    # method implementation\n  end\nend"
}
```

## Getting Started

### Prerequisites

- Ruby (version 2.7+ recommended)
- Git
- `parser` gem for Ruby AST processing

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

5. **View extracted data:**
   ```bash
   ls output/
   head -n 20 output/methods.json
   ```

## Project Structure

```
jubilant-palm-tree/
├── scripts/
│   ├── 01_clone_repos.sh      # Repository cloning automation
│   └── 02_extract_methods.rb  # Method extraction from Ruby files
├── output/
│   ├── methods.json           # Complete extracted methods dataset
│   └── sinatra_methods.json   # Sinatra-specific subset
├── repos/                     # Cloned repositories (excluded from git)
├── Gemfile                    # Ruby dependency management
└── README.md                  # Project documentation
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

With Phase 1 complete, the project is ready to move forward with:
- **Phase 2**: AST Processing & Feature Engineering
- **Phase 3**: GNN Model Architecture & Training
- **Phase 4**: Complexity Prediction & Validation