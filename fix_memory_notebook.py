#!/usr/bin/env python3
"""
Script to fix memory issues in evaluate_autoencoder_consolidated.ipynb
by implementing memory-efficient batch processing and JSONL caching.
"""

import json
import re

def inject_memory_optimizations():
    """Inject memory optimization code into the notebook"""
    
    # Read the original notebook
    with open('notebooks/evaluate_autoencoder_consolidated.ipynb', 'r') as f:
        notebook = json.load(f)
    
    # Find the configuration cell and update it
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'CONFIG = {' in ''.join(cell['source']):
            # Update CONFIG to include memory optimization settings
            new_config_lines = [
                "# Configuration - adjust these values to control evaluation scope\n",
                "CONFIG = {\n",
                "    'num_samples': 10,  # Number of samples to evaluate (4 -> 10 -> 100 -> 500+ -> 1000+)\n",
                "    'random_seed': 42,   # For reproducible sample selection\n",
                "    'enable_ruby_conversion': True,  # Enable Ruby pretty-printing (slower but comprehensive)\n",
                "    'ruby_timeout': 15,  # Timeout for Ruby subprocess calls\n",
                "    'save_results': True,  # Save detailed results to file\n",
                "    'show_comparisons': 5,  # Number of detailed comparisons to show\n",
                "    'batch_size': 50,  # Process samples in batches to control memory usage\n",
                "    'enable_memory_optimization': True,  # Enable memory-efficient processing\n",
                "    'max_memory_mb': 2048,  # Maximum memory usage before triggering cleanup (MB)\n",
                "    'cache_jsonl_data': True,  # Cache JSONL data to avoid repeated file reads\n",
                "}\n",
                "\n",
                "print(f\"Configuration:\")\n",
                "for key, value in CONFIG.items():\n",
                "    print(f\"  {key}: {value}\")\n",
                "    \n",
                "# Set random seed for reproducible results\n",
                "np.random.seed(CONFIG['random_seed'])\n",
                "torch.manual_seed(CONFIG['random_seed'])"
            ]
            
            # Find where CONFIG starts and ends
            source_lines = cell['source']
            config_start = None
            config_end = None
            
            for i, line in enumerate(source_lines):
                if 'CONFIG = {' in line:
                    config_start = i
                if config_start is not None and line.strip() == '}' and 'CONFIG' not in source_lines[i+1] if i+1 < len(source_lines) else True:
                    config_end = i + 1
                    break
            
            if config_start is not None:
                # Replace the CONFIG section
                cell['source'] = source_lines[:config_start] + new_config_lines + source_lines[config_end+1:]
            break
    
    # Find the helper functions cell and add memory optimization functions
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'def convert_sample_to_torch' in ''.join(cell['source']):
            # Add memory optimization functions
            memory_optimization_code = [
                "\n",
                "# Memory optimization functions\n",
                "import gc\n",
                "import psutil\n",
                "import os\n",
                "\n",
                "def get_memory_usage_mb():\n",
                "    \"\"\"Get current memory usage in MB\"\"\"\n",
                "    process = psutil.Process(os.getpid())\n",
                "    return process.memory_info().rss / 1024 / 1024\n",
                "\n",
                "def cleanup_memory():\n",
                "    \"\"\"Force garbage collection and memory cleanup\"\"\"\n",
                "    gc.collect()\n",
                "    if hasattr(torch.cuda, 'empty_cache'):\n",
                "        torch.cuda.empty_cache()\n",
                "\n",
                "def should_cleanup_memory():\n",
                "    \"\"\"Check if memory cleanup is needed\"\"\"\n",
                "    if not CONFIG.get('enable_memory_optimization', True):\n",
                "        return False\n",
                "    \n",
                "    current_memory = get_memory_usage_mb()\n",
                "    max_memory = CONFIG.get('max_memory_mb', 2048)\n",
                "    return current_memory > max_memory\n",
                "\n",
                "class JSONLCache:\n",
                "    \"\"\"Cache for JSONL file data to avoid repeated reads\"\"\"\n",
                "    \n",
                "    def __init__(self, jsonl_path):\n",
                "        self.jsonl_path = jsonl_path\n",
                "        self.cache = {}\n",
                "        self.loaded = False\n",
                "    \n",
                "    def get_sample_data(self, idx):\n",
                "        \"\"\"Get sample data by index, loading from cache or file\"\"\"\n",
                "        if not CONFIG.get('cache_jsonl_data', True):\n",
                "            # Fall back to reading from file each time\n",
                "            return self._read_sample_from_file(idx)\n",
                "        \n",
                "        if not self.loaded:\n",
                "            self._load_cache()\n",
                "        \n",
                "        return self.cache.get(idx, None)\n",
                "    \n",
                "    def _load_cache(self):\n",
                "        \"\"\"Load all data into cache (memory intensive but faster)\"\"\"\n",
                "        try:\n",
                "            with open(self.jsonl_path, 'r') as f:\n",
                "                for idx, line in enumerate(f):\n",
                "                    line = line.strip()\n",
                "                    if line:\n",
                "                        data_dict = json.loads(line)\n",
                "                        self.cache[idx] = {\n",
                "                            'raw_source': data_dict.get('method_source', ''),\n",
                "                            'ast_json': data_dict.get('ast_json', '{}')\n",
                "                        }\n",
                "            self.loaded = True\n",
                "            print(f\"Cached {len(self.cache)} samples from {self.jsonl_path}\")\n",
                "        except Exception as e:\n",
                "            print(f\"Warning: Could not cache JSONL data: {e}\")\n",
                "            self.loaded = False\n",
                "    \n",
                "    def _read_sample_from_file(self, idx):\n",
                "        \"\"\"Read a single sample from file (slower but memory efficient)\"\"\"\n",
                "        try:\n",
                "            with open(self.jsonl_path, 'r') as f:\n",
                "                for i, line in enumerate(f):\n",
                "                    if i == idx:\n",
                "                        data_dict = json.loads(line)\n",
                "                        return {\n",
                "                            'raw_source': data_dict.get('method_source', ''),\n",
                "                            'ast_json': data_dict.get('ast_json', '{}')\n",
                "                        }\n",
                "        except Exception as e:\n",
                "            print(f\"Warning: Could not read sample {idx}: {e}\")\n",
                "        return None\n",
                "\n",
                "# Initialize JSONL cache\n",
                "jsonl_cache = JSONLCache('../dataset/test.jsonl')\n",
                "\n"
            ]
            
            # Insert at the end of the cell
            cell['source'] = cell['source'] + memory_optimization_code
            break
    
    # Find the core evaluation functions cell and replace them with optimized versions
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'def evaluate_samples_batch' in ''.join(cell['source']):
            # Replace with memory-optimized evaluation functions
            optimized_functions = [
                "def evaluate_sample_fast_optimized(sample, sample_idx, include_ruby=True):\n",
                "    \"\"\"Memory-optimized version of evaluate_sample_fast\"\"\"\n",
                "    # Convert to torch format\n",
                "    data = convert_sample_to_torch(sample)\n",
                "    \n",
                "    # Pass through autoencoder\n",
                "    with torch.no_grad():\n",
                "        result = autoencoder(data)\n",
                "        embedding = result['embedding']\n",
                "        reconstruction = result['reconstruction']\n",
                "    \n",
                "    # Get original data using cached JSONL reader\n",
                "    original_code = None\n",
                "    original_ast = None\n",
                "    \n",
                "    cached_data = jsonl_cache.get_sample_data(sample_idx)\n",
                "    if cached_data:\n",
                "        original_code = cached_data['raw_source']\n",
                "        try:\n",
                "            original_ast = json.loads(cached_data['ast_json'])\n",
                "        except json.JSONDecodeError:\n",
                "            original_ast = None\n",
                "    \n",
                "    # Reconstruct AST from decoder output\n",
                "    reconstructed_ast = reconstruct_ast_from_features(\n",
                "        reconstruction['node_features'],\n",
                "        reconstruction\n",
                "    )\n",
                "    \n",
                "    result_dict = {\n",
                "        'sample_idx': sample_idx,\n",
                "        'embedding_dim': embedding.shape[1],\n",
                "        'original_code': original_code,\n",
                "        'original_ast': original_ast,\n",
                "        'reconstructed_ast': reconstructed_ast,\n",
                "        'original_nodes': len(sample['x']),\n",
                "        'reconstructed_nodes': reconstruction['node_features'].shape[1],\n",
                "        'reconstructed_code': None,  # Will be filled later if Ruby conversion enabled\n",
                "        'ruby_conversion_error': None\n",
                "    }\n",
                "    \n",
                "    return result_dict\n",
                "\n",
                "def evaluate_samples_batch_optimized(sample_indices, include_ruby=True):\n",
                "    \"\"\"Memory-optimized batch evaluation with streaming results\"\"\"\n",
                "    print(f\"\\nEvaluating {len(sample_indices)} samples with memory optimization...\")\n",
                "    \n",
                "    batch_size = CONFIG.get('batch_size', 50)\n",
                "    total_samples = len(sample_indices)\n",
                "    all_results = []\n",
                "    \n",
                "    print(f\"Processing in batches of {batch_size} samples\")\n",
                "    print(f\"Initial memory usage: {get_memory_usage_mb():.1f} MB\")\n",
                "    \n",
                "    # Process samples in batches\n",
                "    for batch_start in range(0, total_samples, batch_size):\n",
                "        batch_end = min(batch_start + batch_size, total_samples)\n",
                "        batch_indices = sample_indices[batch_start:batch_end]\n",
                "        \n",
                "        print(f\"\\nProcessing batch {batch_start//batch_size + 1}/{(total_samples-1)//batch_size + 1} (samples {batch_start+1}-{batch_end})\")\n",
                "        \n",
                "        # Phase 1: Fast autoencoder inference for this batch\n",
                "        batch_results = []\n",
                "        for idx in safe_tqdm(batch_indices, desc=f\"Batch {batch_start//batch_size + 1} inference\", leave=False, dynamic_ncols=False):\n",
                "            if idx < len(test_dataset):\n",
                "                sample = test_dataset[idx]\n",
                "                result = evaluate_sample_fast_optimized(sample, idx, include_ruby=False)\n",
                "                batch_results.append(result)\n",
                "                \n",
                "                # Check memory usage periodically\n",
                "                if should_cleanup_memory():\n",
                "                    print(f\"Memory usage high ({get_memory_usage_mb():.1f} MB), cleaning up...\")\n",
                "                    cleanup_memory()\n",
                "        \n",
                "        # Phase 2: Ruby code conversion for this batch (if enabled)\n",
                "        if include_ruby and CONFIG['enable_ruby_conversion']:\n",
                "            for result in safe_tqdm(batch_results, desc=f\"Batch {batch_start//batch_size + 1} Ruby conversion\", leave=False, dynamic_ncols=False):\n",
                "                # Convert reconstructed AST\n",
                "                reconstructed_code = ast_to_ruby_code_safe(\n",
                "                    result['reconstructed_ast'], \n",
                "                    CONFIG['ruby_timeout']\n",
                "                )\n",
                "                result['reconstructed_code'] = reconstructed_code\n",
                "                if reconstructed_code.startswith('Error:'):\n",
                "                    result['ruby_conversion_error'] = reconstructed_code\n",
                "        \n",
                "        # Add batch results to all results\n",
                "        all_results.extend(batch_results)\n",
                "        \n",
                "        # Clean up memory after each batch\n",
                "        cleanup_memory()\n",
                "        \n",
                "        current_memory = get_memory_usage_mb()\n",
                "        print(f\"Completed batch {batch_start//batch_size + 1}, memory usage: {current_memory:.1f} MB\")\n",
                "        \n",
                "        # Optional: Save intermediate results for very large evaluations\n",
                "        if CONFIG.get('save_intermediate_results', False) and len(all_results) >= 100:\n",
                "            intermediate_file = f\"../output/intermediate_results_batch_{batch_start//batch_size + 1}.json\"\n",
                "            os.makedirs('../output', exist_ok=True)\n",
                "            with open(intermediate_file, 'w') as f:\n",
                "                json.dump(batch_results, f)\n",
                "            print(f\"Saved intermediate results to {intermediate_file}\")\n",
                "    \n",
                "    print(f\"\\nCompleted all batches. Final memory usage: {get_memory_usage_mb():.1f} MB\")\n",
                "    print(f\"Total samples processed: {len(all_results)}\")\n",
                "    \n",
                "    return all_results\n",
                "\n",
                "# Wrapper function to choose between optimized and original implementation\n",
                "def evaluate_samples_batch(sample_indices, include_ruby=True):\n",
                "    \"\"\"Evaluate multiple samples with optional memory optimization\"\"\"\n",
                "    if CONFIG.get('enable_memory_optimization', True):\n",
                "        return evaluate_samples_batch_optimized(sample_indices, include_ruby)\n",
                "    else:\n",
                "        # Fall back to original implementation\n",
                "        print(f\"\\nEvaluating {len(sample_indices)} samples...\")\n",
                "        \n",
                "        # Phase 1: Fast autoencoder inference\n",
                "        print(\"Phase 1: Running autoencoder inference...\")\n",
                "        evaluation_results = []\n",
                "        \n",
                "        for idx in safe_tqdm(sample_indices, desc=\"Autoencoder inference\", leave=True, dynamic_ncols=False):\n",
                "            if idx < len(test_dataset):\n",
                "                sample = test_dataset[idx]\n",
                "                result = evaluate_sample_fast_optimized(sample, idx, include_ruby=False)\n",
                "                evaluation_results.append(result)\n",
                "        \n",
                "        print(f\"Completed autoencoder inference for {len(evaluation_results)} samples\")\n",
                "        \n",
                "        # Phase 2: Ruby code conversion (if enabled)\n",
                "        if include_ruby and CONFIG['enable_ruby_conversion']:\n",
                "            print(\"\\nPhase 2: Converting ASTs to Ruby code...\")\n",
                "            \n",
                "            for result in safe_tqdm(evaluation_results, desc=\"Ruby conversion\", leave=True, dynamic_ncols=False):\n",
                "                # Convert reconstructed AST\n",
                "                reconstructed_code = ast_to_ruby_code_safe(\n",
                "                    result['reconstructed_ast'], \n",
                "                    CONFIG['ruby_timeout']\n",
                "                )\n",
                "                result['reconstructed_code'] = reconstructed_code\n",
                "                if reconstructed_code.startswith('Error:'):\n",
                "                    result['ruby_conversion_error'] = reconstructed_code\n",
                "        \n",
                "        return evaluation_results\n",
                "\n",
                "print(\"Memory-optimized evaluation functions defined\")"
            ]
            
            cell['source'] = optimized_functions
            break
    
    # Save the modified notebook
    with open('notebooks/evaluate_autoencoder_consolidated.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print("Successfully injected memory optimizations into the notebook!")

if __name__ == "__main__":
    inject_memory_optimizations()