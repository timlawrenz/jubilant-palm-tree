#!/usr/bin/env python3
"""
Integration test for the memory-optimized notebook.
This script executes key cells from the notebook to ensure they work correctly.
"""

import sys
import os
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

def setup_test_environment():
    """Set up the test environment similar to the notebook"""
    
    # Mock configuration for testing
    CONFIG = {
        'num_samples': 5,  # Small number for testing
        'random_seed': 42,
        'enable_ruby_conversion': False,  # Disable Ruby for faster testing
        'ruby_timeout': 15,
        'save_results': False,
        'show_comparisons': 2,
        'batch_size': 2,  # Small batch size
        'enable_memory_optimization': True,
        'max_memory_mb': 1024,
        'cache_jsonl_data': True,
    }
    
    # Set random seeds
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    return CONFIG

def test_notebook_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        # Test basic imports (similar to notebook)
        import time
        import gc
        import psutil
        from tqdm import tqdm
        
        # Test ML imports
        import torch
        import pandas as pd
        from torch_geometric.data import Data
        import numpy as np
        
        # Test custom imports
        from data_processing import RubyASTDataset
        from models import ASTAutoencoder
        
        print("✓ All imports successful")
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_dataset_loading():
    """Test dataset loading with small sample"""
    print("\nTesting dataset loading...")
    
    try:
        # Use the example dataset since test.jsonl is in LFS
        dataset_path = "dataset_paired_data_example.jsonl"
        if not os.path.exists(dataset_path):
            print(f"✗ Dataset not found: {dataset_path}")
            return False
        
        # Try to load a small subset
        from data_processing import RubyASTDataset
        
        # Load dataset
        test_dataset = RubyASTDataset(dataset_path)
        
        if len(test_dataset) == 0:
            print("✗ No samples loaded from dataset")
            return False
        
        print(f"✓ Loaded {len(test_dataset)} samples")
        
        # Test sample access
        sample = test_dataset[0]
        print(f"✓ Sample 0: {len(sample['x'])} nodes, feature shape: {np.array(sample['x']).shape if len(sample['x']) > 0 else 'no features'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_memory_functions():
    """Test memory monitoring functions"""
    print("\nTesting memory functions...")
    
    try:
        import psutil
        import gc
        
        # Test memory usage function
        def get_memory_usage_mb():
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        
        # Test cleanup function
        def cleanup_memory():
            gc.collect()
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        
        # Test usage
        initial_memory = get_memory_usage_mb()
        print(f"✓ Initial memory: {initial_memory:.1f} MB")
        
        # Create some data and clean up
        big_data = [torch.randn(1000, 1000) for _ in range(10)]
        memory_after = get_memory_usage_mb()
        print(f"✓ Memory after allocation: {memory_after:.1f} MB")
        
        del big_data
        cleanup_memory()
        memory_after_cleanup = get_memory_usage_mb()
        print(f"✓ Memory after cleanup: {memory_after_cleanup:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory functions test failed: {e}")
        return False

def test_jsonl_cache():
    """Test JSONL caching functionality"""
    print("\nTesting JSONL cache...")
    
    try:
        class JSONLCache:
            def __init__(self, jsonl_path):
                self.jsonl_path = jsonl_path
                self.cache = {}
                self.loaded = False
            
            def get_sample_data(self, idx):
                if not self.loaded:
                    self._load_cache()
                return self.cache.get(idx, None)
            
            def _load_cache(self):
                try:
                    with open(self.jsonl_path, 'r') as f:
                        for idx, line in enumerate(f):
                            if idx >= 10:  # Only cache first 10 for testing
                                break
                            line = line.strip()
                            if line:
                                data_dict = json.loads(line)
                                self.cache[idx] = {
                                    'raw_source': data_dict.get('method_source', ''),
                                    'ast_json': data_dict.get('ast_json', '{}')
                                }
                    self.loaded = True
                    print(f"✓ Cached {len(self.cache)} samples")
                except Exception as e:
                    print(f"Cache loading failed: {e}")
        
        # Test with example dataset
        cache = JSONLCache("dataset_paired_data_example.jsonl")
        
        # Test accessing data
        for i in range(3):
            data = cache.get_sample_data(i)
            if data:
                print(f"✓ Sample {i}: {len(data['raw_source'])} chars")
            else:
                print(f"✗ Sample {i}: No data")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ JSONL cache test failed: {e}")
        return False

def test_autoencoder_loading():
    """Test autoencoder initialization"""
    print("\nTesting autoencoder loading...")
    
    try:
        from models import ASTAutoencoder
        
        # Initialize autoencoder (similar to notebook)
        autoencoder = ASTAutoencoder(
            encoder_input_dim=74,
            node_output_dim=74,
            hidden_dim=64,
            num_layers=3,
            conv_type='GCN',
            freeze_encoder=True,
            encoder_weights_path="models/best_model.pt" if os.path.exists("models/best_model.pt") else None
        )
        
        autoencoder.eval()
        
        # Test basic info
        total_params = sum(p.numel() for p in autoencoder.parameters())
        trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
        
        print(f"✓ Autoencoder loaded: {total_params:,} total params ({trainable_params:,} trainable)")
        
        return True
        
    except Exception as e:
        print(f"✗ Autoencoder loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run integration tests"""
    print("Running notebook integration tests...")
    print("=" * 60)
    
    # Setup
    CONFIG = setup_test_environment()
    print(f"Configuration: {CONFIG}")
    
    all_passed = True
    
    # Run tests
    tests = [
        test_notebook_imports,
        test_memory_functions,
        test_jsonl_cache,
        test_dataset_loading,
        test_autoencoder_loading,
    ]
    
    for test_func in tests:
        try:
            if not test_func():
                all_passed = False
        except Exception as e:
            print(f"✗ Test {test_func.__name__} failed with exception: {e}")
            all_passed = False
    
    print("=" * 60)
    if all_passed:
        print("✓ All integration tests passed!")
        print("The memory-optimized notebook should work correctly.")
    else:
        print("✗ Some tests failed.")
        print("Please check the implementation before running the full notebook.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)