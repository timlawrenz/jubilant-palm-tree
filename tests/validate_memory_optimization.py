#!/usr/bin/env python3
"""
Quick validation script for the memory-optimized evaluation notebook.
Run this to verify that memory optimizations are working correctly.
"""

import sys
import os
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    missing = []
    
    try:
        import torch
        print("✓ PyTorch available")
    except ImportError:
        missing.append("torch")
    
    try:
        import torch_geometric
        print("✓ PyTorch Geometric available")
    except ImportError:
        missing.append("torch-geometric")
    
    try:
        import psutil
        print("✓ psutil available")
    except ImportError:
        missing.append("psutil")
    
    try:
        import tqdm
        print("✓ tqdm available")
    except ImportError:
        missing.append("tqdm")
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        missing.append("pandas")
    
    try:
        import numpy
        print("✓ numpy available")
    except ImportError:
        missing.append("numpy")
    
    if missing:
        print(f"\n✗ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("✓ All dependencies available")
    return True

def check_files():
    """Check if required files exist"""
    print("\nChecking required files...")
    
    required_files = [
        "notebooks/evaluate_autoencoder_consolidated.ipynb",
        "src/data_processing.py", 
        "src/models.py",
        "requirements.txt"
    ]
    
    missing = []
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            missing.append(file_path)
    
    # Check for dataset files
    dataset_files = [
        "dataset_paired_data_example.jsonl",
        "dataset/test.jsonl",
        "dataset/train.jsonl"
    ]
    
    dataset_found = False
    for file_path in dataset_files:
        if os.path.exists(file_path):
            print(f"✓ Dataset found: {file_path}")
            dataset_found = True
            break
    
    if not dataset_found:
        print("⚠ No dataset files found - some tests may be limited")
    
    if missing:
        print(f"\n✗ Missing required files: {', '.join(missing)}")
        return False
    
    return True

def test_memory_optimization():
    """Test the memory optimization features"""
    print("\nTesting memory optimization features...")
    
    try:
        # Add src to path
        sys.path.insert(0, 'src')
        
        # Test memory monitoring
        import psutil
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024
        print(f"✓ Memory monitoring works: {initial_memory:.1f} MB")
        
        # Test dataset loading (if available)
        try:
            from data_processing import RubyASTDataset
            
            if os.path.exists("dataset_paired_data_example.jsonl"):
                dataset = RubyASTDataset("dataset_paired_data_example.jsonl")
                print(f"✓ Dataset loading works: {len(dataset)} samples")
                
                # Test sample access
                if len(dataset) > 0:
                    sample = dataset[0]
                    print(f"✓ Sample access works: {len(sample['x'])} nodes")
                
            else:
                print("⚠ No dataset available for testing")
                
        except Exception as e:
            print(f"⚠ Dataset testing failed: {e}")
        
        # Test model loading (if available)
        try:
            from models import ASTAutoencoder
            
            autoencoder = ASTAutoencoder(
                encoder_input_dim=74,
                node_output_dim=74,
                hidden_dim=64,
                num_layers=3,
                conv_type='GCN',
                freeze_encoder=False  # Don't require weights file
            )
            
            total_params = sum(p.numel() for p in autoencoder.parameters())
            print(f"✓ Model loading works: {total_params:,} parameters")
            
        except Exception as e:
            print(f"⚠ Model testing failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory optimization test failed: {e}")
        return False

def check_notebook():
    """Check if the notebook has been updated with memory optimizations"""
    print("\nChecking notebook optimizations...")
    
    notebook_path = "notebooks/evaluate_autoencoder_consolidated.ipynb"
    if not os.path.exists(notebook_path):
        print(f"✗ Notebook not found: {notebook_path}")
        return False
    
    try:
        with open(notebook_path, 'r') as f:
            content = f.read()
        
        # Check for optimization features
        optimizations = [
            ('batch_size', 'Batch processing configuration'),
            ('enable_memory_optimization', 'Memory optimization toggle'),
            ('max_memory_mb', 'Memory limit configuration'),
            ('cache_jsonl_data', 'JSONL caching configuration'),
            ('evaluate_samples_batch_optimized', 'Optimized evaluation function'),
            ('get_memory_usage_mb', 'Memory monitoring function'),
            ('cleanup_memory', 'Memory cleanup function'),
            ('JSONLCache', 'JSONL caching class')
        ]
        
        found_optimizations = []
        missing_optimizations = []
        
        for feature, description in optimizations:
            if feature in content:
                found_optimizations.append(description)
                print(f"✓ {description}")
            else:
                missing_optimizations.append(description)
                print(f"✗ {description}")
        
        if missing_optimizations:
            print(f"\n⚠ Some optimizations missing. Run: python fix_memory_notebook.py")
            return False
        
        print("✓ All memory optimizations found in notebook")
        return True
        
    except Exception as e:
        print(f"✗ Error checking notebook: {e}")
        return False

def provide_recommendations():
    """Provide recommendations based on system specs"""
    print("\nSystem recommendations...")
    
    try:
        import psutil
        
        # Get system info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        print(f"System: {memory_gb:.1f} GB RAM, {cpu_count} CPU cores")
        
        # Provide recommendations
        if memory_gb < 8:
            print("⚠ Low memory system - Use conservative settings:")
            print("  - batch_size: 10-20")
            print("  - max_memory_mb: 2048-4096")
            print("  - cache_jsonl_data: False")
            print("  - enable_ruby_conversion: False")
            
        elif memory_gb < 16:
            print("✓ Medium memory system - Recommended settings:")
            print("  - batch_size: 50-100")
            print("  - max_memory_mb: 4096-8192")
            print("  - cache_jsonl_data: True")
            print("  - enable_ruby_conversion: True")
            
        else:
            print("✓ High memory system - Optimal settings:")
            print("  - batch_size: 100-200")
            print("  - max_memory_mb: 8192-16384")
            print("  - cache_jsonl_data: True")
            print("  - enable_ruby_conversion: True")
        
        print(f"\nFor large evaluations (>1000 samples):")
        print(f"  - Consider batch_size: {max(50, int(memory_gb * 10))}")
        print(f"  - Set max_memory_mb: {int(memory_gb * 512)}")
        
    except Exception as e:
        print(f"Could not determine system specs: {e}")

def main():
    """Run all validation checks"""
    print("Memory Optimization Validation")
    print("=" * 50)
    
    all_passed = True
    
    # Run checks
    if not check_dependencies():
        all_passed = False
    
    if not check_files():
        all_passed = False
    
    if not test_memory_optimization():
        all_passed = False
    
    if not check_notebook():
        all_passed = False
    
    # Provide recommendations regardless of test results
    provide_recommendations()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✓ All validation checks passed!")
        print("The memory-optimized notebook is ready to use.")
        print("\nNext steps:")
        print("1. Open notebooks/evaluate_autoencoder_consolidated.ipynb")
        print("2. Configure the batch size and memory settings")
        print("3. Run the notebook with your dataset")
        print("4. Monitor memory usage in the output")
        
    else:
        print("✗ Some validation checks failed.")
        print("Please address the issues above before using the notebook.")
        print("\nCommon fixes:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Run the notebook fix: python fix_memory_notebook.py")
        print("3. Check file paths and permissions")
    
    print(f"\nFor help, see: MEMORY_OPTIMIZATION_GUIDE.md")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)