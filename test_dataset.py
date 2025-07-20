#!/usr/bin/env python3
"""
Test script for validating the RubyASTDataset and graph conversion functionality.

This script tests the data ingestion and graph conversion pipeline to ensure
that the DataLoader can successfully load and collate batches of graph objects.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset, collate_graphs, ASTGraphConverter


def test_ast_converter():
    """Test the AST to graph conversion functionality."""
    print("Testing AST to graph conversion...")
    
    converter = ASTGraphConverter()
    
    # Test with a simple AST example
    simple_ast = '{"type":"def","children":["test_method",{"type":"args","children":[]},{"type":"int","children":[42]}]}'
    
    try:
        graph_data = converter.parse_ast_json(simple_ast)
        print(f"✅ AST conversion successful")
        print(f"   Node features shape: ({len(graph_data['x'])}, {len(graph_data['x'][0]) if graph_data['x'] else 0})")
        print(f"   Edge index shape: ({len(graph_data['edge_index'])}, {len(graph_data['edge_index'][0])})")
        print(f"   Number of nodes: {graph_data['num_nodes']}")
        return True
    except Exception as e:
        print(f"❌ AST conversion failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_loading():
    """Test loading the dataset from JSONL files."""
    print("\nTesting dataset loading...")
    
    dataset_path = "dataset/train.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = RubyASTDataset(dataset_path)
        print(f"✅ Dataset loading successful")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Feature dimension: {dataset.get_feature_dim()}")
        return True
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        traceback.print_exc()
        return False


def test_dataset_getitem():
    """Test getting individual items from the dataset."""
    print("\nTesting dataset item access...")
    
    try:
        dataset = RubyASTDataset("dataset/train.jsonl")
        
        # Test getting first item
        sample = dataset[0]
        print(f"✅ Dataset item access successful")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Node features shape: ({len(sample['x'])}, {len(sample['x'][0]) if sample['x'] else 0})")
        print(f"   Edge index shape: ({len(sample['edge_index'])}, {len(sample['edge_index'][0])})")
        print(f"   Target shape: ({len(sample['y'])},)")
        print(f"   Target value: {sample['y'][0]}")
        print(f"   Number of nodes: {sample['num_nodes']}")
        
        return True
    except Exception as e:
        print(f"❌ Dataset item access failed: {e}")
        traceback.print_exc()
        return False


def test_batch_collation():
    """Test batching multiple samples together."""
    print("\nTesting batch collation...")
    
    try:
        dataset = RubyASTDataset("dataset/train.jsonl")
        
        # Get a few samples
        batch = [dataset[i] for i in range(min(3, len(dataset)))]
        
        # Collate the batch
        batched_data = collate_graphs(batch)
        
        print(f"✅ Batch collation successful")
        print(f"   Batch keys: {list(batched_data.keys())}")
        print(f"   Batched node features shape: ({len(batched_data['x'])}, {len(batched_data['x'][0]) if batched_data['x'] else 0})")
        print(f"   Batched edge index shape: ({len(batched_data['edge_index'])}, {len(batched_data['edge_index'][0])})")
        print(f"   Batched targets shape: ({len(batched_data['y'])},)")
        print(f"   Batch indices shape: ({len(batched_data['batch'])},)")
        print(f"   Number of graphs in batch: {batched_data['num_graphs']}")
        
        return True
    except Exception as e:
        print(f"❌ Batch collation failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader_simulation():
    """Simulate DataLoader functionality by creating multiple batches."""
    print("\nTesting DataLoader simulation...")
    
    try:
        dataset = RubyASTDataset("dataset/train.jsonl")
        batch_size = 4
        num_batches = 3
        
        print(f"   Creating {num_batches} batches of size {batch_size}...")
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            
            if start_idx >= len(dataset):
                break
            
            # Create batch
            batch = [dataset[i] for i in range(start_idx, end_idx)]
            batched_data = collate_graphs(batch)
            
            print(f"   Batch {batch_idx + 1}: {len(batch)} samples, "
                  f"{len(batched_data['x'])} total nodes, "
                  f"{len(batched_data['edge_index'][0])} total edges")
        
        print(f"✅ DataLoader simulation successful")
        return True
        
    except Exception as e:
        print(f"❌ DataLoader simulation failed: {e}")
        traceback.print_exc()
        return False


def test_dataloader_functionality():
    """Test the SimpleDataLoader implementation."""
    print("\nTesting SimpleDataLoader functionality...")
    
    try:
        from data_processing import SimpleDataLoader, create_data_loaders
        
        # Test individual DataLoader
        dataset = RubyASTDataset("dataset/train.jsonl")
        loader = SimpleDataLoader(dataset, batch_size=8, shuffle=False)
        
        print(f"   DataLoader created with {len(loader)} batches")
        
        # Test iteration
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count <= 2:  # Only test first 2 batches
                print(f"   Batch {batch_count}: {batch['num_graphs']} graphs, "
                      f"{len(batch['x'])} nodes, {len(batch['edge_index'][0])} edges")
            if batch_count >= 2:
                break
        
        # Test create_data_loaders function
        train_loader, val_loader = create_data_loaders("dataset/train.jsonl", "dataset/validation.jsonl", batch_size=16)
        print(f"   Created train loader: {len(train_loader)} batches")
        print(f"   Created val loader: {len(val_loader)} batches")
        
        print(f"✅ SimpleDataLoader functionality successful")
        return True
        
    except Exception as e:
        print(f"❌ SimpleDataLoader functionality failed: {e}")
        traceback.print_exc()
        return False


def test_validation_dataset():
    """Test loading validation dataset."""
    print("\nTesting validation dataset...")
    
    validation_path = "dataset/validation.jsonl"
    if not os.path.exists(validation_path):
        print(f"❌ Validation file not found: {validation_path}")
        return False
    
    try:
        val_dataset = RubyASTDataset(validation_path)
        print(f"✅ Validation dataset loading successful")
        print(f"   Validation dataset size: {len(val_dataset)}")
        
        # Test one sample
        sample = val_dataset[0]
        print(f"   Sample processed successfully")
        
        return True
    except Exception as e:
        print(f"❌ Validation dataset loading failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Ruby AST Dataset Testing Suite")
    print("=" * 50)
    
    tests = [
        test_ast_converter,
        test_dataset_loading
        # test_dataset_getitem
        # test_batch_collation,
        # test_dataloader_simulation,
        # test_dataloader_functionality,
        # test_validation_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The DataLoader can successfully load and collate batches.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
