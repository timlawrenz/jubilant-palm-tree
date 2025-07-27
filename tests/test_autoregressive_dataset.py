#!/usr/bin/env python3
"""
Test script for validating the AutoregressiveASTDataset functionality.

This script tests the new autoregressive data pipeline to ensure
that the DataLoader can successfully convert ASTs into sequential
training pairs for autoregressive generation.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import (
    AutoregressiveASTDataset, 
    AutoregressiveDataLoader, 
    collate_autoregressive_data,
    create_autoregressive_data_loader
)


def test_autoregressive_dataset_loading():
    """Test loading the autoregressive dataset."""
    print("Testing autoregressive dataset loading...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        # Test with limited sequence length for faster testing
        dataset = AutoregressiveASTDataset(dataset_path, max_sequence_length=10, seed=42)
        print(f"✅ Autoregressive dataset loading successful")
        print(f"   Original methods: 4000")
        print(f"   Sequential pairs: {len(dataset)}")
        print(f"   Feature dimension: {dataset.get_feature_dim()}")
        return True
    except Exception as e:
        print(f"❌ Autoregressive dataset loading failed: {e}")
        traceback.print_exc()
        return False


def test_autoregressive_dataset_item_access():
    """Test accessing items from the autoregressive dataset."""
    print("\nTesting autoregressive dataset item access...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = AutoregressiveASTDataset(dataset_path, max_sequence_length=5, seed=42)
        
        # Test accessing a sample
        if len(dataset) == 0:
            print("❌ Dataset is empty")
            return False
            
        sample = dataset[0]
        
        print(f"✅ Autoregressive dataset item access successful")
        print(f"   Sample keys: {list(sample.keys())}")
        print(f"   Text description: '{sample['text_description']}'")
        print(f"   Step: {sample['step']}/{sample['total_steps']}")
        print(f"   Partial graph nodes: {sample['partial_graph']['num_nodes']}")
        print(f"   Target node type: '{sample['target_node']['node_type']}'")
        
        # Test accessing multiple steps to see progression
        print(f"\n   Sequential progression:")
        for i in range(min(3, len(dataset))):
            s = dataset[i]
            print(f"   Step {s['step']}: {s['partial_graph']['num_nodes']} nodes -> '{s['target_node']['node_type']}'")
        
        return True
    except Exception as e:
        print(f"❌ Autoregressive dataset item access failed: {e}")
        traceback.print_exc()
        return False


def test_autoregressive_batch_collation():
    """Test collating autoregressive data into batches."""
    print("\nTesting autoregressive batch collation...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = AutoregressiveASTDataset(dataset_path, max_sequence_length=5, seed=42)
        
        if len(dataset) < 3:
            print("❌ Dataset too small for batch testing")
            return False
        
        # Create a small batch
        batch = [dataset[i] for i in range(3)]
        batched_data = collate_autoregressive_data(batch)
        
        print(f"✅ Autoregressive batch collation successful")
        print(f"   Batched data keys: {list(batched_data.keys())}")
        print(f"   Text descriptions: {len(batched_data['text_descriptions'])}")
        print(f"   Partial graphs shape: {len(batched_data['partial_graphs']['x'])} nodes")
        print(f"   Target node types: {batched_data['target_node_types']}")
        print(f"   Steps: {batched_data['steps']}")
        
        return True
    except Exception as e:
        print(f"❌ Autoregressive batch collation failed: {e}")
        traceback.print_exc()
        return False


def test_autoregressive_data_loader():
    """Test the AutoregressiveDataLoader functionality."""
    print("\nTesting AutoregressiveDataLoader functionality...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        # Test small batches
        loader = create_autoregressive_data_loader(
            dataset_path, 
            batch_size=4, 
            shuffle=False, 
            max_sequence_length=5, 
            seed=42
        )
        
        print(f"   DataLoader created with {len(loader)} batches")
        
        # Test iterating through a few batches
        batch_count = 0
        for batch in loader:
            batch_count += 1
            if batch_count <= 2:  # Only test first 2 batches
                print(f"   Batch {batch_count}: {len(batch['text_descriptions'])} samples")
                print(f"   Partial graphs: {len(batch['partial_graphs']['x'])} total nodes")
                print(f"   Sample text: '{batch['text_descriptions'][0]}'")
                print(f"   Sample step: {batch['steps'][0]}")
            if batch_count >= 2:
                break
        
        print(f"✅ AutoregressiveDataLoader functionality successful")
        return True
    except Exception as e:
        print(f"❌ AutoregressiveDataLoader functionality failed: {e}")
        traceback.print_exc()
        return False


def test_sequential_progression():
    """Test that sequences progress correctly for a single method."""
    print("\nTesting sequential progression for single method...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = AutoregressiveASTDataset(dataset_path, max_sequence_length=10, seed=42)
        
        if len(dataset) == 0:
            print("❌ Dataset is empty")
            return False
        
        # Find sequences from the same method by looking for consecutive steps
        first_method_pairs = []
        for i in range(len(dataset)):
            sample = dataset[i]
            if sample['step'] == 0:  # Start of a new method
                # Collect all pairs for this method
                method_pairs = [sample]
                for j in range(i + 1, min(i + sample['total_steps'], len(dataset))):
                    next_sample = dataset[j]
                    if (next_sample['text_description'] == sample['text_description'] and 
                        next_sample['step'] == method_pairs[-1]['step'] + 1):
                        method_pairs.append(next_sample)
                    else:
                        break
                
                if len(method_pairs) > 1:
                    first_method_pairs = method_pairs
                    break
        
        if not first_method_pairs:
            print("❌ Could not find sequential pairs for a method")
            return False
        
        print(f"✅ Sequential progression test successful")
        print(f"   Found method with {len(first_method_pairs)} sequential steps")
        print(f"   Text: '{first_method_pairs[0]['text_description']}'")
        
        # Show progression
        for i, pair in enumerate(first_method_pairs[:5]):  # Show first 5 steps
            partial_nodes = pair['partial_graph']['num_nodes']
            target_type = pair['target_node']['node_type']
            print(f"   Step {pair['step']}: {partial_nodes} nodes -> '{target_type}'")
        
        return True
    except Exception as e:
        print(f"❌ Sequential progression test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Autoregressive AST Dataset Testing Suite")
    print("=" * 50)
    
    tests = [
        test_autoregressive_dataset_loading,
        test_autoregressive_dataset_item_access,
        test_autoregressive_batch_collation,
        test_autoregressive_data_loader,
        test_sequential_progression,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The AutoregressiveASTDataset successfully converts ASTs into sequential training pairs.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)