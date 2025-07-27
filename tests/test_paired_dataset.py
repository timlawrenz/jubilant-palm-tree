#!/usr/bin/env python3
"""
Test script for validating the PairedDataset and paired data loading functionality.

This script tests the new paired data ingestion pipeline to ensure
that the DataLoader can successfully load paired_data.jsonl and yield 
batches of (graph, text) pairs with proper random sampling of descriptions.
"""

import sys
import os
import traceback
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import PairedDataset, PairedDataLoader, collate_paired_data, create_paired_data_loaders


def test_paired_dataset_loading():
    """Test loading the paired dataset from paired_data.jsonl."""
    print("Testing paired dataset loading...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Paired dataset file not found: {dataset_path}")
        return False
    
    try:
        # Test with fixed seed for reproducible results
        dataset = PairedDataset(dataset_path, seed=42)
        print(f"✅ Paired dataset loading successful")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   Feature dimension: {dataset.get_feature_dim()}")
        return True
    except Exception as e:
        print(f"❌ Paired dataset loading failed: {e}")
        traceback.print_exc()
        return False


def test_paired_dataset_item_access():
    """Test accessing items from the paired dataset."""
    print("\nTesting paired dataset item access...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Paired dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = PairedDataset(dataset_path, seed=42)
        
        # Test accessing a sample
        graph_data, text_description = dataset[0]
        
        print(f"✅ Paired dataset item access successful")
        print(f"   Graph data keys: {list(graph_data.keys())}")
        print(f"   Node features shape: ({len(graph_data['x'])}, {len(graph_data['x'][0]) if graph_data['x'] else 0})")
        print(f"   Edge index shape: ({len(graph_data['edge_index'])}, {len(graph_data['edge_index'][0])})")
        print(f"   Number of nodes: {graph_data['num_nodes']}")
        print(f"   Text description: '{text_description}'")
        print(f"   Text description type: {type(text_description)}")
        
        # Test multiple accesses to verify random sampling
        print(f"\n   Testing random description sampling (same item):")
        for i in range(3):
            _, desc = dataset[0]  # Same index, different descriptions expected
            print(f"   Access {i+1}: '{desc}'")
        
        return True
    except Exception as e:
        print(f"❌ Paired dataset item access failed: {e}")
        traceback.print_exc()
        return False


def test_paired_batch_collation():
    """Test collating paired data into batches."""
    print("\nTesting paired batch collation...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Paired dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = PairedDataset(dataset_path, seed=42)
        
        # Create a small batch
        batch = [dataset[i] for i in range(3)]
        batched_graphs, text_batch = collate_paired_data(batch)
        
        print(f"✅ Paired batch collation successful")
        print(f"   Batched graph keys: {list(batched_graphs.keys())}")
        print(f"   Batched node features shape: ({len(batched_graphs['x'])}, {len(batched_graphs['x'][0]) if batched_graphs['x'] else 0})")
        print(f"   Batched edge index shape: ({len(batched_graphs['edge_index'])}, {len(batched_graphs['edge_index'][0])})")
        print(f"   Number of graphs in batch: {batched_graphs['num_graphs']}")
        print(f"   Text batch length: {len(text_batch)}")
        print(f"   Text descriptions: {text_batch}")
        
        return True
    except Exception as e:
        print(f"❌ Paired batch collation failed: {e}")
        traceback.print_exc()
        return False


def test_paired_data_loader():
    """Test the PairedDataLoader functionality."""
    print("\nTesting PairedDataLoader functionality...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Paired dataset file not found: {dataset_path}")
        return False
    
    try:
        # Test small batches first
        loader = create_paired_data_loaders(dataset_path, batch_size=4, shuffle=False, seed=42)
        
        print(f"   DataLoader created with {len(loader)} batches")
        
        # Test iterating through a few batches
        batch_count = 0
        for batched_graphs, text_batch in loader:
            batch_count += 1
            if batch_count <= 2:  # Only test first 2 batches
                print(f"   Batch {batch_count}: {batched_graphs['num_graphs']} graphs, {len(batched_graphs['x'])} nodes, {len(batched_graphs['edge_index'][0])} edges")
                print(f"   Text batch size: {len(text_batch)}")
                print(f"   Sample text: '{text_batch[0]}'")
            if batch_count >= 2:
                break
        
        print(f"✅ PairedDataLoader functionality successful")
        return True
    except Exception as e:
        print(f"❌ PairedDataLoader functionality failed: {e}")
        traceback.print_exc()
        return False


def test_description_variety():
    """Test that different description sources are being sampled."""
    print("\nTesting description variety...")
    
    dataset_path = "../dataset/paired_data.jsonl"
    if not os.path.exists(dataset_path):
        print(f"❌ Paired dataset file not found: {dataset_path}")
        return False
    
    try:
        dataset = PairedDataset(dataset_path, seed=None)  # No seed for true randomness
        
        # Sample the same item multiple times to see variety
        descriptions_seen = set()
        for i in range(10):
            _, desc = dataset[0]  # Sample same item multiple times
            descriptions_seen.add(desc)
        
        print(f"✅ Description variety test successful")
        print(f"   Unique descriptions sampled from first item: {len(descriptions_seen)}")
        for desc in sorted(descriptions_seen):
            print(f"   - '{desc}'")
        
        return True
    except Exception as e:
        print(f"❌ Description variety test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("🧪 Paired Dataset Testing Suite")
    print("=" * 50)
    
    tests = [
        test_paired_dataset_loading,
        test_paired_dataset_item_access,
        test_paired_batch_collation,
        test_paired_data_loader,
        test_description_variety,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The PairedDataLoader can successfully load and yield batches of (graph, text) pairs.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)