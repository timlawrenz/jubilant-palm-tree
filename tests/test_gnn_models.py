#!/usr/bin/env python3
"""
Test script for validating the enhanced GNN models.

This script tests both GCN and SAGE convolution types with the DataLoader
to ensure they can process batched data from the Ruby AST dataset.
"""

import sys
import os
import torch
from torch_geometric.data import Data, Batch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import RubyASTDataset, create_data_loaders, collate_graphs
from models import RubyComplexityGNN


def test_single_sample():
    """Test model with a single sample."""
    print("🔍 Testing Single Sample Processing")
    print("-" * 40)
    
    # Load dataset and get a sample
    dataset = RubyASTDataset("../dataset/train.jsonl")
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    y = torch.tensor(sample['y'], dtype=torch.float)
    batch = torch.zeros(x.size(0), dtype=torch.long)  # Single graph
    
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test both model types
    for conv_type in ['GCN', 'SAGE']:
        model = RubyComplexityGNN(input_dim=74, hidden_dim=64, conv_type=conv_type)
        model.eval()
        
        with torch.no_grad():
            output = model(data)
        
        print(f"✅ {conv_type} Model: {model.get_model_info()}")
        print(f"   Output: {output.item():.4f}, Target: {y.item():.4f}")
    
    return True


def test_batch_processing():
    """Test model with batched data from DataLoader."""
    print("\n🔄 Testing Batch Processing with DataLoader")
    print("-" * 40)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "../dataset/train.jsonl", 
        "../dataset/validation.jsonl", 
        batch_size=4
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    
    # Convert to PyTorch format
    x = torch.tensor(batch['x'], dtype=torch.float)
    edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)
    y = torch.tensor(batch['y'], dtype=torch.float)
    batch_idx = torch.tensor(batch['batch'], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    print(f"Batch info:")
    print(f"  Graphs: {batch['num_graphs']}")
    print(f"  Total nodes: {x.size(0)}")
    print(f"  Total edges: {edge_index.size(1)}")
    print(f"  Target values: {y.tolist()}")
    
    # Test both model types
    for conv_type in ['GCN', 'SAGE']:
        model = RubyComplexityGNN(
            input_dim=74, 
            hidden_dim=64, 
            num_layers=3,
            conv_type=conv_type,
            dropout=0.1
        )
        model.eval()
        
        with torch.no_grad():
            output = model(data)
        
        print(f"\n✅ {conv_type} Model Results:")
        print(f"   Model: {model.get_model_info()}")
        print(f"   Output shape: {output.shape}")
        print(f"   Predictions: {[f'{pred.item():.4f}' for pred in output]}")
    
    return True


def test_model_configurations():
    """Test different model configurations."""
    print("\n⚙️  Testing Model Configurations")
    print("-" * 40)
    
    # Load a sample for testing
    dataset = RubyASTDataset("../dataset/train.jsonl")
    sample = dataset[0]
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test different configurations
    configs = [
        {'conv_type': 'GCN', 'num_layers': 2, 'hidden_dim': 32},
        {'conv_type': 'GCN', 'num_layers': 4, 'hidden_dim': 128},
        {'conv_type': 'SAGE', 'num_layers': 2, 'hidden_dim': 32},
        {'conv_type': 'SAGE', 'num_layers': 4, 'hidden_dim': 128},
    ]
    
    for config in configs:
        model = RubyComplexityGNN(input_dim=74, **config)
        model.eval()
        
        with torch.no_grad():
            output = model(data)
        
        print(f"✅ {model.get_model_info()}: output = {output.item():.4f}")
    
    return True


def test_error_handling():
    """Test error handling for invalid configurations."""
    print("\n🚨 Testing Error Handling")
    print("-" * 40)
    
    try:
        model = RubyComplexityGNN(input_dim=74, conv_type='INVALID')
        print("❌ Should have raised ValueError for invalid conv_type")
        return False
    except ValueError as e:
        print(f"✅ Correctly caught error: {e}")
    
    return True


def main():
    """Run all tests."""
    print("🧪 Enhanced GNN Model Testing Suite")
    print("=" * 50)
    
    tests = [
        test_single_sample,
        test_batch_processing,
        test_model_configurations,
        test_error_handling
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Enhanced GNN model is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()