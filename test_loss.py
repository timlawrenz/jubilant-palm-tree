#!/usr/bin/env python3
"""
Test script for the AST reconstruction loss functions.

This script tests the loss functions to ensure they work correctly
with torch_geometric.data.Data objects and autoencoder outputs.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset, create_data_loaders
from models import ASTAutoencoder
from loss import ast_reconstruction_loss, ast_reconstruction_loss_simple, compute_node_type_loss


def test_loss_with_identical_data():
    """Test that loss is zero when original and reconstructed are identical."""
    print("🔍 Testing Loss with Identical Data")
    print("-" * 40)
    
    # Create simple test data
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    batch = torch.zeros(3, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create "reconstructed" data with high logits for correct classes
    # If original is [1,0,0], then logits [10,0,0] should give nearly perfect prediction
    recon_logits = torch.tensor([[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]]], dtype=torch.float)
    reconstructed = {
        'node_features': recon_logits,  # These are logits, not one-hot
        'edge_index': edge_index,
        'batch': batch,
        'num_nodes_per_graph': [3]
    }
    
    # Test losses
    loss_full = ast_reconstruction_loss(original, reconstructed)
    loss_simple = ast_reconstruction_loss_simple(original, reconstructed)
    
    print(f"✅ Full loss with identical data: {loss_full.item():.6f}")
    print(f"✅ Simple loss with identical data: {loss_simple.item():.6f}")
    
    # Loss should be very small (near zero) for nearly perfect predictions
    assert loss_full.item() < 0.1, f"Expected loss <0.1, got {loss_full.item()}"
    assert loss_simple.item() < 0.1, f"Expected simple loss <0.1, got {loss_simple.item()}"
    
    return True


def test_loss_with_different_data():
    """Test that loss increases when data is different."""
    print("\n🔍 Testing Loss with Different Data")
    print("-" * 40)
    
    # Original data
    x_orig = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    batch = torch.zeros(3, dtype=torch.long)
    original = Data(x=x_orig, edge_index=edge_index, batch=batch)
    
    # Different reconstructed data (wrong node types)
    x_recon = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
    reconstructed_different = {
        'node_features': x_recon.unsqueeze(0),  # Add batch dimension
        'edge_index': edge_index,
        'batch': batch,
        'num_nodes_per_graph': [3]
    }
    
    # Test losses
    loss_different = ast_reconstruction_loss_simple(original, reconstructed_different)
    
    print(f"✅ Loss with different data: {loss_different.item():.6f}")
    
    # Loss should be larger for different data
    assert loss_different.item() > 0.5, f"Expected loss > 0.5, got {loss_different.item()}"
    
    return True


def test_loss_with_real_autoencoder():
    """Test loss function with real autoencoder output."""
    print("\n🤖 Testing Loss with Real Autoencoder")
    print("-" * 40)
    
    # Load real data
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    print(f"Original AST - Nodes: {x.shape[0]}, Features: {x.shape[1]}, Edges: {edge_index.shape[1]}")
    
    # Create autoencoder and get reconstruction
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    autoencoder.eval()
    
    with torch.no_grad():
        result = autoencoder(original)
        reconstruction = result['reconstruction']
    
    # Compute losses
    loss_full = ast_reconstruction_loss(original, reconstruction)
    loss_simple = ast_reconstruction_loss_simple(original, reconstruction)
    
    print(f"✅ Full reconstruction loss: {loss_full.item():.6f}")
    print(f"✅ Simple reconstruction loss: {loss_simple.item():.6f}")
    
    # Loss should be computable and reasonable
    assert not torch.isnan(loss_full), "Loss should not be NaN"
    assert not torch.isnan(loss_simple), "Simple loss should not be NaN"
    assert loss_full.item() >= 0, "Loss should be non-negative"
    assert loss_simple.item() >= 0, "Simple loss should be non-negative"
    
    print(f"✅ Loss computed successfully for real autoencoder output")
    
    return True


def test_loss_gradient_flow():
    """Test that gradients flow through the loss function."""
    print("\n⚡ Testing Gradient Flow")
    print("-" * 40)
    
    # Create simple test data with gradients
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long).t()
    batch = torch.zeros(2, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create reconstructed data with gradient tracking
    recon_features = torch.tensor([[[0.5, 0.3, 0.2], [0.3, 0.6, 0.1]]], dtype=torch.float, requires_grad=True)
    reconstructed = {
        'node_features': recon_features,
        'edge_index': edge_index,
        'batch': batch,
        'num_nodes_per_graph': [2]
    }
    
    # Compute loss and backpropagate
    loss = ast_reconstruction_loss_simple(original, reconstructed)
    loss.backward()
    
    print(f"✅ Loss value: {loss.item():.6f}")
    print(f"✅ Gradients computed: {recon_features.grad is not None}")
    print(f"✅ Gradient values: {recon_features.grad}")
    
    assert recon_features.grad is not None, "Gradients should be computed"
    assert not torch.isnan(recon_features.grad).any(), "Gradients should not contain NaN"
    
    return True


def test_batch_loss():
    """Test loss function with batched data."""
    print("\n📦 Testing Batched Loss Computation")
    print("-" * 40)
    
    # Create data loaders for batch testing
    train_loader, _ = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=2
    )
    
    # Get a batch
    batch_data = next(iter(train_loader))
    
    # Convert to PyTorch format
    x = torch.tensor(batch_data['x'], dtype=torch.float)
    edge_index = torch.tensor(batch_data['edge_index'], dtype=torch.long)
    batch_idx = torch.tensor(batch_data['batch'], dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    print(f"Batch - Graphs: {batch_data['num_graphs']}, Total nodes: {x.size(0)}")
    
    # Create autoencoder and get reconstruction
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    autoencoder.eval()
    
    with torch.no_grad():
        result = autoencoder(original)
        reconstruction = result['reconstruction']
    
    # Compute losses
    loss_full = ast_reconstruction_loss(original, reconstruction)
    loss_simple = ast_reconstruction_loss_simple(original, reconstruction)
    
    print(f"✅ Batch full loss: {loss_full.item():.6f}")
    print(f"✅ Batch simple loss: {loss_simple.item():.6f}")
    
    # Verify loss is reasonable for batch
    assert not torch.isnan(loss_full), "Batch loss should not be NaN"
    assert not torch.isnan(loss_simple), "Batch simple loss should not be NaN"
    
    return True


def main():
    """Run all loss function tests."""
    print("🧪 AST Reconstruction Loss Testing Suite")
    print("=" * 50)
    
    tests = [
        test_loss_with_identical_data,
        test_loss_with_different_data,
        test_loss_with_real_autoencoder,
        test_loss_gradient_flow,
        test_batch_loss
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
        print("🎉 All loss function tests passed! AST reconstruction loss is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()