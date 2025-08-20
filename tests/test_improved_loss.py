#!/usr/bin/env python3
"""
Test script for the improved AST reconstruction loss function.

This script specifically tests the new ast_reconstruction_loss_improved function
to ensure it works correctly and provides the expected weighted combination of
Type Loss, Edge Loss, Role Loss, and Name Loss components.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from loss import ast_reconstruction_loss_improved, compute_node_type_loss, compute_edge_prediction_loss


def test_improved_loss_basic():
    """Test basic functionality of the improved loss function."""
    print("🔍 Testing Improved Loss Basic Functionality")
    print("-" * 50)
    
    # Create simple test data
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    batch = torch.zeros(3, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create reconstructed data with reasonable predictions
    recon_logits = torch.tensor([[[8.0, -2.0, -2.0], [-2.0, 8.0, -2.0], [-2.0, -2.0, 8.0]]], dtype=torch.float)
    reconstructed = {
        'node_features': recon_logits,
        'edge_index': edge_index,
        'batch': batch,
        'num_nodes_per_graph': [3]
    }
    
    # Test improved loss with default weights
    loss = ast_reconstruction_loss_improved(original, reconstructed)
    
    print(f"✅ Improved loss with default weights: {loss.item():.6f}")
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    
    return True


def test_improved_loss_weighted_components():
    """Test that different weight configurations produce different losses."""
    print("\n🔍 Testing Weighted Components")
    print("-" * 40)
    
    # Create test data
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Remove .t() to get correct shape [2, 1]
    batch = torch.zeros(2, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create imperfect reconstruction
    recon_logits = torch.tensor([[[5.0, 1.0, 1.0], [1.0, 5.0, 1.0]]], dtype=torch.float)
    reconstructed = {
        'node_features': recon_logits,
        'edge_index': edge_index,
    }
    
    # Test with different weight configurations
    loss_default = ast_reconstruction_loss_improved(original, reconstructed)
    
    # High type weight
    loss_high_type = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=10.0, edge_weight=1.0, role_weight=1.0, name_weight=0.1
    )
    
    # High edge weight  
    loss_high_edge = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=1.0, edge_weight=10.0, role_weight=1.0, name_weight=0.1
    )
    
    print(f"✅ Default weights loss: {loss_default.item():.6f}")
    print(f"✅ High type weight loss: {loss_high_type.item():.6f}")
    print(f"✅ High edge weight loss: {loss_high_edge.item():.6f}")
    
    # Verify different configurations give different results
    assert abs(loss_default.item() - loss_high_type.item()) > 1e-6, "Different weights should produce different losses"
    
    return True


def test_improved_loss_components():
    """Test that all four loss components are computed without errors."""
    print("\n🔍 Testing Individual Loss Components")
    print("-" * 45)
    
    # Create test data with multiple nodes for better component testing
    x = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],  # Node type 0
        [0.0, 1.0, 0.0, 0.0],  # Node type 1  
        [0.0, 0.0, 1.0, 0.0],  # Node type 2
        [0.0, 1.0, 0.0, 0.0],  # Node type 1 (repeated for role testing)
    ], dtype=torch.float)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long).t()
    batch = torch.zeros(4, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create reconstructed data
    recon_logits = torch.rand(1, 4, 4) * 2  # Random but reasonable logits
    reconstructed = {
        'node_features': recon_logits,
        'edge_index': edge_index,
    }
    
    # Test each component individually by setting others to zero
    type_only_loss = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=1.0, edge_weight=0.0, role_weight=0.0, name_weight=0.0
    )
    
    edge_only_loss = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=0.0, edge_weight=1.0, role_weight=0.0, name_weight=0.0
    )
    
    role_only_loss = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=0.0, edge_weight=0.0, role_weight=1.0, name_weight=0.0
    )
    
    name_only_loss = ast_reconstruction_loss_improved(
        original, reconstructed, type_weight=0.0, edge_weight=0.0, role_weight=0.0, name_weight=1.0
    )
    
    print(f"✅ Type loss component: {type_only_loss.item():.6f}")
    print(f"✅ Edge loss component: {edge_only_loss.item():.6f}")
    print(f"✅ Role loss component: {role_only_loss.item():.6f}")
    print(f"✅ Name loss component: {name_only_loss.item():.6f}")
    
    # Verify all components are computable
    for loss_name, loss_val in [
        ("Type", type_only_loss), ("Edge", edge_only_loss), 
        ("Role", role_only_loss), ("Name", name_only_loss)
    ]:
        assert not torch.isnan(loss_val), f"{loss_name} loss should not be NaN"
        assert loss_val.item() >= 0, f"{loss_name} loss should be non-negative"
    
    return True


def test_improved_loss_gradient_flow():
    """Test that gradients flow properly through the improved loss function."""
    print("\n🔍 Testing Gradient Flow")
    print("-" * 30)
    
    # Create test data with requires_grad
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)  # Correct shape [2, 1]
    batch = torch.zeros(2, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create reconstructed data with gradients enabled
    recon_logits = torch.tensor([[[2.0, 1.0, 1.0], [1.0, 2.0, 1.0]]], dtype=torch.float, requires_grad=True)
    reconstructed = {
        'node_features': recon_logits,
        'edge_index': edge_index,
    }
    
    # Compute loss and backpropagate
    loss = ast_reconstruction_loss_improved(original, reconstructed)
    loss.backward()
    
    print(f"✅ Loss value: {loss.item():.6f}")
    print(f"✅ Gradients computed: {recon_logits.grad is not None}")
    
    if recon_logits.grad is not None:
        print(f"✅ Gradient values: {recon_logits.grad}")
        assert not torch.isnan(recon_logits.grad).any(), "Gradients should not contain NaN"
    
    return True


def test_improved_loss_backward_compatibility():
    """Test that improved loss works with existing data structures."""
    print("\n🔍 Testing Backward Compatibility")
    print("-" * 40)
    
    # Use the same test pattern as existing loss tests for compatibility
    x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long).t()
    batch = torch.zeros(3, dtype=torch.long)
    original = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test with identical data (should give low loss)
    recon_logits = torch.tensor([[[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0], [-10.0, -10.0, 10.0]]], dtype=torch.float)
    reconstructed = {
        'node_features': recon_logits,
        'edge_index': edge_index,
        'batch': batch,
        'num_nodes_per_graph': [3]
    }
    
    loss = ast_reconstruction_loss_improved(original, reconstructed)
    
    print(f"✅ Loss with near-identical data: {loss.item():.6f}")
    assert loss.item() < 1.0, "Loss should be low for near-identical data"
    
    return True


def main():
    """Run all improved loss function tests."""
    print("🧪 Improved AST Reconstruction Loss Testing Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_improved_loss_basic),
        ("Weighted Components", test_improved_loss_weighted_components), 
        ("Individual Components", test_improved_loss_components),
        ("Gradient Flow", test_improved_loss_gradient_flow),
        ("Backward Compatibility", test_improved_loss_backward_compatibility),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("=" * 60)
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The improved loss function is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()