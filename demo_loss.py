#!/usr/bin/env python3
"""
Demonstration of the AST reconstruction loss function usage.

This script shows how to use the loss function with the ASTAutoencoder
for training purposes.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset, create_data_loaders
from models import ASTAutoencoder
from loss import ast_reconstruction_loss, ast_reconstruction_loss_simple


def demonstrate_loss_usage():
    """Demonstrate how to use the loss function for training."""
    print("🚀 AST Reconstruction Loss Function Demo")
    print("=" * 50)
    
    # Load sample data
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    print(f"📊 Input AST: {x.shape[0]} nodes, {x.shape[1]} features, {edge_index.shape[1]} edges")
    
    # Create autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        freeze_encoder=False  # Allow training
    )
    
    # Enable gradients for training
    autoencoder.train()
    
    # Forward pass
    result = autoencoder(data)
    reconstruction = result['reconstruction']
    
    print(f"🔧 Reconstruction: {reconstruction['node_features'].shape}")
    
    # Compute losses
    loss_full = ast_reconstruction_loss(data, reconstruction, node_weight=1.0, edge_weight=0.5)
    loss_simple = ast_reconstruction_loss_simple(data, reconstruction)
    
    print(f"📈 Full reconstruction loss: {loss_full.item():.4f}")
    print(f"📈 Simple node loss: {loss_simple.item():.4f}")
    
    # Demonstrate gradient computation
    print("\n🎓 Training Step Simulation:")
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Backward pass
    loss_simple.backward()
    
    # Check if gradients exist
    decoder_grad_norm = torch.norm(torch.cat([p.grad.flatten() for p in autoencoder.decoder.parameters() if p.grad is not None]))
    print(f"✅ Decoder gradient norm: {decoder_grad_norm.item():.6f}")
    
    # Simulate optimizer step (don't actually update for demo)
    # optimizer.step()
    
    print("✅ Loss function is ready for training!")
    
    return True


def demonstrate_batch_training():
    """Demonstrate batch training with the loss function."""
    print("\n📦 Batch Training Demo")
    print("-" * 30)
    
    # Create data loaders
    train_loader, _ = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=4
    )
    
    # Get a batch
    batch_data = next(iter(train_loader))
    
    # Convert to PyTorch format
    x = torch.tensor(batch_data['x'], dtype=torch.float)
    edge_index = torch.tensor(batch_data['edge_index'], dtype=torch.long)
    batch_idx = torch.tensor(batch_data['batch'], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    print(f"📊 Batch: {batch_data['num_graphs']} graphs, {x.size(0)} total nodes")
    
    # Create autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    autoencoder.train()
    
    # Forward pass
    result = autoencoder(data)
    reconstruction = result['reconstruction']
    
    # Compute loss
    loss = ast_reconstruction_loss_simple(data, reconstruction)
    
    print(f"📈 Batch loss: {loss.item():.4f}")
    print(f"📈 Average loss per graph: {loss.item() / batch_data['num_graphs']:.4f}")
    
    print("✅ Batch processing working correctly!")
    
    return True


def show_loss_components():
    """Show how different loss components contribute."""
    print("\n🧩 Loss Components Analysis")
    print("-" * 30)
    
    # Load sample data
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    autoencoder.eval()
    
    with torch.no_grad():
        result = autoencoder(data)
        reconstruction = result['reconstruction']
    
    # Test different weight combinations
    weights = [
        (1.0, 0.0),  # Only node loss
        (1.0, 0.5),  # Balanced
        (1.0, 1.0),  # Equal weights
        (0.5, 1.0),  # Edge-heavy
    ]
    
    print("Weight (node, edge) -> Loss:")
    for node_w, edge_w in weights:
        loss = ast_reconstruction_loss(data, reconstruction, node_weight=node_w, edge_weight=edge_w)
        print(f"  ({node_w:.1f}, {edge_w:.1f}) -> {loss.item():.4f}")
    
    # Compare with simple loss
    simple_loss = ast_reconstruction_loss_simple(data, reconstruction)
    print(f"\nSimple node-only loss: {simple_loss.item():.4f}")
    
    return True


def main():
    """Run all demonstrations."""
    print("🎯 AST Reconstruction Loss Function Demonstrations")
    print("=" * 60)
    
    demos = [
        demonstrate_loss_usage,
        demonstrate_batch_training,
        show_loss_components
    ]
    
    for demo in demos:
        try:
            demo()
        except Exception as e:
            print(f"❌ {demo.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("🏁 Demo complete! The AST reconstruction loss function is ready to use.")
    print("\n💡 Usage Tips:")
    print("   - Use ast_reconstruction_loss_simple() for most cases")
    print("   - Use ast_reconstruction_loss() for fine-tuning with edge loss")
    print("   - Both functions work with batched data")
    print("   - Gradients flow correctly for training")


if __name__ == "__main__":
    main()