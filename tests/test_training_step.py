#!/usr/bin/env python3
"""
Quick test to validate the training script works after the device fix.
Uses a small subset of data to quickly verify training can proceed.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import create_data_loaders
from models import ASTAutoencoder
from loss import ast_reconstruction_loss_simple


def test_training_step():
    """Test that a single training step works without device errors."""
    print("🏋️  Testing Training Step")
    print("-" * 40)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create small data loaders for quick testing
    try:
        train_loader, _ = create_data_loaders(
            "../dataset/train.jsonl",
            "../dataset/validation.jsonl", 
            batch_size=2,  # Small batch size
            shuffle=True
        )
    except FileNotFoundError:
        print("❌ Dataset files not found, creating synthetic data")
        # Create synthetic data for testing
        x = torch.randn(10, 74)
        edge_index = torch.randint(0, 10, (2, 15))
        batch_idx = torch.zeros(10, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Mock batch format
        batch = {
            'x': x.numpy(),
            'edge_index': edge_index.numpy(), 
            'batch': batch_idx.numpy(),
            'num_graphs': 1
        }
        
        # Mock iterator
        train_loader = [batch]
    
    # Initialize model
    model = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        dropout=0.1,
        freeze_encoder=True
    ).to(device)
    
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    
    print(f"Model on device: {next(model.parameters()).device}")
    
    # Test one training step
    model.train()
    
    try:
        for i, batch in enumerate(train_loader):
            if i >= 1:  # Only test first batch
                break
                
            # Convert to PyTorch tensors and move to device
            x = torch.tensor(batch['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
            batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            print(f"Input data on device: {data.x.device}")
            
            # Forward pass through autoencoder
            optimizer.zero_grad()
            result = model(data)
            
            print("✅ Forward pass successful!")
            print(f"✅ Embedding device: {result['embedding'].device}")
            print(f"✅ Reconstruction device: {result['reconstruction']['node_features'].device}")
            
            # Compute reconstruction loss
            loss = ast_reconstruction_loss_simple(data, result['reconstruction'])
            print(f"✅ Loss computed: {loss.item():.4f}")
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            print("✅ Backward pass successful!")
            print("✅ Training step completed without device errors!")
            
            return True
            
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"❌ Device mismatch error still occurs: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            raise e
    
    return True


def main():
    """Run training validation test."""
    print("🧪 Training Validation Test")
    print("=" * 50)
    
    success = test_training_step()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Training validation successful! Device mismatch issue is fixed.")
    else:
        print("❌ Training validation failed.")
    
    return success


if __name__ == "__main__":
    main()