#!/usr/bin/env python3
"""
Short training test to validate the fix works with minimal epochs.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_data_loaders
from models import ASTAutoencoder
from loss import ast_reconstruction_loss_simple


def train_epoch_short(model, train_loader, optimizer, device, max_batches=3):
    """Train for a few batches to validate the fix."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        if num_batches >= max_batches:
            break
            
        # Convert to PyTorch tensors and move to device
        x = torch.tensor(batch['x'], dtype=torch.float).to(device)
        edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
        batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Forward pass through autoencoder
        optimizer.zero_grad()
        result = model(data)
        
        # Compute reconstruction loss
        loss = ast_reconstruction_loss_simple(data, result['reconstruction'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        print(f"   Batch {num_batches}: loss={loss.item():.4f}")
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def test_short_training():
    """Test training with minimal configuration."""
    print("🏋️  Short Training Test")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    print("📂 Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl",
        "dataset/validation.jsonl", 
        batch_size=4,  # Small batch size for speed
        shuffle=True
    )
    
    print(f"   Using small batch size for testing")
    
    # Initialize autoencoder model
    print("🧠 Initializing AST Autoencoder...")
    model = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        dropout=0.1,
        freeze_encoder=True,
        encoder_weights_path='best_model.pt'
    ).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=0.001
    )
    
    print("🏋️  Starting short training test...")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        # Train for just a few batches
        train_loss = train_epoch_short(model, train_loader, optimizer, device, max_batches=3)
        
        end_time = time.time()
        
        print(f"✅ Training completed successfully!")
        print(f"   Average loss: {train_loss:.4f}")
        print(f"   Time taken: {end_time - start_time:.2f}s")
        
        return True
        
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"❌ Device mismatch error still occurs: {e}")
            return False
        else:
            print(f"❌ Other error: {e}")
            raise e


def main():
    """Run short training test."""
    success = test_short_training()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Short training test successful! The device mismatch issue is fixed.")
        print("✅ The full training script should now work properly.")
    else:
        print("❌ Short training test failed.")
    
    return success


if __name__ == "__main__":
    main()