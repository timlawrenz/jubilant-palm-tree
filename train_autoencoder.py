#!/usr/bin/env python3
"""
Training script for AST Autoencoder using Graph Neural Networks.

This script implements the training loop for the ASTAutoencoder model that
reconstructs Ruby method ASTs from learned embeddings. It uses a frozen encoder
and only trains the decoder weights.
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


def train_epoch(model, train_loader, optimizer, device):
    """
    Train the autoencoder for one epoch.
    
    Args:
        model: The ASTAutoencoder model
        train_loader: Training data loader
        optimizer: Optimizer instance
        device: Device to run on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in train_loader:
        # Convert to PyTorch tensors and move to device
        x = torch.tensor(batch['x'], dtype=torch.float).to(device)
        edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
        batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Forward pass through autoencoder
        optimizer.zero_grad()
        result = model(data)
        
        # Compute reconstruction loss (input and target are the same AST)
        loss = ast_reconstruction_loss_simple(data, result['reconstruction'])
        
        # Backward pass (only decoder weights will be updated)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, device):
    """
    Validate the autoencoder for one epoch.
    
    Args:
        model: The ASTAutoencoder model
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Convert to PyTorch tensors and move to device
            x = torch.tensor(batch['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
            batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            # Forward pass
            result = model(data)
            loss = ast_reconstruction_loss_simple(data, result['reconstruction'])
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_decoder_weights(model, filepath, epoch, train_loss, val_loss):
    """
    Save decoder weights and training metadata.
    
    Args:
        model: The autoencoder model
        filepath: Path to save the decoder weights
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
    """
    torch.save({
        'epoch': epoch,
        'decoder_state_dict': model.decoder.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': {
            'embedding_dim': model.decoder.embedding_dim,
            'output_node_dim': model.decoder.output_node_dim,
            'hidden_dim': model.decoder.hidden_dim,
            'num_layers': model.decoder.num_layers,
            'max_nodes': model.decoder.max_nodes
        }
    }, filepath)


def main():
    """Main training function."""
    print("🚀 AST Autoencoder Training")
    print("=" * 50)
    
    # Training configuration
    config = {
        'epochs': 10,  # More than 2 epochs as required
        'batch_size': 16,  # Smaller batch size for autoencoder training
        'learning_rate': 0.001,
        'hidden_dim': 64,
        'num_layers': 3,
        'conv_type': 'GCN',
        'dropout': 0.1,
        'freeze_encoder': True,  # Key requirement: freeze encoder
        'encoder_weights_path': 'best_model.pt'  # Pre-trained encoder weights
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    print("📂 Loading datasets...")
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl",
        "dataset/validation.jsonl", 
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print()
    
    # Initialize autoencoder model
    print("🧠 Initializing AST Autoencoder...")
    model = ASTAutoencoder(
        encoder_input_dim=74,  # AST node feature dimension
        node_output_dim=74,    # Reconstruct same dimension
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        conv_type=config['conv_type'],
        dropout=config['dropout'],
        freeze_encoder=config['freeze_encoder'],
        encoder_weights_path=config['encoder_weights_path']
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"   Model: {model.get_model_info()}")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,} (decoder only)")
    print(f"   Frozen parameters: {frozen_params:,} (encoder)")
    print()
    
    # Setup optimizer (only for decoder parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['learning_rate']
    )
    
    print("⚙️  Training setup:")
    print(f"   Optimizer: Adam (lr={config['learning_rate']}) - decoder only")
    print(f"   Loss function: AST Reconstruction Loss (simple)")
    print(f"   Input/Target: Same AST graph (autoencoder)")
    print()
    
    # Training loop
    print("🏋️  Starting training...")
    print("=" * 50)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print results for each epoch (required by Definition of Done)
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best decoder weights (required by Definition of Done)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_decoder_weights(model, 'best_decoder.pt', epoch, train_loss, val_loss)
            print(f"   💾 New best decoder saved (val_loss: {val_loss:.4f})")
    
    total_time = time.time() - start_time
    
    print("=" * 50)
    print("🎉 Training completed successfully!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best decoder weights saved to: best_decoder.pt")
    
    # Final decoder save
    save_decoder_weights(model, 'final_decoder.pt', config['epochs']-1, train_loss, val_loss)
    print(f"   Final decoder weights saved to: final_decoder.pt")
    
    # Verify training objectives
    print("\n✅ Training Objectives Met:")
    print(f"   ✓ Trained for {config['epochs']} epochs (≥2 required)")
    print(f"   ✓ Only decoder weights trained (encoder frozen)")
    print(f"   ✓ Used AST reconstruction loss function")
    print(f"   ✓ Input and target are same AST graph")
    print(f"   ✓ Best decoder weights saved to best_decoder.pt")
    if config['epochs'] > 1:
        print(f"   ✓ Training completed successfully over multiple epochs")


if __name__ == "__main__":
    main()