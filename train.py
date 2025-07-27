#!/usr/bin/env python3
"""
Training script for Ruby complexity prediction using Graph Neural Networks.

This script implements the main training and validation loop for the GNN model
that predicts Ruby method complexity based on AST structure.
"""

import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_data_loaders
from models import RubyComplexityGNN


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train the model for one epoch.
    
    Args:
        model: The GNN model
        train_loader: Training data loader
        optimizer: Optimizer instance
        criterion: Loss function
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
        y = torch.tensor(batch['y'], dtype=torch.float).to(device)
        batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions.squeeze(), y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, criterion, device):
    """
    Validate the model for one epoch.
    
    Args:
        model: The GNN model
        val_loader: Validation data loader
        criterion: Loss function
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
            y = torch.tensor(batch['y'], dtype=torch.float).to(device)
            batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            # Forward pass
            predictions = model(data)
            loss = criterion(predictions.squeeze(), y)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def save_model(model, filepath, epoch, train_loss, val_loss):
    """
    Save model weights and training metadata.
    
    Args:
        model: The model to save
        filepath: Path to save the model
        epoch: Current epoch number
        train_loss: Training loss
        val_loss: Validation loss
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_config': {
            'input_dim': 74,
            'hidden_dim': model.convs[0].out_channels if hasattr(model.convs[0], 'out_channels') else 64,
            'num_layers': model.num_layers,
            'conv_type': model.conv_type,
            'dropout': model.dropout
        }
    }, filepath)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Ruby complexity prediction GNN model')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to dataset directory (default: dataset/)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--output_path', type=str, default='models/best_model.pt',
                        help='Path to save the best model (default: models/best_model.pt)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension size (default: 64)')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of GNN layers (default: 3)')
    parser.add_argument('--conv_type', type=str, default='SAGE', choices=['GCN', 'SAGE'],
                        help='GNN convolution type (default: SAGE)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("🚀 Ruby Complexity GNN Training")
    print("=" * 50)
    
    # Training configuration from args
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'conv_type': args.conv_type,
        'dropout': args.dropout
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print(f"   dataset_path: {args.dataset_path}")
    print(f"   output_path: {args.output_path}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    print("📂 Loading datasets...")
    
    # Handle sample dataset naming convention
    if args.dataset_path.rstrip('/').endswith('samples'):
        train_data_path = os.path.join(args.dataset_path, "train_sample.jsonl")
        val_data_path = os.path.join(args.dataset_path, "validation_sample.jsonl")
    else:
        train_data_path = os.path.join(args.dataset_path, "train.jsonl")
        val_data_path = os.path.join(args.dataset_path, "validation.jsonl")
    
    train_loader, val_loader = create_data_loaders(
        train_data_path,
        val_data_path,
        batch_size=config['batch_size'],
        shuffle=True
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    print()
    
    # Initialize model
    print("🧠 Initializing model...")
    model = RubyComplexityGNN(
        input_dim=74,  # AST node feature dimension
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        conv_type=config['conv_type'],
        dropout=config['dropout']
    ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Model: {model.get_model_info()}")
    print(f"   Parameters: {param_count:,}")
    print()
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = torch.nn.MSELoss()
    
    print("⚙️  Training setup:")
    print(f"   Optimizer: Adam (lr={config['learning_rate']})")
    print(f"   Loss function: MSELoss")
    print()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Training loop
    print("🏋️  Starting training...")
    print("=" * 50)
    
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        epoch_time = time.time() - epoch_start
        
        # Print results for each epoch (required by Definition of Done)
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Time: {epoch_time:.2f}s")
        
        # Save best model (required by Definition of Done)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_model(model, args.output_path, epoch, train_loss, val_loss)
            print(f"   💾 New best model saved (val_loss: {val_loss:.4f})")
    
    total_time = time.time() - start_time
    
    print("=" * 50)
    print("🎉 Training completed successfully!")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Best model saved to: {args.output_path}")
    
    # Final model save (optional, keeping for compatibility)
    final_path = args.output_path.replace('.pt', '_final.pt')
    save_model(model, final_path, config['epochs']-1, train_loss, val_loss)
    print(f"   Final model saved to: {final_path}")


if __name__ == "__main__":
    main()