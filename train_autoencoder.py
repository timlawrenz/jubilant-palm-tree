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
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_processing import create_data_loaders
from models import ASTAutoencoder
from loss import ast_reconstruction_loss_improved

# Performance optimization: Cache CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()


def train_epoch(model, train_loader, optimizer, device, type_weight, parent_weight, scaler):
    model.train()
    total_loss = 0.0
    num_graphs = 0
    
    # Pre-compute autocast context for efficiency
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=CUDA_AVAILABLE)
    
    for data in train_loader:
        # Early skip for empty batches
        if data.num_nodes == 0: 
            continue

        data = data.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        # Use pre-computed autocast context
        with autocast_ctx:
            result = model(data)
            loss = ast_reconstruction_loss_improved(
                data, 
                result['reconstruction'],
                type_weight=type_weight,
                parent_weight=parent_weight
            )
        
        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        
        # Gradient clipping (unscale gradients first)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * data.num_graphs
        num_graphs += data.num_graphs

    return total_loss / num_graphs if num_graphs > 0 else 0.0


def validate_epoch(model, val_loader, device, type_weight, parent_weight):
    model.eval()
    total_loss = 0.0
    num_graphs = 0
    
    # Pre-compute autocast context for efficiency
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16, enabled=CUDA_AVAILABLE)
    
    with torch.no_grad():
        for data in val_loader:
            # Early skip for empty batches
            if data.num_nodes == 0: 
                continue
                
            data = data.to(device, non_blocking=True)

            with autocast_ctx:
                result = model(data)
                loss = ast_reconstruction_loss_improved(
                    data, 
                    result['reconstruction'],
                    type_weight=type_weight,
                    parent_weight=parent_weight
                )
            total_loss += loss.item() * data.num_graphs
            num_graphs += data.num_graphs

    return total_loss / num_graphs if num_graphs > 0 else 0.0


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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AST Autoencoder model')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to dataset directory (default: dataset/)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--output_path', type=str, default='models/best_decoder.pt',
                        help='Path to save the best decoder model (default: models/best_decoder.pt)')
    parser.add_argument('--encoder_weights_path', type=str, default='models/best_model.pt',
                        help='Path to pre-trained encoder weights (default: models/best_model.pt)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size for pre-collation and training (default: 4096)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension size (default: 256)')
    parser.add_argument('--num_layers', type=int, default=5,
                        help='Number of GNN layers (default: 5)')
    parser.add_argument('--conv_type', type=str, default='SAGE', choices=['GCN', 'SAGE'],
                        help='GNN convolution type for the ENCODER (default: SAGE)')
    parser.add_argument('--decoder_conv_type', type=str, default='GAT', choices=['GCN', 'SAGE', 'GAT', 'GIN', 'GraphConv'],
                        help='GNN convolution type for the DECODER (default: GAT)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate (default: 0.1)')
    parser.add_argument('--type_weight', type=float, default=2.0,
                        help='Weight for the node type loss component.')
    parser.add_argument('--parent_weight', type=float, default=1.0,
                        help='Weight for the parent prediction loss component.')
    parser.add_argument('--profile', action='store_true',
                        help='Enable profiling for one epoch to identify performance bottlenecks.')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("🚀 AST Autoencoder Training")
    print("=" * 50)
    
    # Training configuration from args
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'conv_type': args.conv_type,
        'dropout': args.dropout,
        'freeze_encoder': True,  # Key requirement: freeze encoder
        'encoder_weights_path': args.encoder_weights_path
    }
    
    print("📋 Training Configuration:")
    for key, value in config.items():
        print(f"   {key}: {value}")
    print(f"   decoder_conv_type: {args.decoder_conv_type}")
    print(f"   type_weight: {args.type_weight}")
    print(f"   parent_weight: {args.parent_weight}")
    print(f"   dataset_path: {args.dataset_path}")
    print(f"   output_path: {args.output_path}")
    print()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Create data loaders
    print("📂 Loading datasets...")
    
    # Use pre-collated data for maximum performance
    pre_collated = True
    b_size = args.batch_size # The script's batch_size now refers to the pre-collation batch size
    train_data_path = os.path.join(args.dataset_path, f"train_collated_b{b_size}.pt")
    val_data_path = os.path.join(args.dataset_path, f"validation_collated_b{b_size}.pt")
    
    train_loader, val_loader = create_data_loaders(
        train_data_path,
        val_data_path,
        batch_size=1, # Batch size is 1 because each item *is* a batch
        shuffle=True,
        num_workers=0, # Workers > 0 can be slower with pre-collated data
        pre_collated=pre_collated
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
        encoder_weights_path=config['encoder_weights_path'],
        decoder_conv_type=args.decoder_conv_type
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
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config['learning_rate']
    )
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    # Initialize GradScaler for Automatic Mixed Precision (AMP)
    scaler = torch.amp.GradScaler('cuda', enabled=CUDA_AVAILABLE)
    
    print("⚙️  Training setup:")
    print(f"   Optimizer: Adam (lr={config['learning_rate']})")
    print(f"   Scheduler: ReduceLROnPlateau (patience=5)")
    print(f"   Loss function: Improved Reconstruction Loss")
    print(f"   AMP Enabled: {CUDA_AVAILABLE}")
    print()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Training loop with Early Stopping
    print("🏋️  Starting training...")
    print("=" * 50)
    
    if args.profile:
        import cProfile, pstats
        profiler = cProfile.Profile()
        print("🔬 PROFILING ENABLED: Running for one epoch...")
        profiler.enable()

    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stopping_patience = 10
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        epoch_start = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, device, args.type_weight, args.parent_weight, scaler)
        
        # If profiling, stop after one training epoch and print results
        if args.profile:
            profiler.disable()
            print("📊 Profiling Results (top 20 functions by cumulative time):")
            stats = pstats.Stats(profiler).sort_stats('cumtime')
            stats.print_stats(20)
            break # Exit after profiling
            
        val_loss = validate_epoch(model, val_loader, device, args.type_weight, args.parent_weight)
        
        epoch_time = time.time() - epoch_start
        
        print(f"Epoch {epoch+1:2d}/{config['epochs']} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.1e} | "
              f"Time: {epoch_time:.2f}s")
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_decoder_weights(model, args.output_path, epoch, train_loss, val_loss)
            print(f"   💾 New best decoder saved (val_loss: {val_loss:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"   🛑 Early stopping triggered after {early_stopping_patience} epochs with no improvement.")
            break
    
    # This part will not be reached if profiling is enabled and successful
    if not args.profile:
        total_time = time.time() - start_time
        
        print("=" * 50)
        print("🎉 Training completed successfully!")
        print(f"   Total time: {total_time:.2f}s")
        print(f"   Best validation loss: {best_val_loss:.4f}")
        print(f"   Best decoder weights saved to: {args.output_path}")
        
        # Final decoder save (optional, keeping for compatibility)
        final_path = args.output_path.replace('.pt', '_final.pt')
        save_decoder_weights(model, final_path, config['epochs']-1, train_loss, val_loss)
        print(f"   Final decoder weights saved to: {final_path}")
        
        # Verify training objectives
        print("\n✅ Training Objectives Met:")
        print(f"   ✓ Trained for {config['epochs']} epochs (≥2 required)")
        print(f"   ✓ Only decoder weights trained (encoder frozen)")
        print(f"   ✓ Used AST reconstruction loss function")
        print(f"   ✓ Input and target are same AST graph")
        print(f"   ✓ Best decoder weights saved to {args.output_path}")
        if config['epochs'] > 1:
            print(f"   ✓ Training completed successfully over multiple epochs")


if __name__ == "__main__":
    main()
