#!/usr/bin/env python3
"""
Training script for text-code alignment using Graph Neural Networks and contrastive learning.

This script implements the training loop for the AlignmentModel that learns to align
text descriptions with code embeddings in a shared 64-dimensional space. It uses:
- A frozen RubyComplexityGNN (code encoder) to preserve learned AST representations
- A trainable text encoder with projection head to map text to the code embedding space
- InfoNCE contrastive loss to encourage alignment between matching code-text pairs
"""

import sys
import os
import time
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_paired_data_loaders
from models import AlignmentModel
from loss import info_nce_loss


def train_epoch(model, train_loader, optimizer, device):
    """
    Train the alignment model for one epoch.
    
    Args:
        model: The AlignmentModel
        train_loader: Training data loader (yields graph-text pairs)
        optimizer: Optimizer instance (only updates trainable text projection head)
        device: Device to run on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batched_graphs, text_descriptions in progress_bar:
        # Convert graph data to PyTorch tensors and move to device
        x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
        edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
        batch_idx = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
        
        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        # Forward pass through alignment model
        optimizer.zero_grad()
        outputs = model(data, text_descriptions)
        
        # Extract embeddings
        code_embeddings = outputs['code_embeddings']
        text_embeddings = outputs['text_embeddings']
        
        # Compute contrastive loss (InfoNCE)
        loss = info_nce_loss(code_embeddings, text_embeddings)
        
        # Backward pass (only text projection head weights will be updated)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate_epoch(model, val_loader, device):
    """
    Validate the alignment model for one epoch.
    
    Args:
        model: The AlignmentModel
        val_loader: Validation data loader
        device: Device to run on
        
    Returns:
        Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating", leave=False)
        
        for batched_graphs, text_descriptions in progress_bar:
            # Convert graph data to PyTorch tensors and move to device
            x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
            batch_idx = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            # Forward pass
            outputs = model(data, text_descriptions)
            
            # Extract embeddings
            code_embeddings = outputs['code_embeddings']
            text_embeddings = outputs['text_embeddings']
            
            # Compute contrastive loss
            loss = info_nce_loss(code_embeddings, text_embeddings)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'val_loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def calculate_alignment_metrics(model, val_loader, device, max_batches=10):
    """
    Calculate alignment metrics to monitor training progress.
    
    Args:
        model: The AlignmentModel
        val_loader: Validation data loader
        device: Device to run on
        max_batches: Maximum number of batches to evaluate (for speed)
        
    Returns:
        Dictionary containing alignment metrics
    """
    model.eval()
    all_similarities = []
    all_cross_similarities = []
    
    with torch.no_grad():
        for batch_idx, (batched_graphs, text_descriptions) in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            # Convert graph data to PyTorch tensors and move to device
            x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
            batch_tensor = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_tensor)
            
            # Forward pass
            outputs = model(data, text_descriptions)
            
            # Extract embeddings
            code_embeddings = outputs['code_embeddings']
            text_embeddings = outputs['text_embeddings']
            
            # Normalize embeddings
            code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
            text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
            
            # Calculate positive pair similarities (diagonal of similarity matrix)
            positive_similarities = F.cosine_similarity(code_embeddings, text_embeddings, dim=1)
            all_similarities.extend(positive_similarities.cpu().tolist())
            
            # Calculate cross similarities (off-diagonal elements)
            if code_embeddings.size(0) > 1:
                similarity_matrix = torch.matmul(code_embeddings, text_embeddings.t())
                mask = torch.eye(similarity_matrix.size(0), device=device).bool()
                cross_similarities = similarity_matrix[~mask]
                all_cross_similarities.extend(cross_similarities.cpu().tolist())
    
    if all_similarities and all_cross_similarities:
        return {
            'avg_positive_similarity': sum(all_similarities) / len(all_similarities),
            'avg_negative_similarity': sum(all_cross_similarities) / len(all_cross_similarities),
            'alignment_gap': (sum(all_similarities) / len(all_similarities)) - (sum(all_cross_similarities) / len(all_cross_similarities))
        }
    else:
        return {'avg_positive_similarity': 0.0, 'avg_negative_similarity': 0.0, 'alignment_gap': 0.0}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train text-code alignment model')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to dataset directory (default: dataset/)')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--output_path', type=str, default='models/best_alignment_model.pt',
                        help='Path to save the best alignment model (default: models/best_alignment_model.pt)')
    parser.add_argument('--code_encoder_weights_path', type=str, default='models/best_model.pt',
                        help='Path to pre-trained code encoder weights (default: models/best_model.pt)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training (default: 8)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience (default: 5)')
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("🚀 Starting Alignment Training")
    print("=" * 50)
    
    # Training hyperparameters from args
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_epochs = args.epochs
    patience = args.patience
    
    # Data paths
    # Handle sample dataset naming convention
    if args.dataset_path.rstrip('/').endswith('samples'):
        train_data_path = os.path.join(args.dataset_path, "train_paired_data_sample.jsonl")
        val_data_path = os.path.join(args.dataset_path, "validation_paired_data_sample.jsonl")
    else:
        train_data_path = os.path.join(args.dataset_path, "train_paired_data.jsonl")
        val_data_path = os.path.join(args.dataset_path, "validation_paired_data.jsonl")
    
    code_encoder_weights_path = args.code_encoder_weights_path
    output_path = args.output_path
    
    print(f"📋 Training Configuration:")
    print(f"   dataset_path: {args.dataset_path}")
    print(f"   epochs: {num_epochs}")
    print(f"   batch_size: {batch_size}")
    print(f"   learning_rate: {learning_rate}")
    print(f"   patience: {patience}")
    print(f"   output_path: {output_path}")
    print(f"   code_encoder_weights_path: {code_encoder_weights_path}")
    print()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load paired dataset
    print(f"\n📊 Loading training dataset from {train_data_path}")
    try:
        train_loader = create_paired_data_loaders(
            paired_data_path=train_data_path,
            batch_size=batch_size,
            shuffle=True,
            seed=42  # For reproducible training
        )
        print(f"✅ Loaded training data: {len(train_loader)} batches")
        
        # Load separate validation dataset
        print(f"📊 Loading validation dataset from {val_data_path}")
        val_loader = create_paired_data_loaders(
            paired_data_path=val_data_path,
            batch_size=batch_size,
            shuffle=False,
            seed=123  # Different seed for validation
        )
        print(f"✅ Loaded validation data: {len(val_loader)} batches")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Initialize AlignmentModel
    print(f"\n🧠 Initializing AlignmentModel")
    try:
        # Get feature dimension from dataset
        sample_batch = next(iter(train_loader))
        feature_dim = len(sample_batch[0]['x'][0])  # First node's feature dimension
        print(f"Node feature dimension: {feature_dim}")
        
        model = AlignmentModel(
            input_dim=feature_dim,
            hidden_dim=64,  # 64-dimensional shared embedding space
            num_layers=3,
            conv_type='GCN',
            dropout=0.1,
            text_model_name='all-MiniLM-L6-v2',  # Will fallback to SimpleTextEncoder if unavailable
            code_encoder_weights_path=code_encoder_weights_path
        )
        
        model.to(device)
        print(f"✅ Model initialized")
        print(f"Model info:\n{model.get_model_info()}")
        
        # Count trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return
    
    # Setup optimizer (only for trainable parameters)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    print(f"\n🏋️ Starting training for {num_epochs} epochs")
    print("=" * 70)
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        
        # Validation (use subset for speed)
        val_loss = validate_epoch(model, val_loader, device)
        val_losses.append(val_loss)
        
        # Calculate alignment metrics
        metrics = calculate_alignment_metrics(model, val_loader, device, max_batches=5)
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch results
        print(f"Epoch {epoch+1:2d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Pos Sim: {metrics['avg_positive_similarity']:.3f} | "
              f"Neg Sim: {metrics['avg_negative_similarity']:.3f} | "
              f"Gap: {metrics['alignment_gap']:.3f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model (only trainable text projection weights)
            try:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'text_projection_state_dict': model.text_projection.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'model_config': {
                        'input_dim': feature_dim,
                        'hidden_dim': 64,
                        'num_layers': 3,
                        'conv_type': 'GCN',
                        'dropout': 0.1
                    }
                }
                torch.save(checkpoint, output_path)
                print(f"✅ Saved best model to {output_path}")
                
            except Exception as e:
                print(f"⚠️  Error saving model: {e}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping triggered after {patience} epochs without improvement")
            break
    
    # Training summary
    print("\n" + "=" * 70)
    print("🏁 Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {train_losses[-1]:.4f}")
    print(f"Loss improvement: {train_losses[0]:.4f} → {train_losses[-1]:.4f}")
    print(f"Best weights saved to: {output_path}")
    
    # Verify loss decreased
    if len(train_losses) > 5:
        initial_avg = sum(train_losses[:5]) / 5
        final_avg = sum(train_losses[-5:]) / 5
        improvement = initial_avg - final_avg
        
        if improvement > 0:
            print(f"✅ Training loss successfully decreased by {improvement:.4f}")
        else:
            print(f"⚠️  Training loss did not improve significantly")
    
    print("\n🎉 Alignment training completed successfully!")


if __name__ == "__main__":
    main()
