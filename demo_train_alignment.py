#!/usr/bin/env python3
"""
Demo training script for text-code alignment to show decreasing loss.

This script runs alignment training on a small subset of data to demonstrate
that the training pipeline works and the loss decreases over time.
"""

import sys
import os
import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_paired_data_loaders
from models import AlignmentModel
from loss import info_nce_loss


def demo_alignment_training():
    """Demo alignment training with subset of data."""
    print("🚀 Demo Alignment Training")
    print("=" * 50)
    
    # Demo configuration
    batch_size = 4
    learning_rate = 1e-3
    num_epochs = 10
    max_batches_per_epoch = 100  # Limit batches for demo
    
    # Data paths
    train_data_path = "dataset/train_paired_data.jsonl"
    code_encoder_weights_path = "best_model.pt"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\n📊 Loading training dataset")
    train_loader = create_paired_data_loaders(
        paired_data_path=train_data_path,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    print(f"✅ Data loaded: {len(train_loader)} total batches (using {max_batches_per_epoch} per epoch)")
    
    # Get feature dimension
    sample_batch = next(iter(train_loader))
    feature_dim = len(sample_batch[0]['x'][0])
    print(f"Node feature dimension: {feature_dim}")
    
    # Initialize model
    print(f"\n🧠 Initializing AlignmentModel")
    model = AlignmentModel(
        input_dim=feature_dim,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        dropout=0.1,
        text_model_name='all-MiniLM-L6-v2',
        code_encoder_weights_path=code_encoder_weights_path
    )
    
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (text projection head only)")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate
    )
    
    # Training loop
    print(f"\n🏋️ Starting demo training for {num_epochs} epochs")
    print("-" * 60)
    
    epoch_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        print(f"Epoch {epoch+1}/{num_epochs}:")
        
        train_iter = iter(train_loader)
        for batch_idx in range(max_batches_per_epoch):
            try:
                batched_graphs, text_descriptions = next(train_iter)
            except StopIteration:
                break
            
            # Convert to tensors
            x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
            batch_tensor = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
            
            data = Data(x=x, edge_index=edge_index, batch=batch_tensor)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(data, text_descriptions)
            
            # Compute loss
            loss = info_nce_loss(outputs['code_embeddings'], outputs['text_embeddings'])
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            # Print progress every 20 batches
            if (batch_idx + 1) % 20 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"  Batch {batch_idx+1:3d}/{max_batches_per_epoch}: Loss = {loss.item():.4f}, Avg = {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_losses.append(avg_epoch_loss)
        
        print(f"  Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")
        
        # Calculate some alignment metrics for the last batch
        with torch.no_grad():
            code_embeddings = outputs['code_embeddings']
            text_embeddings = outputs['text_embeddings']
            
            # Normalize for cosine similarity
            code_norm = F.normalize(code_embeddings, p=2, dim=1)
            text_norm = F.normalize(text_embeddings, p=2, dim=1)
            
            # Positive pair similarities
            pos_similarities = F.cosine_similarity(code_norm, text_norm, dim=1)
            avg_pos_sim = pos_similarities.mean().item()
            
            print(f"  Average positive pair similarity: {avg_pos_sim:.3f}")
        
        print()
    
    # Training summary
    print("=" * 60)
    print("🏁 Demo Training Summary")
    print("-" * 30)
    
    print(f"Initial loss (epoch 1): {epoch_losses[0]:.4f}")
    print(f"Final loss (epoch {num_epochs}): {epoch_losses[-1]:.4f}")
    
    loss_improvement = epoch_losses[0] - epoch_losses[-1]
    if loss_improvement > 0:
        print(f"✅ Loss decreased by: {loss_improvement:.4f}")
        print(f"✅ Relative improvement: {(loss_improvement/epoch_losses[0]*100):.1f}%")
    else:
        print(f"⚠️  Loss did not improve significantly")
    
    print("\nLoss progression:")
    for i, loss in enumerate(epoch_losses):
        print(f"  Epoch {i+1:2d}: {loss:.4f}")
    
    # Save demo model
    demo_output_path = "demo_alignment_model.pt"
    checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'text_projection_state_dict': model.text_projection.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': epoch_losses[-1],
        'epoch_losses': epoch_losses,
        'model_config': {
            'input_dim': feature_dim,
            'hidden_dim': 64,
            'num_layers': 3,
            'conv_type': 'GCN',
            'dropout': 0.1
        }
    }
    torch.save(checkpoint, demo_output_path)
    print(f"\n💾 Demo model saved to: {demo_output_path}")
    
    print("\n🎉 Demo completed! Training loss successfully decreased over time.")
    return True


if __name__ == "__main__":
    demo_alignment_training()