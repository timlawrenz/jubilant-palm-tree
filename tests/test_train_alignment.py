#!/usr/bin/env python3
"""
Quick test script for train_alignment.py to verify functionality.

This script runs a minimal version of the alignment training to ensure
everything works correctly before running the full training.
"""

import sys
import os
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import create_paired_data_loaders
from models import AlignmentModel
from loss import info_nce_loss


def test_alignment_training():
    """Test the alignment training pipeline with minimal data."""
    print("🧪 Testing Alignment Training Pipeline")
    print("=" * 50)
    
    # Test configuration
    batch_size = 2  # Very small batch for testing
    paired_data_path = "../dataset/paired_data.jsonl"
    code_encoder_weights_path = "models/best_model.pt"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Test data loading
        print("\n📊 Testing data loading...")
        train_loader = create_paired_data_loaders(
            paired_data_path=paired_data_path,
            batch_size=batch_size,
            shuffle=True,
            seed=42
        )
        print(f"✅ Data loader created: {len(train_loader)} batches")
        
        # Test getting a sample batch
        sample_batch = next(iter(train_loader))
        batched_graphs, text_descriptions = sample_batch
        feature_dim = len(batched_graphs['x'][0])
        print(f"✅ Sample batch retrieved")
        print(f"   - Feature dimension: {feature_dim}")
        print(f"   - Batch size: {len(text_descriptions)}")
        print(f"   - Sample text: '{text_descriptions[0]}'")
        
        # Test model initialization
        print("\n🧠 Testing model initialization...")
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
        print(f"✅ Model initialized successfully")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - Total parameters: {total_params:,}")
        print(f"   - Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        print("\n🔄 Testing forward pass...")
        x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
        edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
        batch_idx = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
        
        from torch_geometric.data import Data
        data = Data(x=x, edge_index=edge_index, batch=batch_idx)
        
        outputs = model(data, text_descriptions)
        code_embeddings = outputs['code_embeddings']
        text_embeddings = outputs['text_embeddings']
        
        print(f"✅ Forward pass successful")
        print(f"   - Code embeddings shape: {code_embeddings.shape}")
        print(f"   - Text embeddings shape: {text_embeddings.shape}")
        
        # Test loss calculation
        print("\n📉 Testing loss calculation...")
        loss = info_nce_loss(code_embeddings, text_embeddings)
        print(f"✅ Loss calculated: {loss.item():.4f}")
        
        # Test backward pass
        print("\n⬅️  Testing backward pass...")
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-3
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"✅ Backward pass successful")
        
        # Test a few training steps
        print("\n🏋️ Testing training steps...")
        model.train()
        initial_loss = loss.item()
        
        for step in range(3):
            # Get another batch
            try:
                batched_graphs, text_descriptions = next(iter(train_loader))
            except:
                break
                
            x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
            batch_idx = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
            
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            optimizer.zero_grad()
            outputs = model(data, text_descriptions)
            loss = info_nce_loss(outputs['code_embeddings'], outputs['text_embeddings'])
            loss.backward()
            optimizer.step()
            
            print(f"   Step {step+1}: Loss = {loss.item():.4f}")
        
        print(f"✅ Training steps completed")
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {loss.item():.4f}")
        
        # Test model saving
        print("\n💾 Testing model saving...")
        test_save_path = "test_alignment_model.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'text_projection_state_dict': model.text_projection.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
            'model_config': {
                'input_dim': feature_dim,
                'hidden_dim': 64,
                'num_layers': 3,
                'conv_type': 'GCN',
                'dropout': 0.1
            }
        }
        torch.save(checkpoint, test_save_path)
        print(f"✅ Model saved to {test_save_path}")
        
        # Clean up test file
        os.remove(test_save_path)
        print(f"✅ Test file cleaned up")
        
        print("\n" + "=" * 50)
        print("🎉 All tests passed! Alignment training pipeline is ready.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_alignment_training()
    if success:
        print("\n✅ Ready to run full training with: python train_alignment.py")
    else:
        print("\n❌ Fix errors before running full training")