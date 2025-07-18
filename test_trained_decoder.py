#!/usr/bin/env python3
"""
Test script to verify the trained decoder weights can be loaded and used.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset
from models import ASTAutoencoder, ASTDecoder

def test_decoder_loading():
    """Test loading and using the trained decoder weights."""
    print("🧪 Testing Trained Decoder Loading")
    print("=" * 40)
    
    # Load a sample for testing
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    print(f"📊 Test data: {x.shape[0]} nodes, {edge_index.shape[1]} edges")
    
    # Create autoencoder with fresh weights
    print("\n🔧 Creating fresh autoencoder...")
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        freeze_encoder=True
    )
    
    # Get initial reconstruction
    autoencoder.eval()
    with torch.no_grad():
        initial_result = autoencoder(data)
        initial_recon = initial_result['reconstruction']['node_features']
    
    print(f"✅ Initial reconstruction shape: {initial_recon.shape}")
    
    # Load trained decoder weights
    print("\n📥 Loading trained decoder weights...")
    checkpoint = torch.load('best_decoder.pt', map_location='cpu')
    autoencoder.decoder.load_state_dict(checkpoint['decoder_state_dict'])
    
    print(f"✅ Loaded weights from epoch {checkpoint['epoch']}")
    print(f"   Training loss: {checkpoint['train_loss']:.4f}")
    print(f"   Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Get reconstruction with trained weights
    autoencoder.eval()
    with torch.no_grad():
        trained_result = autoencoder(data)
        trained_recon = trained_result['reconstruction']['node_features']
    
    print(f"✅ Trained reconstruction shape: {trained_recon.shape}")
    
    # Verify weights changed
    diff = torch.mean(torch.abs(initial_recon - trained_recon)).item()
    print(f"✅ Reconstruction difference: {diff:.6f}")
    
    if diff > 0.001:
        print("✅ Decoder weights successfully loaded and changed output")
    else:
        print("⚠️  Warning: Decoder output unchanged - may indicate loading issue")
    
    return True

if __name__ == "__main__":
    test_decoder_loading()