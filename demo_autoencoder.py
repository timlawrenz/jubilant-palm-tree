#!/usr/bin/env python3
"""
Demo script showcasing the AST Autoencoder functionality.

This script demonstrates the complete AST → embedding → reconstructed AST pipeline
using the new ASTAutoencoder model with real Ruby method data.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset
from models import ASTAutoencoder


def demo_single_method():
    """Demonstrate autoencoder on a single Ruby method."""
    print("🚀 AST Autoencoder Demo")
    print("=" * 50)
    
    # Load dataset and get a sample
    print("📥 Loading Ruby AST dataset...")
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    print(f"📊 Original AST Statistics:")
    print(f"   Nodes: {len(sample['x'])}")
    print(f"   Edges: {len(sample['edge_index'][0])}")
    print(f"   Node features: {len(sample['x'][0])}")
    print(f"   Complexity: {sample['y']}")
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Initialize autoencoder
    print("\n🏗️  Initializing AST Autoencoder...")
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        freeze_encoder=False  # Allow training
    )
    
    print(f"📋 Model Architecture:")
    print(autoencoder.get_model_info())
    
    # Perform forward pass
    print("\n⚡ Running Forward Pass...")
    autoencoder.eval()
    with torch.no_grad():
        result = autoencoder(data)
    
    embedding = result['embedding']
    reconstruction = result['reconstruction']
    
    print(f"✅ Encoding Complete:")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding sample: {embedding[0][:5].tolist()}")
    
    print(f"\n✅ Decoding Complete:")
    print(f"   Reconstructed nodes: {reconstruction['node_features'].shape[1]}")
    print(f"   Reconstructed edges: {reconstruction['edge_index'].shape[1]}")
    print(f"   Output feature dimension: {reconstruction['node_features'].shape[2]}")
    
    # Compare original vs reconstructed
    print(f"\n📈 Reconstruction Comparison:")
    print(f"   Original nodes:     {x.shape[0]:3d}")
    print(f"   Reconstructed nodes: {reconstruction['node_features'].shape[1]:3d}")
    print(f"   Original edges:     {edge_index.shape[1]:3d}")
    print(f"   Reconstructed edges: {reconstruction['edge_index'].shape[1]:3d}")
    
    return True


def demo_batch_processing():
    """Demonstrate autoencoder on a batch of methods."""
    print("\n\n📦 Batch Processing Demo")
    print("=" * 50)
    
    # Load multiple samples
    dataset = RubyASTDataset("dataset/train.jsonl")
    batch_size = 3
    
    # Manually create a batch (simplified)
    samples = [dataset[i] for i in range(batch_size)]
    
    # Convert to batch format (simplified - in practice you'd use DataLoader)
    batch_data = []
    batch_idx = []
    edge_offset = 0
    
    for i, sample in enumerate(samples):
        x = torch.tensor(sample['x'], dtype=torch.float)
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long) + edge_offset
        
        batch_data.append(x)
        batch_idx.extend([i] * x.shape[0])
        edge_offset += x.shape[0]
    
    # Combine into single tensors
    all_x = torch.cat(batch_data, dim=0)
    all_edges = torch.cat([torch.tensor(s['edge_index'], dtype=torch.long) + 
                          sum(len(samples[j]['x']) for j in range(i)) 
                          for i, s in enumerate(samples)], dim=1)
    batch_tensor = torch.tensor(batch_idx, dtype=torch.long)
    
    data = Data(x=all_x, edge_index=all_edges, batch=batch_tensor)
    
    print(f"📊 Batch Statistics:")
    print(f"   Graphs: {batch_size}")
    print(f"   Total nodes: {all_x.shape[0]}")
    print(f"   Total edges: {all_edges.shape[1]}")
    print(f"   Individual graph sizes: {[len(s['x']) for s in samples]}")
    
    # Initialize autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    
    # Process batch
    print("\n⚡ Processing Batch...")
    autoencoder.eval()
    with torch.no_grad():
        result = autoencoder(data)
    
    embeddings = result['embedding']
    reconstructions = result['reconstruction']
    
    print(f"✅ Batch Results:")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Reconstructed graphs: {len(reconstructions['num_nodes_per_graph'])}")
    print(f"   Individual embeddings:")
    for i in range(batch_size):
        print(f"     Graph {i+1}: {embeddings[i][:3].tolist()}")
    
    return True


def demo_frozen_encoder():
    """Demonstrate autoencoder with frozen encoder."""
    print("\n\n❄️  Frozen Encoder Demo")
    print("=" * 50)
    
    # Create autoencoder with frozen encoder
    autoencoder_frozen = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        freeze_encoder=True,
        encoder_weights_path="nonexistent_model.pt"  # Will warn gracefully
    )
    
    # Check parameter status
    encoder_frozen = all(not p.requires_grad for p in autoencoder_frozen.encoder.parameters())
    decoder_trainable = any(p.requires_grad for p in autoencoder_frozen.decoder.parameters())
    
    print(f"✅ Parameter Status:")
    print(f"   Encoder frozen: {encoder_frozen}")
    print(f"   Decoder trainable: {decoder_trainable}")
    
    # Count parameters
    encoder_params = sum(p.numel() for p in autoencoder_frozen.encoder.parameters())
    decoder_params = sum(p.numel() for p in autoencoder_frozen.decoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder_frozen.decoder.parameters() if p.requires_grad)
    
    print(f"\n📊 Parameter Count:")
    print(f"   Encoder parameters: {encoder_params:,}")
    print(f"   Decoder parameters: {decoder_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return True


def main():
    """Run all demos."""
    try:
        demo_single_method()
        demo_batch_processing() 
        demo_frozen_encoder()
        
        print("\n\n🎉 All Demos Completed Successfully!")
        print("The AST Autoencoder is ready for:")
        print("• AST embedding extraction")
        print("• AST reconstruction from embeddings") 
        print("• Batch processing")
        print("• Transfer learning with frozen encoders")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    main()