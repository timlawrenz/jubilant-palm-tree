#!/usr/bin/env python3
"""
Test script for the new AST Autoencoder models.

This script tests the ASTDecoder and ASTAutoencoder classes to ensure
they work correctly with the existing Ruby AST dataset.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import RubyASTDataset, create_data_loaders
from models import RubyComplexityGNN, ASTDecoder, ASTAutoencoder

# Helper function to get dataset paths relative to this script
def get_dataset_path(relative_path):
    """Get dataset path relative to this script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)


def test_encoder_embedding_extraction():
    """Test that the encoder can extract embeddings."""
    print("🔍 Testing Encoder Embedding Extraction")
    print("-" * 40)
    
    # Load a sample
    dataset = RubyASTDataset(get_dataset_path("../dataset/samples/train_sample.jsonl"))
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Test encoder
    model = RubyComplexityGNN(input_dim=74, hidden_dim=64)
    model.eval()
    
    with torch.no_grad():
        # Test normal prediction
        prediction = model(data, return_embedding=False)
        print(f"✅ Prediction output shape: {prediction.shape}")
        
        # Test embedding extraction
        embedding = model(data, return_embedding=True)
        print(f"✅ Embedding output shape: {embedding.shape}")
        
        assert prediction.shape == (1, 1), f"Expected prediction shape (1, 1), got {prediction.shape}"
        assert embedding.shape == (1, 64), f"Expected embedding shape (1, 64), got {embedding.shape}"
    
    return True


def test_ast_decoder():
    """Test the ASTDecoder module."""
    print("\n🔧 Testing ASTDecoder")
    print("-" * 40)
    
    # Test decoder with synthetic embedding
    decoder = ASTDecoder(
        embedding_dim=64,
        output_node_dim=74,  # Same as input node features
        hidden_dim=64,
        num_layers=2,
        max_nodes=50
    )
    decoder.eval()
    
    # Create fake embedding
    batch_size = 2
    embedding = torch.randn(batch_size, 64)
    
    with torch.no_grad():
        output = decoder(embedding, target_num_nodes=20)
    
    print(f"✅ Decoder output keys: {list(output.keys())}")
    print(f"✅ Node features shape: {output['node_features'].shape}")
    print(f"✅ Edge index shape: {output['edge_index'].shape}")
    print(f"✅ Batch tensor shape: {output['batch'].shape}")
    print(f"✅ Nodes per graph: {output['num_nodes_per_graph']}")
    
    # Verify output shapes
    expected_node_shape = (batch_size, 20, 74)
    assert output['node_features'].shape == expected_node_shape, \
        f"Expected node features shape {expected_node_shape}, got {output['node_features'].shape}"
    
    return True


def test_ast_autoencoder():
    """Test the complete ASTAutoencoder."""
    print("\n🤖 Testing ASTAutoencoder")
    print("-" * 40)
    
    # Load a sample
    dataset = RubyASTDataset(get_dataset_path("../dataset/samples/train_sample.jsonl"))
    sample = dataset[0]
    
    # Convert to PyTorch format
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    print(f"Input AST - Nodes: {x.shape[0]}, Edges: {edge_index.shape[1]}, Features: {x.shape[1]}")
    
    # Test autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN'
    )
    autoencoder.eval()
    
    with torch.no_grad():
        result = autoencoder(data)
    
    print(f"✅ Autoencoder output keys: {list(result.keys())}")
    print(f"✅ Embedding shape: {result['embedding'].shape}")
    print(f"✅ Reconstruction keys: {list(result['reconstruction'].keys())}")
    print(f"✅ Reconstructed node features shape: {result['reconstruction']['node_features'].shape}")
    
    # Print model info
    print(f"\n📋 Model Configuration:")
    print(autoencoder.get_model_info())
    
    return True


def test_autoencoder_with_frozen_encoder():
    """Test autoencoder with frozen encoder."""
    print("\n❄️  Testing ASTAutoencoder with Frozen Encoder")
    print("-" * 40)
    
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        freeze_encoder=True,
        encoder_weights_path="models/samples/best_model.pt"  # Use sample model
    )
    
    # Check that encoder parameters are frozen
    encoder_params_frozen = all(not param.requires_grad for param in autoencoder.encoder.parameters())
    decoder_params_trainable = any(param.requires_grad for param in autoencoder.decoder.parameters())
    
    print(f"✅ Encoder parameters frozen: {encoder_params_frozen}")
    print(f"✅ Decoder parameters trainable: {decoder_params_trainable}")
    
    assert encoder_params_frozen, "Encoder parameters should be frozen"
    assert decoder_params_trainable, "Decoder parameters should be trainable"
    
    return True


def test_batch_processing():
    """Test autoencoder with batched data."""
    print("\n📦 Testing Batch Processing")
    print("-" * 40)
    
    # Create data loaders
    train_loader, _ = create_data_loaders(
        get_dataset_path("../dataset/samples/train_sample.jsonl"), 
        get_dataset_path("../dataset/samples/validation_sample.jsonl"), 
        batch_size=3
    )
    
    # Get a batch
    batch = next(iter(train_loader))
    
    # Convert to PyTorch format
    x = torch.tensor(batch['x'], dtype=torch.float)
    edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)
    batch_idx = torch.tensor(batch['batch'], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    print(f"Batch - Graphs: {batch['num_graphs']}, Total nodes: {x.size(0)}, Edges: {edge_index.size(1)}")
    
    # Test autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    autoencoder.eval()
    
    with torch.no_grad():
        result = autoencoder(data)
    
    print(f"✅ Batch embedding shape: {result['embedding'].shape}")
    print(f"✅ Expected batch size: {batch['num_graphs']}")
    
    assert result['embedding'].shape[0] == batch['num_graphs'], \
        f"Embedding batch size {result['embedding'].shape[0]} != expected {batch['num_graphs']}"
    
    return True


def main():
    """Run all autoencoder tests."""
    print("🧪 AST Autoencoder Testing Suite")
    print("=" * 50)
    
    tests = [
        test_encoder_embedding_extraction,
        test_ast_decoder,
        test_ast_autoencoder,
        test_autoencoder_with_frozen_encoder,
        test_batch_processing
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All autoencoder tests passed! AST Autoencoder is ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()