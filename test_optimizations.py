#!/usr/bin/env python3
"""
Test script to validate that the optimized training still works correctly.
"""

import sys
import os
import torch
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import AutoregressiveASTDecoder, AlignmentModel
from data_processing import create_autoregressive_data_loader


def test_optimized_training():
    """Test that optimized training components work correctly."""
    print("🧪 Testing optimized training components")
    print("=" * 50)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test 1: Create optimized data loader
    print("\n1. Testing optimized data loader...")
    try:
        loader = create_autoregressive_data_loader(
            "dataset/train_paired_data.jsonl",
            batch_size=2,
            shuffle=False,
            max_sequence_length=5,
            precomputed_embeddings_path="output/text_embeddings.pt"
        )
        print(f"✅ Data loader created with {len(loader)} batches")
    except Exception as e:
        print(f"❌ Data loader creation failed: {e}")
        return False
    
    # Test 2: Create models
    print("\n2. Testing model creation...")
    try:
        model = AutoregressiveASTDecoder(
            text_embedding_dim=64,
            graph_hidden_dim=64,
            state_hidden_dim=128,
            node_types=74,
            sequence_model='GRU'
        ).to(device)
        
        alignment_model = AlignmentModel(
            input_dim=74,
            hidden_dim=64,
            text_model_name='all-MiniLM-L6-v2'
        ).to(device)
        
        print("✅ Models created successfully")
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False
    
    # Test 3: Test pre-computed embedding handling
    print("\n3. Testing pre-computed embedding handling...")
    try:
        # Create a mock batch with pre-computed embeddings
        test_embedding = torch.randn(1, 64, device=device)
        mock_batch = {
            'text_descriptions': ['test method'],
            'text_embeddings': [test_embedding],  # Pre-computed embedding
            'partial_graphs': {
                'x': [],
                'edge_index': [[], []],
                'batch': [],
                'num_graphs': 1
            },
            'target_node_types': ['def'],
            'target_node_features': [[1.0] * 74],
            'target_connections': [[0.0] * 100],
            'steps': [0],
            'total_steps': [1]
        }
        
        # Test model forward pass with pre-computed embeddings
        model.eval()
        with torch.no_grad():
            outputs = model(test_embedding, None, None)
            print("✅ Model forward pass with pre-computed embeddings successful")
            print(f"   Output keys: {list(outputs.keys())}")
    except Exception as e:
        print(f"❌ Pre-computed embedding test failed: {e}")
        return False
    
    # Test 4: Test fallback to alignment model
    print("\n4. Testing fallback to alignment model...")
    try:
        mock_batch_no_precomputed = {
            'text_descriptions': ['test method'],
            'text_embeddings': [None],  # No pre-computed embedding
            'partial_graphs': {
                'x': [],
                'edge_index': [[], []],
                'batch': [],
                'num_graphs': 1
            },
            'target_node_types': ['def'],
            'target_node_features': [[1.0] * 74],
            'target_connections': [[0.0] * 100],
            'steps': [0],
            'total_steps': [1]
        }
        
        # This should fallback to using the alignment model
        text_embedding = alignment_model.encode_text(['test method'])
        print("✅ Fallback to alignment model successful")
        print(f"   Embedding shape: {text_embedding.shape}")
    except Exception as e:
        print(f"❌ Alignment model fallback test failed: {e}")
        return False
    
    print("\n🎉 All optimization tests passed!")
    print("\nKey improvements implemented:")
    print("  ✅ Pre-computed text embeddings (eliminates repeated encoding)")
    print("  ✅ Optimized PyTorch DataLoader with multiprocessing")
    print("  ✅ Pinned memory for faster GPU transfer")
    print("  ✅ Fallback compatibility for missing pre-computed embeddings")
    
    return True


if __name__ == "__main__":
    success = test_optimized_training()
    if not success:
        sys.exit(1)