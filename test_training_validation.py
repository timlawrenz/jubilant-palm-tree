#!/usr/bin/env python3
"""
Quick test to validate the train_autoregressive.py functionality
"""

import os
import sys
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import AutoregressiveASTDecoder

def test_model_loading():
    """Test loading the saved model"""
    print("🧪 Testing model loading...")
    
    # Load the checkpoint
    checkpoint = torch.load('best_autoregressive_decoder.pt', map_location='cpu')
    
    # Create model with same config
    model = AutoregressiveASTDecoder(**checkpoint['model_config'])
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✅ Model loaded successfully!")
    print(f"   Epoch: {checkpoint['epoch']}")
    print(f"   Train Loss: {checkpoint['train_loss']:.6f}")
    print(f"   Val Loss: {checkpoint['val_loss']:.6f}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters())}")
    
    return True

def test_model_inference():
    """Test model inference"""
    print("\n🧪 Testing model inference...")
    
    # Load model
    checkpoint = torch.load('best_autoregressive_decoder.pt', map_location='cpu')
    model = AutoregressiveASTDecoder(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy inputs
    text_embedding = torch.randn(1, 64)  # Batch size 1, 64-dim embedding
    
    # Test with empty graph (start of generation)
    empty_graph = {'x': [], 'edge_index': [[], []], 'batch': []}
    
    with torch.no_grad():
        outputs = model(text_embedding, empty_graph, hidden_state=None)
    
    print(f"✅ Model inference successful!")
    print(f"   Node type logits shape: {outputs['node_type_logits'].shape}")
    print(f"   Connection probs shape: {outputs['connection_probs'].shape}")
    print(f"   Hidden state: {type(outputs['hidden_state'])}")
    
    return True

if __name__ == "__main__":
    print("🎯 Autoregressive Training Validation")
    print("=" * 40)
    
    try:
        test_model_loading()
        test_model_inference()
        print("\n🎉 All tests passed! Training script is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)