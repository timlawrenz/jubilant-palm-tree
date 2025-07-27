#!/usr/bin/env python3
"""
Create sample model files for testing purposes.
These are lightweight dummy models that can be used in CI without requiring LFS.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import RubyComplexityGNN, ASTAutoencoder, AlignmentModel, AutoregressiveASTDecoder

def create_sample_complexity_model():
    """Create a sample RubyComplexityGNN model."""
    print("Creating sample complexity model...")
    model = RubyComplexityGNN(input_dim=74, hidden_dim=64)
    
    # Save model state dict in the expected format
    model_save_data = {
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': 74,
            'hidden_dim': 64
        }
    }
    torch.save(model_save_data, 'models/samples/best_model_sample.pt')
    
    # Also save with the original name for backward compatibility
    torch.save(model_save_data, 'models/samples/best_model.pt')
    print("✅ Created models/samples/best_model.pt")

def create_sample_autoencoder_models():
    """Create sample autoencoder models."""
    print("Creating sample autoencoder models...")
    
    # Create autoencoder
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64
    )
    
    # Save just the decoder part with proper format
    decoder_save_data = {
        'model_state_dict': autoencoder.decoder.state_dict(),
        'config': {
            'embedding_dim': 64,
            'output_node_dim': 74,
            'hidden_dim': 64
        }
    }
    torch.save(decoder_save_data, 'models/samples/best_decoder.pt')
    print("✅ Created models/samples/best_decoder.pt")
    
    # Save complete autoencoder
    autoencoder_save_data = {
        'model_state_dict': autoencoder.state_dict(),
        'config': {
            'encoder_input_dim': 74,
            'node_output_dim': 74,
            'hidden_dim': 64
        }
    }
    torch.save(autoencoder_save_data, 'models/samples/autoencoder_sample.pt')
    print("✅ Created models/samples/autoencoder_sample.pt")

def create_sample_alignment_model():
    """Create a sample alignment model."""
    print("Creating sample alignment model...")
    
    alignment_model = AlignmentModel(input_dim=74, hidden_dim=64)
    
    # Save in the expected format
    model_save_data = {
        'model_state_dict': alignment_model.state_dict(),
        'config': {
            'input_dim': 74,
            'hidden_dim': 64
        }
    }
    
    torch.save(model_save_data, 'models/samples/demo_alignment_model.pt')
    print("✅ Created models/samples/demo_alignment_model.pt")
    
    # Also create the variant expected by generate_code.py
    torch.save(model_save_data, 'models/samples/best_alignment_model.pt')
    print("✅ Created models/samples/best_alignment_model.pt")

def create_sample_autoregressive_decoder():
    """Create a sample autoregressive decoder model."""
    print("Creating sample autoregressive decoder...")
    
    autoregressive_decoder = AutoregressiveASTDecoder(
        text_embedding_dim=64,
        graph_hidden_dim=64,
        state_hidden_dim=128,
        node_types=74,
        max_nodes=100
    )
    
    # Save in the expected format
    decoder_save_data = {
        'model_state_dict': autoregressive_decoder.state_dict(),
        'config': {
            'text_embedding_dim': 64,
            'graph_hidden_dim': 64,
            'state_hidden_dim': 128,
            'node_types': 74,
            'max_nodes': 100
        }
    }
    torch.save(decoder_save_data, 'best_autoregressive_decoder.pt')
    print("✅ Created best_autoregressive_decoder.pt")

def main():
    """Create all sample models."""
    print("🔧 Creating Sample Models for Testing")
    print("=" * 40)
    
    os.makedirs('models/samples', exist_ok=True)
    
    try:
        create_sample_complexity_model()
        create_sample_autoencoder_models()
        create_sample_alignment_model()
        create_sample_autoregressive_decoder()
        
        print("\n" + "=" * 40)
        print("🎉 All sample models created successfully!")
        print("These lightweight models can be used for CI testing without LFS.")
        
        return True
    except Exception as e:
        print(f"❌ Error creating sample models: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)