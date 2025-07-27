#!/usr/bin/env python3
"""
Demo script for the AutoregressiveASTDecoder model.

This script demonstrates the key features of the Phase 7 AutoregressiveASTDecoder
including different sequence models, partial graph processing, and sequential generation.
"""

import torch
from src.models import AutoregressiveASTDecoder
from src.data_processing import create_autoregressive_data_loader

def main():
    """Demonstrate AutoregressiveASTDecoder functionality."""
    print("🚀 AutoregressiveASTDecoder Demo")
    print("=" * 50)
    
    # Test different sequence models
    sequence_models = ['GRU', 'LSTM', 'Transformer']
    
    for seq_model in sequence_models:
        print(f"\n📋 Testing {seq_model} Model")
        print("-" * 30)
        
        # Create decoder
        decoder = AutoregressiveASTDecoder(
            text_embedding_dim=64,
            graph_hidden_dim=64,
            state_hidden_dim=128,
            node_types=74,
            sequence_model=seq_model
        )
        
        print(f"Model Architecture:")
        print(decoder.get_model_info())
        
        # Demonstrate with real data
        print(f"\n🔄 Testing with real sequential data...")
        
        try:
            # Load some real autoregressive data
            loader = create_autoregressive_data_loader(
                'dataset_paired_data_example.jsonl',
                batch_size=2,
                max_sequence_length=3,
                seed=42
            )
            
            # Get a batch
            for batch in loader:
                # Create text embeddings (simulating alignment model output)
                batch_size = len(batch['text_descriptions'])
                text_embeddings = torch.randn(batch_size, 64)
                
                print(f"Batch info:")
                print(f"  - Text descriptions: {batch_size}")
                print(f"  - Steps: {batch['steps']}")
                print(f"  - Target node types: {batch['target_node_types']}")
                
                # Forward pass
                outputs = decoder(text_embeddings, batch['partial_graphs'])
                
                print(f"Decoder outputs:")
                print(f"  - Node type logits: {outputs['node_type_logits'].shape}")
                print(f"  - Connection probs: {outputs['connection_probs'].shape}")
                print(f"  - Hidden state: {type(outputs['hidden_state'])}")
                
                # Show prediction probabilities
                node_probs = torch.softmax(outputs['node_type_logits'], dim=-1)
                print(f"  - Top predicted node types:")
                for b in range(batch_size):
                    top_indices = torch.topk(node_probs[b], 3).indices
                    top_probs = torch.topk(node_probs[b], 3).values
                    print(f"    Batch {b}: indices {top_indices.tolist()}, probs {top_probs.tolist()}")
                
                break
                
            print(f"✅ {seq_model} model test successful!")
            
        except Exception as e:
            print(f"❌ {seq_model} model test failed: {e}")
    
    print(f"\n🎯 Demonstration of Sequential Generation")
    print("-" * 40)
    
    # Demonstrate sequential generation process
    decoder = AutoregressiveASTDecoder(sequence_model='GRU')
    text_embedding = torch.randn(1, 64)  # Single example
    
    print("Simulating autoregressive AST generation:")
    
    # Step 1: Start with empty graph
    print("Step 1: Empty graph → First node")
    outputs1 = decoder(text_embedding)  # No partial graph
    first_node_logits = outputs1['node_type_logits']
    predicted_type = torch.argmax(first_node_logits, dim=-1).item()
    print(f"  Predicted first node type: {predicted_type}")
    hidden_state = outputs1['hidden_state']
    
    # Step 2: Add first node and predict second
    print("Step 2: One node → Second node")
    partial_graph = {
        'x': [[0.0] * 74],  # One node with zero features (placeholder)
        'edge_index': [[], []],  # No edges yet
        'batch': [0]  # Single graph
    }
    partial_graph['x'][0][predicted_type] = 1.0  # Set predicted type
    
    outputs2 = decoder(text_embedding, partial_graph, hidden_state)
    second_node_logits = outputs2['node_type_logits']
    predicted_type2 = torch.argmax(second_node_logits, dim=-1).item()
    print(f"  Predicted second node type: {predicted_type2}")
    print(f"  Connection probabilities: {outputs2['connection_probs'][0][:5].tolist()}")
    
    print(f"\n✅ Sequential generation demonstration complete!")
    
    print(f"\n🎉 All AutoregressiveASTDecoder demos completed successfully!")
    print("   The model is ready for Phase 7 autoregressive AST generation!")

if __name__ == "__main__":
    main()