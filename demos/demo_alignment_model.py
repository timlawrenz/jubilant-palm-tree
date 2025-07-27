#!/usr/bin/env python3
"""
AlignmentModel Demonstration Script

This script demonstrates the AlignmentModel's ability to create aligned
embeddings for Ruby code and text descriptions.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from src.models import AlignmentModel
from src.data_processing import create_data_loaders

def demonstrate_alignment_model():
    """Demonstrate the AlignmentModel with both synthetic and real data."""
    
    print("🚀 AlignmentModel Demonstration")
    print("=" * 50)
    
    # Create model - use 74 dimensions based on real data
    print("\n📦 Creating AlignmentModel...")
    model = AlignmentModel(
        input_dim=74,  # Based on real dataset
        hidden_dim=64,
        text_model_name='all-MiniLM-L6-v2'
    )
    
    print(f"✅ Model created successfully")
    print(f"📊 Model info:\n{model.get_model_info()}")
    
    # Demonstrate with synthetic data
    print("\n🔬 Testing with Synthetic Data")
    print("-" * 30)
    
    # Create synthetic graph
    x = torch.randn(10, 74)  # 10 nodes with 74 features
    edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8],
                              [1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index)
    
    texts = ["calculate total price with tax"]
    
    # Forward pass
    outputs = model(graph, texts)
    
    print(f"✅ Code embedding shape: {outputs['code_embeddings'].shape}")
    print(f"✅ Text embedding shape: {outputs['text_embeddings'].shape}")
    
    # Compute similarity
    similarity = F.cosine_similarity(
        outputs['code_embeddings'], 
        outputs['text_embeddings'], 
        dim=1
    )
    print(f"🎯 Code-Text similarity: {similarity.item():.4f}")
    
    # Demonstrate batch processing
    print("\n📊 Testing Batch Processing")
    print("-" * 30)
    
    # Create batch of graphs
    graphs = []
    for i in range(3):
        x_batch = torch.randn(8 + i, 74)  # Variable number of nodes
        edge_index_batch = torch.randint(0, 8 + i, (2, 10))
        graphs.append(Data(x=x_batch, edge_index=edge_index_batch))
    
    batch = Batch.from_data_list(graphs)
    texts_batch = [
        "calculate total amount",
        "process payment method", 
        "validate user input"
    ]
    
    outputs_batch = model(batch, texts_batch)
    
    print(f"✅ Batch code embeddings: {outputs_batch['code_embeddings'].shape}")
    print(f"✅ Batch text embeddings: {outputs_batch['text_embeddings'].shape}")
    
    # Compute pairwise similarities
    code_emb = outputs_batch['code_embeddings']
    text_emb = outputs_batch['text_embeddings']
    
    # Compute similarity matrix
    code_norm = F.normalize(code_emb, p=2, dim=1)
    text_norm = F.normalize(text_emb, p=2, dim=1)
    similarity_matrix = torch.mm(code_norm, text_norm.t())
    
    print("🎯 Code-Text Similarity Matrix:")
    for i, text in enumerate(texts_batch):
        print(f"   Text {i} ('{text[:20]}...'): {similarity_matrix[i].tolist()}")
    
    # Test with real data if available
    print("\n🌍 Testing with Real Data")
    print("-" * 30)
    
    try:
        # Load real data
        train_loader, _ = create_data_loaders(
            train_path="../dataset/train.jsonl",
            val_path="../dataset/validation.jsonl",
            batch_size=3,
            shuffle=False
        )
        
        # Get real sample
        sample_batch = next(iter(train_loader))
        
        # Convert to tensor format
        x_real = torch.tensor(sample_batch['x'], dtype=torch.float32)
        edge_index_real = torch.tensor(sample_batch['edge_index'], dtype=torch.long)
        batch_real = torch.tensor(sample_batch['batch'], dtype=torch.long)
        
        real_graph = Data(x=x_real, edge_index=edge_index_real, batch=batch_real)
        
        # Create sample texts for the batch
        batch_size = len(sample_batch['metadata'])
        real_texts = [
            "get user data from database",
            "calculate method complexity", 
            "process input parameters"
        ][:batch_size]
        
        # Forward pass with real data
        real_outputs = model(real_graph, real_texts)
        
        print(f"✅ Real data processing successful!")
        print(f"   Nodes: {x_real.shape[0]}, Features: {x_real.shape[1]}")
        print(f"   Batch size: {batch_size}")
        print(f"   Code embeddings: {real_outputs['code_embeddings'].shape}")
        print(f"   Text embeddings: {real_outputs['text_embeddings'].shape}")
        
        # Compute similarities for real data
        real_similarities = F.cosine_similarity(
            real_outputs['code_embeddings'], 
            real_outputs['text_embeddings'], 
            dim=1
        )
        print("🎯 Real Data Code-Text Similarities:")
        for i, sim in enumerate(real_similarities):
            print(f"   Sample {i}: {sim.item():.4f}")
            
    except FileNotFoundError:
        print("⚠️  Real dataset not found, skipping real data test")
    except Exception as e:
        print(f"⚠️  Real data test failed: {e}")
    
    print("\n✨ Demonstration completed successfully!")
    print("🎯 The AlignmentModel is ready for text-code alignment training.")

def main():
    """Run the demonstration."""
    demonstrate_alignment_model()

if __name__ == "__main__":
    main()