#!/usr/bin/env python3
"""
Example usage of the GNN models for Ruby complexity prediction.

This script demonstrates how to use the enhanced RubyComplexityGNN models
with both GCN and SAGE convolution types, processing batched data from
the DataLoader as specified in the requirements.
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset, create_data_loaders
from models import RubyComplexityGNN


def demonstrate_gnn_architecture():
    """Demonstrate the GNN architecture components."""
    print("🏗️  GNN Architecture Demonstration")
    print("=" * 50)
    
    # Show the architecture components
    print("GNN Model Components:")
    print("1. ✅ Several SAGEConv or GCNConv layers for message passing")
    print("2. ✅ Global pooling layer (global_mean_pool) for graph-level embedding")
    print("3. ✅ Final linear regression head to output complexity score")
    print("4. ✅ Implemented as torch.nn.Module")
    
    # Create model instances
    gcn_model = RubyComplexityGNN(
        input_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        dropout=0.1
    )
    
    sage_model = RubyComplexityGNN(
        input_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='SAGE',
        dropout=0.1
    )
    
    print(f"\n📊 GCN Model: {gcn_model.get_model_info()}")
    print(f"   Parameters: {sum(p.numel() for p in gcn_model.parameters()):,}")
    
    print(f"\n📊 SAGE Model: {sage_model.get_model_info()}")
    print(f"   Parameters: {sum(p.numel() for p in sage_model.parameters()):,}")


def demonstrate_batch_processing():
    """Demonstrate processing batches from the DataLoader (Ticket 6)."""
    print("\n\n🔄 DataLoader Batch Processing (Ticket 6 Integration)")
    print("=" * 50)
    
    # Create data loaders as defined in Ticket 6
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=8
    )
    
    print(f"✅ DataLoader created successfully:")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Process a batch with both model types
    batch = next(iter(train_loader))
    
    # Convert to PyTorch Geometric format
    x = torch.tensor(batch['x'], dtype=torch.float)
    edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)
    y = torch.tensor(batch['y'], dtype=torch.float)
    batch_idx = torch.tensor(batch['batch'], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    print(f"\n📦 Sample Batch Information:")
    print(f"   Number of graphs: {batch['num_graphs']}")
    print(f"   Total nodes: {x.size(0)}")
    print(f"   Total edges: {edge_index.size(1)}")
    print(f"   Node feature dimension: {x.size(1)}")
    print(f"   Target complexity scores: {[f'{score:.2f}' for score in y.tolist()]}")
    
    # Test both model types
    models = {
        'GCN': RubyComplexityGNN(input_dim=74, hidden_dim=64, conv_type='GCN'),
        'SAGE': RubyComplexityGNN(input_dim=74, hidden_dim=64, conv_type='SAGE')
    }
    
    for model_name, model in models.items():
        model.eval()
        
        with torch.no_grad():
            predictions = model(data)
        
        print(f"\n🔮 {model_name} Model Predictions:")
        print(f"   Model: {model.get_model_info()}")
        print(f"   Output shape: {predictions.shape}")
        
        # Show predictions vs targets
        for i, (pred, target) in enumerate(zip(predictions.squeeze(), y)):
            print(f"   Graph {i+1}: Predicted={pred.item():.4f}, Target={target.item():.2f}")


def demonstrate_training_setup():
    """Demonstrate how to set up training with the models."""
    print("\n\n🏋️  Training Setup Demonstration")
    print("=" * 50)
    
    # Create model
    model = RubyComplexityGNN(
        input_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='SAGE',  # Using SAGE as specified in requirements
        dropout=0.1
    )
    
    # Set up training components
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()
    
    print("✅ Training components initialized:")
    print(f"   Model: {model.get_model_info()}")
    print(f"   Optimizer: Adam (lr=0.001)")
    print(f"   Loss function: MSELoss")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=32
    )
    
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Simulate one training step
    model.train()
    batch = next(iter(train_loader))
    
    # Convert to PyTorch format
    x = torch.tensor(batch['x'], dtype=torch.float)
    edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)
    y = torch.tensor(batch['y'], dtype=torch.float)
    batch_idx = torch.tensor(batch['batch'], dtype=torch.long)
    
    data = Data(x=x, edge_index=edge_index, batch=batch_idx)
    
    # Forward pass
    predictions = model(data)
    loss = criterion(predictions.squeeze(), y)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"\n📈 Sample Training Step:")
    print(f"   Batch size: {batch['num_graphs']}")
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Predictions range: [{predictions.min().item():.4f}, {predictions.max().item():.4f}]")
    print(f"   Targets range: [{y.min().item():.2f}, {y.max().item():.2f}]")


def demonstrate_model_flexibility():
    """Demonstrate the flexibility of model configurations."""
    print("\n\n⚙️  Model Configuration Flexibility")
    print("=" * 50)
    
    # Different model configurations
    configs = [
        {
            'name': 'Lightweight GCN',
            'params': {'input_dim': 74, 'hidden_dim': 32, 'num_layers': 2, 'conv_type': 'GCN'}
        },
        {
            'name': 'Standard SAGE',
            'params': {'input_dim': 74, 'hidden_dim': 64, 'num_layers': 3, 'conv_type': 'SAGE'}
        },
        {
            'name': 'Deep GCN',
            'params': {'input_dim': 74, 'hidden_dim': 128, 'num_layers': 4, 'conv_type': 'GCN', 'dropout': 0.2}
        },
        {
            'name': 'Deep SAGE',
            'params': {'input_dim': 74, 'hidden_dim': 128, 'num_layers': 4, 'conv_type': 'SAGE', 'dropout': 0.2}
        }
    ]
    
    # Load a sample for testing
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)
    
    for config in configs:
        model = RubyComplexityGNN(**config['params'])
        model.eval()
        
        with torch.no_grad():
            output = model(data)
        
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"📊 {config['name']}:")
        print(f"   Config: {model.get_model_info()}")
        print(f"   Parameters: {param_count:,}")
        print(f"   Sample prediction: {output.item():.4f}")


def main():
    """Run all demonstrations."""
    print("🚀 Ruby Complexity GNN Model Demonstration")
    print("✨ Meeting all requirements from the issue specification")
    
    try:
        demonstrate_gnn_architecture()
        demonstrate_batch_processing() 
        demonstrate_training_setup()
        demonstrate_model_flexibility()
        
        print("\n\n🎉 All Demonstrations Completed Successfully!")
        print("=" * 50)
        print("✅ Definition of Done - All Requirements Met:")
        print("   • The model class is defined ✅")
        print("   • It can be instantiated ✅")
        print("   • It can process sample batches from DataLoader (Ticket 6) ✅")
        print("   • Supports both SAGEConv and GCNConv layers ✅")
        print("   • Includes global pooling layer ✅")
        print("   • Has linear regression head for complexity prediction ✅")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()