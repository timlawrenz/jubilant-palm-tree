#!/usr/bin/env python3
"""
Example usage of the Ruby AST Dataset and DataLoader implementation.

This script demonstrates how to use the RubyASTDataset class and SimpleDataLoader
to load and process Ruby method data for GNN training.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import (
    RubyASTDataset, 
    SimpleDataLoader, 
    create_data_loaders,
    ASTGraphConverter
)


def example_basic_usage():
    """Demonstrate basic dataset usage."""
    print("🔍 Basic Dataset Usage Example")
    print("-" * 40)
    
    # Load a dataset
    dataset = RubyASTDataset("dataset/train.jsonl")
    print(f"Dataset size: {len(dataset)}")
    print(f"Feature dimension: {dataset.get_feature_dim()}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample structure:")
    for key, value in sample.items():
        if key in ['x', 'edge_index', 'y', 'batch']:
            if key == 'x':
                print(f"  {key}: shape ({len(value)}, {len(value[0]) if value else 0})")
            elif key == 'edge_index':
                print(f"  {key}: shape ({len(value)}, {len(value[0])})")
            elif key == 'y':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")


def example_dataloader_usage():
    """Demonstrate DataLoader usage."""
    print("\n🔄 DataLoader Usage Example")
    print("-" * 40)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl", 
        "dataset/validation.jsonl", 
        batch_size=8
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Validation loader: {len(val_loader)} batches")
    
    # Process a few batches
    print("\nProcessing first 3 training batches:")
    for i, batch in enumerate(train_loader):
        if i >= 3:
            break
        
        print(f"  Batch {i+1}:")
        print(f"    Graphs: {batch['num_graphs']}")
        print(f"    Total nodes: {len(batch['x'])}")
        print(f"    Total edges: {len(batch['edge_index'][0])}")
        print(f"    Target values: {batch['y']}")


def example_ast_conversion():
    """Demonstrate AST to graph conversion."""
    print("\n🌳 AST Conversion Example")
    print("-" * 40)
    
    converter = ASTGraphConverter()
    
    # Example AST structures
    examples = [
        {
            'name': 'Simple method',
            'ast': '{"type":"def","children":["hello",{"type":"args","children":[]},{"type":"str","children":["world"]}]}'
        },
        {
            'name': 'Method with conditional',
            'ast': '{"type":"def","children":["check",{"type":"args","children":["x"]},{"type":"if","children":[{"type":"lvar","children":["x"]},{"type":"true","children":[]},{"type":"false","children":[]}]}]}'
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        graph = converter.parse_ast_json(example['ast'])
        print(f"  Nodes: {len(graph['x'])}")
        print(f"  Edges: {len(graph['edge_index'][0])}")
        print(f"  Node types: {[i for i, feat in enumerate(graph['x']) if feat[i] == 1.0][:5]}...")


def example_pytorch_compatibility():
    """Show how to make the data PyTorch compatible."""
    print("\n🔥 PyTorch Compatibility Example")
    print("-" * 40)
    
    dataset = RubyASTDataset("dataset/train.jsonl")
    sample = dataset[0]
    
    print("Converting to PyTorch tensors (when available):")
    print("```python")
    print("# When PyTorch is available:")
    print("import torch")
    print("import torch.nn.functional as F")
    print("from torch_geometric.data import Data, Batch")
    print("")
    print("# Convert sample to PyTorch Data object")
    print("data = Data(")
    print("    x=torch.tensor(sample['x'], dtype=torch.float),")
    print("    edge_index=torch.tensor(sample['edge_index'], dtype=torch.long),")
    print("    y=torch.tensor(sample['y'], dtype=torch.float)")
    print(")")
    print("")
    print("# Batch multiple graphs")
    print("batch = Batch.from_data_list([data1, data2, data3])")
    print("```")
    
    print(f"\nCurrent sample data shapes:")
    print(f"  x: ({len(sample['x'])}, {len(sample['x'][0])})")
    print(f"  edge_index: ({len(sample['edge_index'])}, {len(sample['edge_index'][0])})")
    print(f"  y: {sample['y']}")


def example_training_preparation():
    """Show how data would be used in training."""
    print("\n🏋️ Training Preparation Example")
    print("-" * 40)
    
    # Create loaders
    train_loader, val_loader = create_data_loaders(
        "dataset/train.jsonl",
        "dataset/validation.jsonl", 
        batch_size=16
    )
    
    print("Simulated training loop:")
    print("```python")
    print("model = RubyComplexityGNN(input_dim=74, hidden_dim=64)")
    print("optimizer = torch.optim.Adam(model.parameters(), lr=0.001)")
    print("criterion = torch.nn.MSELoss()")
    print("")
    print("for epoch in range(num_epochs):")
    print("    for batch in train_loader:")
    print("        # Convert to PyTorch tensors")
    print("        x = torch.tensor(batch['x'], dtype=torch.float)")
    print("        edge_index = torch.tensor(batch['edge_index'], dtype=torch.long)") 
    print("        y = torch.tensor(batch['y'], dtype=torch.float)")
    print("        batch_idx = torch.tensor(batch['batch'], dtype=torch.long)")
    print("")
    print("        # Create PyTorch Geometric Data object")
    print("        data = Data(x=x, edge_index=edge_index, batch=batch_idx)")
    print("")
    print("        # Forward pass")
    print("        pred = model(data)")
    print("        loss = criterion(pred.squeeze(), y)")
    print("")
    print("        # Backward pass")
    print("        optimizer.zero_grad()")
    print("        loss.backward()")
    print("        optimizer.step()")
    print("```")
    
    # Show actual data statistics
    total_samples = 0
    total_nodes = 0
    total_edges = 0
    
    for i, batch in enumerate(train_loader):
        total_samples += batch['num_graphs']
        total_nodes += len(batch['x'])
        total_edges += len(batch['edge_index'][0])
        if i >= 10:  # Sample first 10 batches
            break
    
    print(f"\nDataset statistics (first 10 batches):")
    print(f"  Samples: {total_samples}")
    print(f"  Average nodes per graph: {total_nodes / total_samples:.1f}")
    print(f"  Average edges per graph: {total_edges / total_samples:.1f}")


def main():
    """Run all examples."""
    print("🚀 Ruby AST Dataset Usage Examples")
    print("=" * 50)
    
    try:
        example_basic_usage()
        example_dataloader_usage()
        example_ast_conversion()
        example_pytorch_compatibility()
        example_training_preparation()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed successfully!")
        print("\nThe Ruby AST Dataset implementation is ready for use with:")
        print("  • Custom Dataset class for loading JSONL files")
        print("  • AST to graph conversion with node type encoding")
        print("  • Batch collation for efficient training")
        print("  • Simple DataLoader implementation")
        print("  • Full compatibility with PyTorch Geometric (when available)")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)