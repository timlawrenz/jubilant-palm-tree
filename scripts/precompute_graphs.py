#!/usr/bin/env python3
"""
Precomputes and caches graph representations of ASTs from JSONL datasets.

This script processes the raw JSONL datasets (train, validation, test) and
converts the ASTs into PyTorch Geometric `Data` objects. These objects are
then saved to disk as `.pt` files.

This one-time precomputation step significantly speeds up training by
eliminating the need for on-the-fly JSON parsing and graph conversion
during each epoch.
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch
from torch_geometric.data import Data

# Add src directory to path to import data_processing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import ASTGraphConverter, load_jsonl_file

def precompute_graphs(input_path, output_path):
    """
    Loads a JSONL file, converts each AST to a PyG Data object, and saves
    the list of Data objects to a .pt file.
    """
    print(f"Processing {input_path}...")
    
    # Load raw data
    raw_data = load_jsonl_file(input_path)
    
    # Initialize converter
    converter = ASTGraphConverter()
    
    # Process each item
    graph_data_objects = []
    for item in tqdm(raw_data, desc=f"Converting {os.path.basename(input_path)}"):
        # Convert AST to graph dictionary
        graph_dict = converter.parse_ast_json(item['ast_json'])
        
        # Create PyG Data object
        data = Data(
            x=torch.tensor(graph_dict['x'], dtype=torch.float),
            edge_index=torch.tensor(graph_dict['edge_index'], dtype=torch.long)
        )
        
        # Add other relevant information if needed
        data.num_nodes = graph_dict['num_nodes']
        data.y = torch.tensor([item.get('complexity_score', 5.0)], dtype=torch.float)
        
        graph_data_objects.append(data)
        
    # Save the list of Data objects
    torch.save(graph_data_objects, output_path)
    print(f"✅ Saved {len(graph_data_objects)} precomputed graphs to {output_path}")


def main():
    """Main function to run the precomputation for all dataset splits."""
    parser = argparse.ArgumentParser(description='Precompute graph data for ASTs.')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to the dataset directory (default: dataset/)')
    args = parser.parse_args()

    print("🚀 Starting Graph Precomputation")
    print("=" * 50)
    
    dataset_dir = args.dataset_path
    
    # Define input and output paths
    splits = {
        "train": ("train.jsonl", "train.pt"),
        "validation": ("validation.jsonl", "validation.pt"),
        "test": ("test.jsonl", "test.pt")
    }
    
    # Ensure output directory exists
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Process each split
    for split_name, (input_file, output_file) in splits.items():
        input_path = os.path.join(dataset_dir, input_file)
        output_path = os.path.join(dataset_dir, output_file)
        
        if os.path.exists(input_path):
            precompute_graphs(input_path, output_path)
        else:
            print(f"⚠️  Warning: Input file not found, skipping: {input_path}")

    print("\n" + "=" * 50)
    print("🎉 Precomputation complete!")

if __name__ == '__main__':
    main()
