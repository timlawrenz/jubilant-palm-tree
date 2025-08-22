#!/usr/bin/env python3
"""
Pre-collates graph data into batches and saves them to disk.

This script takes the precomputed graph data (e.g., train.pt) and collates
it into batches of a specified size. These pre-collated batches are then
saved as a list of `torch_geometric.data.Batch` objects.

This eliminates the need for real-time collation during training, moving the
last significant CPU-bound task into a one-time offline process. The result
is a highly efficient training pipeline where the main thread only needs to
load an already-prepared batch and send it to the GPU.
"""

import os
import sys
import argparse
from tqdm import tqdm
import torch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_processing import PrecomputedRubyASTDataset

# Check if torch_geometric is available
try:
    from torch_geometric.loader import DataLoader
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

def pre_collate_batches(input_path, output_path, batch_size, num_workers):
    """
    Loads a .pt file of individual graphs, collates them into batches,
    and saves the list of Batch objects to a new .pt file.
    """
    if not TORCH_GEOMETRIC_AVAILABLE:
        print("❌ torch_geometric is not installed. Please install it to run pre-collation.")
        return

    print(f"Processing {input_path} with batch size {batch_size}...")
    
    # 1. Load the dataset of individual graphs
    dataset = PrecomputedRubyASTDataset(input_path)
    
    # 2. Use a DataLoader to perform the collation
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # Shuffle during training, not here
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )
    
    # 3. Iterate through the loader to create the collated batches
    collated_batches = [batch for batch in tqdm(loader, desc=f"Collating {os.path.basename(input_path)}")]
    
    # 4. Save the list of pre-collated Batch objects
    torch.save(collated_batches, output_path)
    print(f"✅ Saved {len(collated_batches)} pre-collated batches to {output_path}")


def main():
    """Main function to run pre-collation for all dataset splits."""
    parser = argparse.ArgumentParser(description='Pre-collate graph data into batches.')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to the dataset directory (default: dataset/)')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Batch size to use for pre-collation (default: 4096)')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading (default: all CPU cores)')
    args = parser.parse_args()

    print("🚀 Starting Batch Pre-Collation")
    print("=" * 50)
    
    dataset_dir = args.dataset_path
    num_workers = args.num_workers if args.num_workers is not None else os.cpu_count()
    
    # Define input and output paths
    splits = {
        "train": ("train.pt", f"train_collated_b{args.batch_size}.pt"),
        "validation": ("validation.pt", f"validation_collated_b{args.batch_size}.pt"),
    }
    
    # Process each split
    for split_name, (input_file, output_file) in splits.items():
        input_path = os.path.join(dataset_dir, input_file)
        output_path = os.path.join(dataset_dir, output_file)
        
        if os.path.exists(input_path):
            pre_collate_batches(input_path, output_path, args.batch_size, num_workers)
        else:
            print(f"⚠️  Warning: Input file not found, skipping: {input_path}")

    print("\n" + "=" * 50)
    print("🎉 Pre-collation complete!")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Next, run training pointing to the new `_collated_b{args.batch_size}.pt` files.")

if __name__ == '__main__':
    main()
