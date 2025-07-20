#!/usr/bin/env python3
"""
Script to split paired_data.jsonl into separate training and validation sets.

This script addresses the issue where the same dataset was used for both training
and validation, leading to overly optimistic validation metrics. It creates proper
train/validation splits to ensure reliable model evaluation.
"""

import json
import random
import os
import sys
from typing import List, Dict, Any


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: str) -> None:
    """Save data to a JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def split_dataset(input_path: str, output_dir: str, train_ratio: float = 0.9, seed: int = 42) -> None:
    """
    Split the paired dataset into training and validation sets.
    
    Args:
        input_path: Path to the original paired_data.jsonl file
        output_dir: Directory to save the split files
        train_ratio: Ratio of data to use for training (default: 0.9 for 90/10 split)
        seed: Random seed for reproducible splits
    """
    print(f"Loading dataset from {input_path}")
    data = load_jsonl(input_path)
    total_samples = len(data)
    
    print(f"Total samples: {total_samples:,}")
    
    # Set random seed for reproducible splits
    random.seed(seed)
    
    # Shuffle the data
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    # Calculate split indices
    train_size = int(total_samples * train_ratio)
    val_size = total_samples - train_size
    
    # Split the data
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:]
    
    print(f"Train samples: {len(train_data):,} ({len(train_data)/total_samples*100:.1f}%)")
    print(f"Validation samples: {len(val_data):,} ({len(val_data)/total_samples*100:.1f}%)")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the split datasets
    train_path = os.path.join(output_dir, 'train_paired_data.jsonl')
    val_path = os.path.join(output_dir, 'validation_paired_data.jsonl')
    
    print(f"Saving training data to {train_path}")
    save_jsonl(train_data, train_path)
    
    print(f"Saving validation data to {val_path}")
    save_jsonl(val_data, val_path)
    
    print("✅ Dataset split completed successfully!")
    
    # Verify the splits
    train_check = load_jsonl(train_path)
    val_check = load_jsonl(val_path)
    
    print(f"Verification - Train: {len(train_check):,}, Val: {len(val_check):,}, Total: {len(train_check) + len(val_check):,}")
    
    # Check for overlap (should be none)
    train_ids = {item['id'] for item in train_check}
    val_ids = {item['id'] for item in val_check}
    overlap = train_ids.intersection(val_ids)
    
    if overlap:
        print(f"⚠️  Warning: Found {len(overlap)} overlapping samples between train and validation sets")
    else:
        print("✅ No overlap between train and validation sets - split is clean!")


def main():
    """Main function to run the dataset splitting."""
    # Set up paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    input_path = os.path.join(repo_root, 'dataset', 'paired_data.jsonl')
    output_dir = os.path.join(repo_root, 'dataset')
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Check if output files already exist
    train_path = os.path.join(output_dir, 'train_paired_data.jsonl')
    val_path = os.path.join(output_dir, 'validation_paired_data.jsonl')
    
    if os.path.exists(train_path) or os.path.exists(val_path):
        response = input("Split files already exist. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)
    
    # Perform the split
    print("Splitting dataset with 90/10 train/validation ratio...")
    split_dataset(input_path, output_dir, train_ratio=0.9, seed=42)


if __name__ == "__main__":
    main()