#!/usr/bin/env python3
"""
Validation script for the text-code alignment model.

This script evaluates the performance of a trained AlignmentModel by measuring its
ability to retrieve correct text descriptions for given code snippets (and vice-versa).
It calculates retrieval metrics like Recall@k to provide a quantitative measure of
alignment quality.
"""

import sys
import os
import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_paired_data_loaders
from models import AlignmentModel

def calculate_retrieval_metrics(model, data_loader, device, k_values=[1, 5, 10]):
    """
    Calculates retrieval metrics (Recall@k) for both text-to-code and code-to-text retrieval.

    Args:
        model: The trained AlignmentModel.
        data_loader: The data loader for the validation or test set.
        device: The device to run the model on.
        k_values: A list of integers for k in Recall@k.

    Returns:
        A dictionary containing the retrieval metrics.
    """
    model.eval()
    all_code_embeddings = []
    all_text_embeddings = []

    with torch.no_grad():
        for batched_graphs, text_descriptions in tqdm(data_loader, desc="Generating Embeddings"):
            # Convert graph data to PyTorch tensors and move to device
            x = torch.tensor(batched_graphs['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batched_graphs['edge_index'], dtype=torch.long).to(device)
            batch_idx = torch.tensor(batched_graphs['batch'], dtype=torch.long).to(device)
            
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            outputs = model(data, text_descriptions)
            
            code_embeddings = F.normalize(outputs['code_embeddings'], p=2, dim=1)
            text_embeddings = F.normalize(outputs['text_embeddings'], p=2, dim=1)
            
            all_code_embeddings.append(code_embeddings.cpu())
            all_text_embeddings.append(text_embeddings.cpu())

    all_code_embeddings = torch.cat(all_code_embeddings, dim=0)
    all_text_embeddings = torch.cat(all_text_embeddings, dim=0)
    
    num_samples = all_code_embeddings.size(0)
    
    # --- Text-to-Code Retrieval ---
    text_to_code_similarity = torch.matmul(all_text_embeddings, all_code_embeddings.t())
    t2c_recall = {k: 0.0 for k in k_values}

    for i in tqdm(range(num_samples), desc="Text-to-Code Recall"):
        similarities = text_to_code_similarity[i]
        sorted_indices = torch.argsort(similarities, descending=True)
        for k in k_values:
            if i in sorted_indices[:k]:
                t2c_recall[k] += 1

    for k in k_values:
        t2c_recall[k] /= num_samples

    # --- Code-to-Text Retrieval ---
    code_to_text_similarity = torch.matmul(all_code_embeddings, all_text_embeddings.t())
    c2t_recall = {k: 0.0 for k in k_values}

    for i in tqdm(range(num_samples), desc="Code-to-Text Recall"):
        similarities = code_to_text_similarity[i]
        sorted_indices = torch.argsort(similarities, descending=True)
        for k in k_values:
            if i in sorted_indices[:k]:
                c2t_recall[k] += 1
    
    for k in k_values:
        c2t_recall[k] /= num_samples

    return {
        "text_to_code_recall": t2c_recall,
        "code_to_text_recall": c2t_recall
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Validate text-code alignment model')
    parser.add_argument('--dataset_path', type=str, default='dataset/',
                        help='Path to dataset directory (default: dataset/)')
    parser.add_argument('--model_path', type=str, default='models/best_alignment_model.pt',
                        help='Path to the saved alignment model (default: models/best_alignment_model.pt)')
    parser.add_argument('--code_encoder_weights_path', type=str, default='models/best_model.pt',
                        help='Path to pre-trained code encoder weights (default: models/best_model.pt)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for validation (default: 32)')
    return parser.parse_args()


def main():
    """Main validation function."""
    args = parse_args()

    print("🚀 Starting Alignment Model Validation")
    print("=" * 50)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load validation dataset
    val_data_path = os.path.join(args.dataset_path, "validation_paired_data.jsonl")
    print(f"\n📊 Loading validation dataset from {val_data_path}")
    try:
        val_loader = create_paired_data_loaders(
            paired_data_path=val_data_path,
            batch_size=args.batch_size,
            shuffle=False
        )
        print(f"✅ Loaded validation data: {len(val_loader.dataset)} samples in {len(val_loader)} batches")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    # Initialize AlignmentModel
    print(f"\n🧠 Initializing AlignmentModel from {args.model_path}")
    try:
        # Get feature dimension from dataset
        sample_batch = next(iter(val_loader))
        feature_dim = len(sample_batch[0]['x'][0])

        model = AlignmentModel(
            input_dim=feature_dim,
            hidden_dim=64,
            num_layers=3,
            conv_type='GCN',
            dropout=0.1,
            text_model_name='all-MiniLM-L6-v2',
            code_encoder_weights_path=args.code_encoder_weights_path
        )
        
        # Load the trained state dict
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        print(f"✅ Model initialized and weights loaded")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return

    # Calculate metrics
    print("\n📈 Calculating retrieval metrics...")
    metrics = calculate_retrieval_metrics(model, val_loader, device)

    print("\n" + "=" * 50)
    print("🏁 Validation Completed!")
    print("\n--- Text-to-Code Retrieval ---")
    for k, v in metrics['text_to_code_recall'].items():
        print(f"Recall@{k}: {v:.4f}")

    print("\n--- Code-to-Text Retrieval ---")
    for k, v in metrics['code_to_text_recall'].items():
        print(f"Recall@{k}: {v:.4f}")
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()