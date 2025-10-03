#!/usr/bin/env python3
"""
Training script for the Hierarchical AST Decoder.

This script trains a coarse-to-fine model that generates Abstract Syntax Trees
level by level, as outlined in `docs/README_phase4b.md`.
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import logging
import torch_geometric
from torch_geometric.utils import to_dense_adj

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import HierarchicalASTDecoder, AlignmentModel
from data_processing import create_hierarchical_paired_data_loader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main training function."""
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser(description='Train a Hierarchical AST Decoder.')
    parser.add_argument(
        '--dataset-dir', type=str, default='dataset/hierarchical',
        help='Directory containing the hierarchical dataset files.'
    )
    parser.add_argument(
        '--output-dir', type=str, default='models/hierarchical',
        help='Directory to save the trained models.'
    )
    parser.add_argument(
        '--alignment-model-path', type=str, default='models/best_alignment_model.pt',
        help='Path to the pre-trained alignment model for text embeddings.'
    )
    parser.add_argument('--embedding-dim', type=int, default=64, help='Dimension of input embeddings.')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension for GNNs.')
    parser.add_argument('--max-levels', type=int, default=20, help='Maximum number of levels to train.')
    parser.add_argument('--epochs-per-level', type=int, default=5, help='Number of training epochs for each level.')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--debug-max-samples', type=int, default=None, help='Limit dataset size for fast debugging.')
    args = parser.parse_args()

    logging.info("🚀 Starting Hierarchical AST Decoder Training...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Get feature dimension from a sample dataset
    try:
        sample_loader = create_hierarchical_paired_data_loader(
            str(Path(args.dataset_dir) / "train_paired_data_level_0.jsonl"), 1, False, limit=args.debug_max_samples
        )
        sample_batch, _ = next(iter(sample_loader))
        feature_dim = len(sample_batch['x'][0])
    except (StopIteration, IndexError, FileNotFoundError):
        feature_dim = 74 # Fallback to default
        logging.warning("Could not determine feature dimension from data, using default of 74.")

    # Initialize Hierarchical Decoder Model
    model = HierarchicalASTDecoder(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_levels=args.max_levels,
        node_feature_dim=feature_dim,
        conv_type='GCN'
    ).to(device)
    
    # Initialize and load pre-trained AlignmentModel for text encoding
    alignment_model = AlignmentModel(
        input_dim=feature_dim,
        hidden_dim=args.embedding_dim
    ).to(device)
    
    try:
        alignment_model.load_state_dict(torch.load(args.alignment_model_path, map_location=device)['model_state_dict'])
        alignment_model.eval()
        logging.info(f"Successfully loaded pre-trained alignment model from {args.alignment_model_path}")
    except Exception as e:
        logging.error(f"Could not load alignment model: {e}. Text embeddings will be random.")

    logging.info(f"Initialized Hierarchical Decoder with {args.max_levels} levels and feature dimension {feature_dim}.")

    # --- Training Loop ---
    for level in range(args.max_levels):
        logging.info(f"--- Starting Training for Level {level} ---")
        
        dataset_path = Path(args.dataset_dir) / f"train_paired_data_level_{level}.jsonl"
        if not dataset_path.exists():
            logging.warning(f"Dataset for level {level} not found. Training stopped.")
            break
            
        try:
            train_loader = create_hierarchical_paired_data_loader(
                dataset_path=str(dataset_path),
                batch_size=args.batch_size,
                shuffle=True,
                limit=args.debug_max_samples
            )
        except Exception as e:
            logging.error(f"Failed to create data loader for level {level}: {e}")
            continue

        optimizer = torch.optim.Adam(model.level_generators[level].parameters(), lr=args.learning_rate)
        feature_loss_fn = torch.nn.MSELoss()
        adjacency_loss_fn = torch.nn.BCEWithLogitsLoss()

        model.train()
        for epoch in range(args.epochs_per_level):
            total_loss = 0
            for batch_graphs, text_descriptions in train_loader:
                optimizer.zero_grad()

                true_features = torch.tensor(batch_graphs['x'], dtype=torch.float, device=device)
                batch_vector = torch.tensor(batch_graphs['batch'], dtype=torch.long, device=device)
                num_nodes_per_graph = torch.bincount(batch_vector)
                
                edge_index = torch.tensor(batch_graphs['edge_index'], dtype=torch.long, device=device)
                
                # Create a single dense adjacency matrix for the whole batch
                true_adjacency = to_dense_adj(edge_index, max_num_nodes=true_features.shape[0]).squeeze(0)

                with torch.no_grad():
                    input_tensor = alignment_model.encode_text(list(text_descriptions))

                expanded_input = input_tensor.repeat_interleave(num_nodes_per_graph, dim=0)

                predictions = model(expanded_input, target_level=level)
                pred_features = predictions['pred_features']
                pred_adjacency = predictions['pred_adjacency']

                # Ensure dimensions match for loss calculation
                num_nodes = true_features.shape[0]
                pred_adjacency = pred_adjacency[:num_nodes, :num_nodes]

                feature_loss = feature_loss_fn(pred_features, true_features)
                adj_loss = adjacency_loss_fn(pred_adjacency, true_adjacency)
                
                loss = feature_loss + adj_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if len(train_loader) > 0:
                avg_loss = total_loss / len(train_loader)
                logging.info(f"Level {level}, Epoch {epoch+1}/{args.epochs_per_level}, Avg. Loss: {avg_loss:.4f}")
            else:
                logging.warning(f"No batches were processed for Level {level}.")

        level_model_path = output_dir / f"hierarchical_decoder_level_{level}.pt"
        torch.save(model.level_generators[level].state_dict(), level_model_path)
        logging.info(f"Saved model for level {level} to {level_model_path}")

    logging.info("✅ Hierarchical AST Decoder training complete!")

if __name__ == '__main__':
    main()