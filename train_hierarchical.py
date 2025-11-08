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
from torch_geometric.data import Data
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
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers.')
    parser.add_argument('--debug-max-samples', type=int, default=None, help='Limit dataset size for fast debugging.')
    parser.add_argument('--reset', action='store_true', help='Reset training by removing all checkpoints and starting from level 0.')
    parser.add_argument('--resume', action='store_true', help='Resume training from the last checkpoint (default behavior).')
    args = parser.parse_args()

    logging.info("🚀 Starting Hierarchical AST Decoder Training...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_checkpoint.pt"
    
    # Handle reset flag
    if args.reset:
        logging.info("🔄 Reset flag detected. Removing all checkpoints and trained models...")
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logging.info(f"Removed checkpoint: {checkpoint_path}")
        for level in range(args.max_levels):
            level_model_path = output_dir / f"hierarchical_decoder_level_{level}.pt"
            if level_model_path.exists():
                level_model_path.unlink()
                logging.info(f"Removed model: {level_model_path}")
        logging.info("✅ Reset complete. Starting training from level 0.")

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

    # Load checkpoint if exists (unless reset was requested)
    start_level = 0
    if checkpoint_path.exists() and not args.reset:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            start_level = checkpoint['last_completed_level'] + 1
            logging.info(f"📂 Found checkpoint. Resuming from level {start_level}")
            
            # Load previously trained level generators
            for level in range(start_level):
                level_model_path = output_dir / f"hierarchical_decoder_level_{level}.pt"
                if level_model_path.exists():
                    model.level_generators[level].load_state_dict(torch.load(level_model_path, map_location=device))
                    logging.info(f"Loaded model for level {level}")
        except Exception as e:
            logging.warning(f"Failed to load checkpoint: {e}. Starting from level 0.")
            start_level = 0

    # --- Training Loop ---
    for level in range(start_level, args.max_levels):
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
                num_workers=args.num_workers,
                limit=args.debug_max_samples
            )
        except Exception as e:
            logging.error(f"Failed to create data loader for level {level}: {e}")
            continue

        optimizer = torch.optim.Adam(model.level_generators[level].parameters(), lr=args.learning_rate)
        feature_loss_fn = torch.nn.MSELoss()
        adjacency_loss_fn = torch.nn.BCEWithLogitsLoss()

        # The input for the first level is the text embedding. For subsequent levels,
        # it's the hidden state from the previous level's model.
        # We'll need to manage this state across batches and epochs for a given level.
        # For this training script, we'll simplify by re-calculating the base embedding each time.
        
        model.train()
        for epoch in range(args.epochs_per_level):
            total_loss = 0
            
            # We need a persistent hidden state for the next level's input
            # For simplicity in training, we will re-generate it from the text embedding for each batch
            
            for batch_graphs, text_descriptions in train_loader:
                optimizer.zero_grad()

                true_features = torch.tensor(batch_graphs['x'], dtype=torch.float, device=device)
                batch_vector = torch.tensor(batch_graphs['batch'], dtype=torch.long, device=device)
                num_nodes_per_graph = torch.bincount(batch_vector)
                
                edge_index = torch.tensor(batch_graphs['edge_index'], dtype=torch.long, device=device)
                
                true_adjacency = to_dense_adj(edge_index, max_num_nodes=true_features.shape[0]).squeeze(0)

                with torch.no_grad():
                    # Get the initial text embedding
                    text_embedding = alignment_model.encode_text(list(text_descriptions))
                
                # For level 0, use text embedding directly
                # For subsequent levels, pass through previous levels to get hidden state
                if level == 0:
                    # Level 0: input is text embedding
                    level_input_features = text_embedding
                else:
                    # Level > 0: need to get hidden state from previous levels
                    with torch.no_grad():
                        level_input_features = text_embedding
                        for i in range(level):
                            # Expand features to match current nodes and create Data object
                            expanded_features = level_input_features.repeat_interleave(num_nodes_per_graph, dim=0)
                            prev_data = Data(
                                x=expanded_features,
                                edge_index=edge_index
                            ).to(device)
                            prev_level_output = model(prev_data, target_level=i)
                            # Use hidden state as input for next level (aggregate back to batch size)
                            level_input_features = prev_level_output['hidden_state'].mean(dim=0, keepdim=True).expand(text_embedding.size(0), -1)

                # Create input Data object for current level
                expanded_input = level_input_features.repeat_interleave(num_nodes_per_graph, dim=0)
                input_data = Data(
                    x=expanded_input,
                    edge_index=edge_index
                ).to(device)

                predictions = model(input_data, target_level=level)
                pred_features = predictions['pred_features']
                
                # Note: Model now returns only diagonal adjacency for memory efficiency
                # We'll skip adjacency loss for now and focus on node type prediction
                feature_loss = feature_loss_fn(pred_features, true_features)
                
                loss = feature_loss
                
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
        
        # Save checkpoint after completing each level
        checkpoint = {
            'last_completed_level': level,
            'total_levels': args.max_levels,
            'feature_dim': feature_dim,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim
        }
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"💾 Checkpoint saved. Level {level} complete ({level+1}/{args.max_levels})")

    logging.info("✅ Hierarchical AST Decoder training complete!")

if __name__ == '__main__':
    main()