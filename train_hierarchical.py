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

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import HierarchicalASTDecoder, AlignmentModel
from data_processing import create_hierarchical_paired_data_loader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main training function."""
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
    args = parser.parse_args()

    logging.info("🚀 Starting Hierarchical AST Decoder Training...")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Initialize Hierarchical Decoder Model
    model = HierarchicalASTDecoder(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_levels=args.max_levels,
        conv_type='GCN'
    ).to(device)
    
    # Initialize and load pre-trained AlignmentModel for text encoding
    # We need the feature_dim from a sample, let's get it from level 0
    try:
        sample_loader = create_hierarchical_paired_data_loader(
            str(Path(args.dataset_dir) / "train_paired_data_level_0.jsonl"), 1, False
        )
        sample_batch, _ = next(iter(sample_loader))
        feature_dim = len(sample_batch['x'][0])
    except (StopIteration, IndexError):
        feature_dim = 74 # Fallback to default if dataset is empty
        logging.warning("Could not determine feature dimension from data, using default.")

    alignment_model = AlignmentModel(
        input_dim=feature_dim,
        hidden_dim=args.embedding_dim
    ).to(device)
    
    try:
        alignment_model.load_state_dict(torch.load(args.alignment_model_path, map_location=device)['model_state_dict'])
        alignment_model.eval() # Set to evaluation mode
        logging.info(f"Successfully loaded pre-trained alignment model from {args.alignment_model_path}")
    except Exception as e:
        logging.error(f"Could not load alignment model: {e}. Text embeddings will be random.")

    logging.info(f"Initialized Hierarchical Decoder with {args.max_levels} levels.")

    # --- Training Loop ---
    for level in range(args.max_levels):
        logging.info(f"--- Starting Training for Level {level} ---")
        
        dataset_path = Path(args.dataset_dir) / f"train_paired_data_level_{level}.jsonl"
        if not dataset_path.exists():
            logging.warning(f"Dataset for level {level} not found. Skipping.")
            continue
            
        try:
            train_loader = create_hierarchical_paired_data_loader(
                dataset_path=str(dataset_path),
                batch_size=args.batch_size,
                shuffle=True
            )
        except Exception as e:
            logging.error(f"Failed to create data loader for level {level}: {e}")
            continue

        optimizer = torch.optim.Adam(model.level_generators[level].parameters(), lr=args.learning_rate)
        loss_fn = torch.nn.MSELoss()

        model.train()
        for epoch in range(args.epochs_per_level):
            total_loss = 0
            for batch_graphs, text_descriptions in train_loader:
                optimizer.zero_grad()
                
                # 1. Generate Text Embedding (Input to the model)
                with torch.no_grad():
                    text_embeddings = alignment_model.encode_text(text_descriptions)

                # 2. Forward pass through the hierarchical model for the current level
                output = model(text_embeddings, target_level=level)
                
                # 3. Placeholder for target.
                # In a real scenario, this would be the ground-truth graph representation for this level.
                dummy_target = torch.randn_like(output)
                
                loss = loss_fn(output, dummy_target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            logging.info(f"Level {level}, Epoch {epoch+1}/{args.epochs_per_level}, Avg. Loss: {avg_loss:.4f}")

        level_model_path = output_dir / f"hierarchical_decoder_level_{level}.pt"
        torch.save(model.level_generators[level].state_dict(), level_model_path)
        logging.info(f"Saved model for level {level} to {level_model_path}")

    logging.info("✅ Hierarchical AST Decoder training complete!")

if __name__ == '__main__':
    main()
