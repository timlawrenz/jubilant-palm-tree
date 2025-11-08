#!/usr/bin/env python3
"""
Validates the integrity of trained Hierarchical AST Decoder model files.

This script checks that each level's model file can be loaded into the
HierarchicalASTDecoder architecture and can perform a forward pass with dummy data.
This verifies that the models are not corrupt and are compatible with the
current model definition in `src/models.py`.
"""

import argparse
import torch
import sys
import os
from pathlib import Path
import logging

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import HierarchicalASTDecoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_models(args):
    """Loads and tests each level of the hierarchical model."""
    logging.info("🚀 Starting validation of hierarchical model files...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # 1. Initialize the full model architecture as a scaffold
    try:
        full_decoder = HierarchicalASTDecoder(
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            num_levels=args.max_depth,
            node_feature_dim=args.node_feature_dim,
            conv_type='GCN'
        ).to(device)
        full_decoder.eval()
        logging.info("Successfully instantiated HierarchicalASTDecoder architecture.")
    except Exception as e:
        logging.error(f"🚨 Failed to instantiate HierarchicalASTDecoder: {e}")
        return

    # 2. Loop through each level and validate the corresponding model file
    all_levels_valid = True
    for level in range(args.max_depth):
        model_path = Path(args.model_dir) / f"hierarchical_decoder_level_{level}.pt"
        logging.info(f"--- Checking Level {level}: {model_path} ---")

        if not model_path.exists():
            logging.info(f"Model file for level {level} not found. End of validation.")
            break

        try:
            # a. Test loading the state dictionary
            state_dict = torch.load(model_path, map_location=device)
            full_decoder.level_generators[level].load_state_dict(state_dict)
            logging.info(f"✅ [Level {level}] Model file loaded successfully.")

            # b. Test forward pass with dummy data
            # For level 0, the input is the text embedding. For subsequent levels,
            # it would be the hidden state from the previous level.
            # A random tensor of the correct shape is sufficient for this test.
            # For level 0, the input is the text embedding. For subsequent levels,
            # it's the hidden state from the previous level.
            input_dim = args.embedding_dim if level == 0 else args.hidden_dim
            
            # The training script expands the input to the number of nodes.
            # Here, we simulate a single-node graph for simplicity.
            dummy_input = torch.randn(1, input_dim).to(device)

            with torch.no_grad():
                output = full_decoder(dummy_input, target_level=level)

            # c. Check output integrity
            assert 'pred_features' in output, "Output missing 'pred_features'"
            assert 'pred_adjacency' in output, "Output missing 'pred_adjacency'"
            
            pred_features_shape = output['pred_features'].shape
            pred_adjacency_shape = output['pred_adjacency'].shape

            logging.info(f"✅ [Level {level}] Forward pass successful.")
            logging.info(f"   - Predicted Features Shape: {pred_features_shape}")
            logging.info(f"   - Predicted Adjacency Shape: {pred_adjacency_shape}")

        except Exception as e:
            logging.error(f"🚨 [Level {level}] FAILED validation. Reason: {e}")
            all_levels_valid = False
            break
    
    if all_levels_valid:
        logging.info("\n✅ All found model files are valid and functional!")
    else:
        logging.error("\n🚨 Validation failed. Please check the errors above.")


def main():
    parser = argparse.ArgumentParser(description="Validate trained hierarchical model files.")
    parser.add_argument("--model_dir", type=str, default="models/hierarchical", help="Directory where the hierarchical models are saved.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for GNNs.")
    parser.add_argument("--node_feature_dim", type=int, default=74, help="Dimension of node features.")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum AST depth to check for.")
    args = parser.parse_args()
    
    validate_models(args)

if __name__ == "__main__":
    main()
