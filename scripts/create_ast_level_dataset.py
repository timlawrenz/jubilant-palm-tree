#!/usr/bin/env python3
"""
Pre-processes the AST dataset for hierarchical, coarse-to-fine model training.

This script reads the JSONL dataset containing full Abstract Syntax Trees (ASTs)
and decomposes each AST into a series of "level-graphs". Each level-graph
represents the AST up to a certain depth, from the root downwards.

This creates the "AST pyramid" required for training the progressive,
hierarchical decoder, as outlined in `docs/README_phase4b.md`.

Example:
- level_0: Contains only the root node of the AST.
- level_1: Contains the root node and its immediate children.
- level_2: Contains the root, its children, and its grandchildren.
- ...and so on.
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_max_depth(node):
    """Recursively calculates the maximum depth of an AST."""
    if not node or 'children' not in node or not node['children']:
        return 0
    return 1 + max(get_max_depth(child) for child in node['children'])

def build_partial_ast(original_node, max_depth, current_depth=0):
    """
    Recursively reconstructs a partial AST up to a specified max_depth.
    Nodes and branches beyond the max_depth are pruned.
    """
    # If the node is not a dictionary (e.g., a string, int, or None), it's a leaf.
    if not isinstance(original_node, dict):
        return original_node

    # Handle cases where the node itself is a stringified JSON
    if isinstance(original_node, str):
        try:
            original_node = json.loads(original_node)
        except json.JSONDecodeError:
            logging.warning(f"Could not decode string node: {original_node}")
            return None

    if not original_node or current_depth > max_depth:
        return None

    # Create a copy of the node, excluding children initially
    new_node = {key: value for key, value in original_node.items() if key != 'children'}
    new_node['children'] = []

    if 'children' in original_node and original_node['children']:
        for child in original_node['children']:
            partial_child = build_partial_ast(child, max_depth, current_depth + 1)
            if partial_child:
                new_node['children'].append(partial_child)

    return new_node

def process_dataset(input_path, output_dir, max_levels):
    """
    Reads a JSONL dataset, decomposes each AST, and writes the level-graphs
    to separate output files.
    """
    # Create file handles for each output level
    output_files = {}
    try:
        for i in range(max_levels):
            # e.g., output_dir / "train_level_0.jsonl"
            file_name = f"{input_path.stem}_level_{i}.jsonl"
            output_path = output_dir / file_name
            output_files[i] = open(output_path, 'w')
            logging.info(f"Opened {output_path} for writing.")

        with open(input_path, 'r') as f:
            # Use tqdm for a progress bar
            for line in tqdm(f, desc=f"Processing {input_path.name}"):
                try:
                    method_data = json.loads(line)
                    if 'ast_json' not in method_data:
                        continue

                    full_ast = json.loads(method_data['ast_json'])
                    if not full_ast:
                        continue

                    # Generate and write the partial AST for each level
                    for i in range(max_levels):
                        partial_ast = build_partial_ast(full_ast, max_depth=i)
                        if partial_ast:
                            # Create a new record with the partial AST
                            new_record = method_data.copy()
                            new_record['ast_json'] = json.dumps(partial_ast)
                            output_files[i].write(json.dumps(new_record) + '\n')
                except (json.JSONDecodeError, TypeError) as e:
                    logging.warning(f"Skipping line due to parsing error: {e}")
                    continue

    finally:
        # Ensure all file handles are closed
        for f in output_files.values():
            f.close()
        logging.info("All output files closed.")


def main():
    """Main function to orchestrate the dataset creation."""
    parser = argparse.ArgumentParser(description='Create hierarchical AST dataset.')
    parser.add_argument(
        '--dataset-dir',
        type=str,
        default='dataset',
        help='Path to the directory containing the source dataset files (e.g., train.jsonl).'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='dataset/hierarchical',
        help='Directory to save the processed hierarchical dataset files.'
    )
    parser.add_argument(
        '--max-depth',
        type=int,
        default=20,
        help='Maximum AST depth to process. ASTs deeper than this will be truncated.'
    )
    args = parser.parse_args()

    logging.info("🚀 Starting hierarchical dataset creation...")
    
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)

    # Create the output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output will be saved to {output_dir.resolve()}")

    # Process both training and validation sets
    files_to_process = ['train_paired_data.jsonl', 'validation_paired_data.jsonl']
    for filename in files_to_process:
        input_path = dataset_dir / filename
        if input_path.exists():
            logging.info(f"Found dataset file: {input_path}")
            process_dataset(input_path, output_dir, args.max_depth)
        else:
            logging.warning(f"Dataset file not found, skipping: {input_path}")

    logging.info("✅ Hierarchical dataset creation complete!")


if __name__ == '__main__':
    main()
