#!/usr/bin/env python3
"""
Evaluation Script for Hierarchical AST Decoder

This script evaluates the performance of the Hierarchical AST Decoder by
assessing its ability to reconstruct Ruby method ASTs level by level.
"""

import os
import sys
import json
import subprocess
import pandas as pd
import torch
from torch_geometric.data import Data
import networkx as nx
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import argparse

# Add src to path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.models import HierarchicalASTDecoder, AlignmentModel
from src.data_processing import ASTGraphConverter

# --- Configuration ---
DEFAULT_MODEL_DIR = "models/hierarchical"
DEFAULT_TEST_DATASET_PATH = "dataset/test.jsonl"
DEFAULT_ALIGNMENT_MODEL_PATH = "models/best_alignment_model.pt"
DEFAULT_NUM_SAMPLES = 100

# --- Helper Functions (copied from evaluate_model.py) ---
def run_ruby_script(script_path, input_text):
    """Helper to run a Ruby script with input text via stdin."""
    try:
        process = subprocess.run(
            ["ruby", script_path],
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
            timeout=30
        )
        return process.stdout.strip()
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        error_output = e.stderr.strip() if hasattr(e, "stderr") and e.stderr else str(e)
        return f"RUBY_SCRIPT_ERROR: {error_output}"

def check_syntax(code, script_path):
    """Checks if a string of Ruby code has valid syntax."""
    if code.startswith("RUBY_SCRIPT_ERROR"): return False
    result = run_ruby_script(script_path, code)
    return result == "OK"

def data_to_networkx(data):
    """Converts a PyG Data object to a NetworkX graph for isomorphism checking."""
    g = nx.Graph()
    node_types = data.x.argmax(dim=1).tolist()
    for i, node_type in enumerate(node_types):
        g.add_node(i, type=node_type)
    edges = data.edge_index.t().tolist()
    g.add_edges_from(edges)
    return g

def check_isomorphism(data1, data2):
    """Checks if two PyG Data objects are isomorphic based on node types and structure."""
    if data1.num_nodes != data2.num_nodes or data1.num_edges != data2.num_edges:
        return False
    g1 = data_to_networkx(data1)
    g2 = data_to_networkx(data2)
    return nx.is_isomorphic(g1, g2, node_match=lambda n1, n2: n1["type"] == n2["type"])

def calculate_bleu(reference, candidate):
    """Calculates sentence BLEU score with smoothing."""
    if candidate.startswith("RUBY_SCRIPT_ERROR"): return 0.0
    ref_tokens = reference.split()
    can_tokens = candidate.split()
    chencherry = SmoothingFunction()
    return sentence_bleu([ref_tokens], can_tokens, smoothing_function=chencherry.method1)

def load_hierarchical_model(model_dir, device, args):
    """Loads the hierarchical model from a directory."""
    print(f"🧠 Loading hierarchical model from {model_dir}...")
    model = HierarchicalASTDecoder(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_levels=args.max_depth,
        node_feature_dim=args.node_feature_dim,
        conv_type='GCN'
    ).to(device)

    for level in range(args.max_depth):
        model_path = os.path.join(model_dir, f"hierarchical_decoder_level_{level}.pt")
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.level_generators[level].load_state_dict(state_dict)
        else:
            print(f"Stopping at level {level-1}, model file not found.")
            model.num_levels = level
            break
    
    model.eval()
    print("✅ Model loaded successfully.")
    return model

def main(args):
    """Main evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # --- Script paths ---
    scripts_dir = os.path.join(module_path, 'scripts')
    pretty_print_script = os.path.join(scripts_dir, "pretty_print_ast.rb")
    check_syntax_script = os.path.join(scripts_dir, "check_syntax.rb")
    normalize_code_script = os.path.join(scripts_dir, "normalize_code.rb")

    # --- Load Model and Data ---
    model = load_hierarchical_model(args.model_dir, device, args)

    # Load AlignmentModel for text encoding
    alignment_model = AlignmentModel(
        input_dim=args.node_feature_dim, # Assuming node_feature_dim is appropriate for input_dim
        hidden_dim=args.embedding_dim
    ).to(device)
    try:
        alignment_model.load_state_dict(torch.load(args.alignment_model_path, map_location=device)['model_state_dict'])
        alignment_model.eval()
        print(f"✅ Successfully loaded pre-trained alignment model from {args.alignment_model_path}")
    except Exception as e:
        print(f"❌ Could not load alignment model: {e}. Evaluation will use random text embeddings.")

    print(f"📂 Loading dataset from {args.dataset_path}...")
    with open(args.dataset_path, "r") as f:
        test_data = [json.loads(line) for line in f]
    dataset_sample = test_data[:args.num_samples]
    print(f"Loaded {len(dataset_sample)} samples for evaluation.")

    results = []
    converter = ASTGraphConverter()

    with torch.no_grad():
        for item in tqdm(dataset_sample, desc="Evaluating Samples"):
            original_code = item["raw_source"]
            
            graph_dict = converter.parse_ast_json(item["ast_json"])
            original_data = Data(
                x=torch.tensor(graph_dict["x"], dtype=torch.float),
                edge_index=torch.tensor(graph_dict["edge_index"], dtype=torch.long)
            )
            original_data.batch = torch.zeros(original_data.num_nodes, dtype=torch.long)
            original_data = original_data.to(device)

            if original_data.num_nodes == 0: continue

            # --- Hierarchical Reconstruction using generate() ---
            # Use alignment model to get initial embedding
            text_embedding = alignment_model.encode_text([original_code])
            
            # Use the new generate() method
            try:
                generated_ast = model.generate(
                    embedding=text_embedding.squeeze(0),
                    max_levels=args.max_depth,
                    max_nodes_per_level=10
                )
                
                # Map type indices to actual node type strings
                def map_types_in_ast(ast_node, converter):
                    if isinstance(ast_node, dict):
                        type_str = ast_node.get('type', 'unknown')
                        # Extract numeric index from 'type_N' format
                        if type_str.startswith('type_'):
                            try:
                                type_idx = int(type_str.split('_')[1])
                                if type_idx < len(converter.node_encoder.node_types):
                                    ast_node['type'] = converter.node_encoder.node_types[type_idx]
                                else:
                                    ast_node['type'] = 'unknown'
                            except (ValueError, IndexError):
                                ast_node['type'] = 'unknown'
                        
                        # Recursively map children
                        if 'children' in ast_node:
                            for child in ast_node['children']:
                                map_types_in_ast(child, converter)
                    return ast_node
                
                # Map all node types
                reconstructed_ast_json = [map_types_in_ast(node, converter) for node in generated_ast]
                reconstructed_code = run_ruby_script(pretty_print_script, json.dumps(reconstructed_ast_json))
                
                # Create reconstructed_data for isomorphism check
                # Flatten generated AST to get node features and edges
                def flatten_ast(ast_nodes):
                    nodes = []
                    edges = []
                    node_id = [0]
                    
                    def traverse(node, parent_id=None):
                        current_id = node_id[0]
                        node_id[0] += 1
                        
                        # Get node type index
                        node_type = node.get('type', 'unknown')
                        try:
                            type_idx = converter.node_encoder.node_types.index(node_type)
                        except ValueError:
                            type_idx = 0  # Unknown
                        
                        nodes.append(type_idx)
                        
                        if parent_id is not None:
                            edges.append([parent_id, current_id])
                        
                        for child in node.get('children', []):
                            traverse(child, current_id)
                    
                    for root in ast_nodes:
                        traverse(root)
                    
                    return nodes, edges
                
                node_types, edge_list = flatten_ast(reconstructed_ast_json)
                if node_types:
                    recon_node_features = torch.nn.functional.one_hot(
                        torch.tensor(node_types, dtype=torch.long),
                        num_classes=args.node_feature_dim
                    ).float()
                    recon_edge_index = torch.tensor(edge_list, dtype=torch.long).t() if edge_list else torch.empty((2, 0), dtype=torch.long)
                    reconstructed_data = Data(
                        x=recon_node_features,
                        edge_index=recon_edge_index
                    ).to(device)
                else:
                    reconstructed_data = Data(x=torch.empty(0), edge_index=torch.empty((2, 0), dtype=torch.long))
                
            except Exception as e:
                print(f"Error during generation: {e}")
                reconstructed_code = ""
                reconstructed_ast_json = []
                reconstructed_data = Data(x=torch.empty(0), edge_index=torch.empty((2, 0), dtype=torch.long))


            # --- Calculate Metrics ---
            is_valid_syntax = check_syntax(reconstructed_code, check_syntax_script)
            is_isomorphic = check_isomorphism(original_data, reconstructed_data)
            
            normalized_original = run_ruby_script(normalize_code_script, original_code)
            normalized_reconstructed = run_ruby_script(normalize_code_script, reconstructed_code)
            
            standard_bleu = calculate_bleu(original_code, reconstructed_code)
            normalized_bleu = calculate_bleu(normalized_original, normalized_reconstructed)
            
            results.append({
                "original_code": original_code,
                "reconstructed_code": reconstructed_code,
                "reconstructed_ast_json": reconstructed_ast_json,
                "syntactic_validity": is_valid_syntax,
                "ast_isomorphism": is_isomorphic,
                "standard_bleu": standard_bleu,
                "normalized_bleu": normalized_bleu
            })

    results_df = pd.DataFrame(results)

    # --- Quantitative Results ---
    summary = {
        "Metric": [
            "Syntactic Validity (%)",
            "AST Isomorphism (%)",
            "Avg. Normalized BLEU Score",
            "Avg. Standard BLEU Score"
        ],
        "Score": [
            results_df["syntactic_validity"].mean() * 100,
            results_df["ast_isomorphism"].mean() * 100,
            results_df["normalized_bleu"].mean(),
            results_df["standard_bleu"].mean()
        ]
    }
    summary_df = pd.DataFrame(summary)
    summary_df["Score"] = summary_df["Score"].round(4)

    print("\n--- Evaluation Summary ---")
    print(f"Model: {args.model_dir}")
    print(f"Samples Evaluated: {len(results_df)}")
    print("--------------------------")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Hierarchical AST Decoder model.")
    parser.add_argument(
        '--model_dir',
        type=str,
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing the hierarchical model files (default: {DEFAULT_MODEL_DIR})"
    )
    parser.add_argument(
        '--dataset_path',
        type=str,
        default=DEFAULT_TEST_DATASET_PATH,
        help=f"Path to the test dataset .jsonl file (default: {DEFAULT_TEST_DATASET_PATH})"
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of samples to evaluate from the test set (default: {DEFAULT_NUM_SAMPLES})"
    )
    parser.add_argument(
        '--alignment_model_path',
        type=str,
        default=DEFAULT_ALIGNMENT_MODEL_PATH,
        help=f"Path to the pre-trained alignment model (default: {DEFAULT_ALIGNMENT_MODEL_PATH})"
    )
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for GNNs.")
    parser.add_argument("--node_feature_dim", type=int, default=74, help="Dimension of node features.")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum AST depth to check for.")
    parser.add_argument(
        '--save_graphs',
        action='store_true',
        help="Save the reconstructed AST graphs to a JSONL file."
    )
    main(parser.parse_args())
