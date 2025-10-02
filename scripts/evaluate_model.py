#!/usr/bin/env python3
"""
Consolidated Evaluation Script for AST Autoencoder

This script provides a standardized and rigorous evaluation of the AST
Autoencoder's performance. It serves as the "final exam" for the model,
assessing its ability to reconstruct Ruby method ASTs with semantic and
structural correctness.
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

from src.models import ASTAutoencoder
from src.data_processing import ASTGraphConverter

# --- Configuration ---
# These can be overridden by command-line arguments
DEFAULT_MODEL_CHECKPOINT = "models/best_decoder.pt"
DEFAULT_TEST_DATASET_PATH = "dataset/test.jsonl"
DEFAULT_NUM_SAMPLES = 100

# --- Helper Functions ---
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

def load_model(checkpoint_path, device, decoder_conv_type):
    """Loads the autoencoder model from a checkpoint."""
    print(f"🧠 Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("model_config", {})
    model = ASTAutoencoder(
        encoder_input_dim=74, 
        node_output_dim=74,
        hidden_dim=config.get("hidden_dim", 64),
        num_layers=config.get("num_layers", 3),
        conv_type=config.get("conv_type", "SAGE"),
        dropout=config.get("dropout", 0.1),
        freeze_encoder=True,
        encoder_weights_path=None,
        max_nodes=config.get("max_nodes", 100),
        decoder_conv_type=decoder_conv_type
    ).to(device)
    model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
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
    model = load_model(args.model_path, device, args.decoder_conv_type)
    
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

            reconstruction_output = model(original_data)
            reconstruction_result = reconstruction_output["reconstruction"]
            
            # --- Build Reconstructed Graph ---
            recon_node_features = reconstruction_result["node_features"].squeeze(0)
            recon_node_types_indices = recon_node_features.argmax(dim=1)
            
            recon_parent_logits = reconstruction_result["parent_logits"].squeeze(0)
            recon_parents = recon_parent_logits.argmax(dim=1)

            # Create edge_index from parent predictions for isomorphism check
            child_indices = torch.arange(len(recon_parents), device=device)
            # Filter out self-loops for the edge_index
            valid_edges = recon_parents != child_indices
            recon_edge_index_tensor = torch.stack([recon_parents[valid_edges], child_indices[valid_edges]], dim=0)

            reconstructed_data = Data(
                x=torch.nn.functional.one_hot(recon_node_types_indices, num_classes=74).float(),
                edge_index=recon_edge_index_tensor.to(device)
            ).to(device)

            # --- Convert to Code for BLEU and Syntax Check ---
            # Build a tree structure from the parent predictions for the pretty-printer
            nodes_for_print = []
            node_map = {}
            for i, type_idx in enumerate(recon_node_types_indices.cpu().numpy()):
                node_type_str = converter.node_encoder.node_types[type_idx] if type_idx < len(converter.node_encoder.node_types) else "unknown"
                node_map[i] = {"type": node_type_str, "children": []}

            root_nodes = []
            for i, parent_idx in enumerate(recon_parents.cpu().numpy()):
                if i == parent_idx or parent_idx >= len(node_map): # Root node or invalid parent
                    root_nodes.append(node_map[i])
                else:
                    if parent_idx in node_map:
                        node_map[parent_idx]["children"].append(node_map[i])

            reconstructed_ast_json = root_nodes
            reconstructed_code = run_ruby_script(pretty_print_script, json.dumps(reconstructed_ast_json))
            
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
    print(f"Model: {args.model_path}")
    print(f"Samples Evaluated: {len(results_df)}")
    print("--------------------------")
    print(summary_df.to_string(index=False))

    # --- Qualitative Results ---
    if not args.no_qualitative:
        pd.set_option("display.max_colwidth", 25)
        pd.set_option("display.width", 120)
        comparison_df = results_df[["original_code", "reconstructed_code", "syntactic_validity", "ast_isomorphism", "normalized_bleu"]].copy()
        print("\n--- Sample Reconstructions (Top 5) ---")
        print(comparison_df.head(5))

    # --- Save Reconstructed Graphs ---
    if args.save_graphs:
        import time
        timestamp = int(time.time())
        output_filename = f"output/reconstructed_graphs_{timestamp}.jsonl"
        with open(output_filename, "w") as f:
            for result in results:
                # Add the reconstructed graph structure to the result dict
                # The `reconstructed_code` is generated from this structure
                f.write(json.dumps({"original_code": result["original_code"], "reconstructed_ast": result["reconstructed_ast_json"]}) + "\n")
        print(f"\n💾 Reconstructed graphs saved to {output_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained AST Autoencoder model.")
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_CHECKPOINT,
        help=f"Path to the model checkpoint file (default: {DEFAULT_MODEL_CHECKPOINT})"
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
        '--decoder_conv_type',
        type=str,
        default='SAGE',
        help="The GNN layer type for the decoder (e.g., 'SAGE', 'GCN', 'GraphConv')."
    )
    parser.add_argument(
        '--no_qualitative',
        action='store_true',
        help="Suppress the qualitative (side-by-side) output."
    )
    parser.add_argument(
        '--save_graphs',
        action='store_true',
        help="Save the reconstructed AST graphs to a JSONL file."
    )
    main(parser.parse_args())
