import argparse
import json
import torch
import os
import sys
import subprocess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pathlib import Path
from src.models import HierarchicalASTDecoder, AlignmentModel
from src.data_processing import ASTNodeEncoder

def convert_ast_to_ruby(ast_dict):
    """Converts an AST dictionary to Ruby code by calling the pretty_print_ast.rb script."""
    script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'pretty_print_ast.rb')
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Pretty printer script not found at {script_path}")

    # Use the specific Ruby executable from the project's environment if available
    ruby_executable = os.environ.get('RUBY_EXECUTABLE', 'ruby')

    try:
        ast_json = json.dumps(ast_dict)
        result = subprocess.run(
            [ruby_executable, script_path],
            input=ast_json,
            text=True,
            capture_output=True,
            check=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        error_message = f"Error running pretty_print_ast.rb: {e}\n"
        error_message += f"Stderr: {e.stderr}"
        raise RuntimeError(error_message)
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred while converting AST to Ruby: {e}")

def load_models(args, device):
    """Loads the text encoder and hierarchical decoder models."""
    
    # 1. Load Text Encoder from the Alignment Model
    print(f"Loading alignment model from {args.alignment_model_path} to get text encoder...")
    alignment_model = AlignmentModel(
        input_dim=args.node_feature_dim,
        hidden_dim=args.embedding_dim
    ).to(device)
    
    try:
        checkpoint = torch.load(args.alignment_model_path, map_location=device)
        alignment_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        alignment_model.eval()
        print("✅ Text encoder loaded successfully.")
    except FileNotFoundError:
        print(f"🚨 ERROR: Alignment model not found at {args.alignment_model_path}. Cannot proceed.")
        sys.exit(1)
    except Exception as e:
        print(f"🚨 ERROR: Failed to load alignment model: {e}")
        sys.exit(1)

    # 2. Load Hierarchical Decoder
    print(f"Loading hierarchical decoder models from {args.model_dir}...")
    decoder = HierarchicalASTDecoder(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_levels=args.max_depth,
        node_feature_dim=args.node_feature_dim,
        conv_type='GCN'
    ).to(device)

    loaded_levels = 0
    for i in range(args.max_depth):
        level_model_path = Path(args.model_dir) / f"hierarchical_decoder_level_{i}.pt"
        if level_model_path.exists():
            try:
                state_dict = torch.load(level_model_path, map_location=device)
                decoder.level_generators[i].load_state_dict(state_dict)
                loaded_levels += 1
            except Exception as e:
                print(f"⚠️ WARNING: Could not load model for level {i}: {e}")
        else:
            print(f"No model found for level {i}. Stopping search.")
            break
            
    if loaded_levels == 0:
        print(f"🚨 ERROR: No hierarchical models found in {args.model_dir}. Cannot proceed.")
        sys.exit(1)
        
    decoder.eval()
    print(f"✅ Hierarchical decoder loaded with {loaded_levels} levels.")
    
    return alignment_model, decoder

def generate_ast(prompt, alignment_model, decoder, args, device):
    """Generates an AST from a text prompt using an iterative, hierarchical process."""
    
    print(f"\nGenerating AST for prompt: '{prompt}'")

    node_encoder = ASTNodeEncoder()
    # Create a reverse mapping from index to node type
    idx_to_type = {i: t for t, i in node_encoder.type_to_idx.items()}
    idx_to_type[node_encoder.unknown_idx] = 'unknown'


    with torch.no_grad():
        # Start with the text embedding as the input for the first level.
        level_input = alignment_model.encode_text([prompt]).to(device)

    print("🧠 Performing hierarchical generation...")
    
    all_nodes = []
    all_edges = []
    node_offset = 0

    # Iteratively generate each level of the AST
    for level in range(decoder.num_levels):
        print(f"   - Generating Level {level}...")
        try:
            # First, run the decoder with a single node to get an initial prediction
            # for the adjacency matrix, which we'll use to decide how many nodes to generate.
            initial_output = decoder(level_input.repeat(1, 1), target_level=level)

            # For this simplified generation, we'll determine the number of nodes
            # to generate at this level based on the density of the predicted adjacency matrix.
            # This is a heuristic to allow for dynamic, multi-node generation per level.
            num_nodes_at_this_level = max(1, int(torch.sigmoid(initial_output['pred_adjacency']).sum().item() / 2))

            # The input for the generator needs to be expanded to match the number of nodes we'll create.
            expanded_input = level_input.repeat(num_nodes_at_this_level, 1)

            # Re-run the decoder for this level with the correct number of nodes.
            output = decoder(expanded_input, target_level=level)
            
            # The next level's input is the *mean* of the hidden states from this level's nodes.
            level_input = output['hidden_state'].mean(dim=0, keepdim=True)

            # --- Process the output of the current level ---
            # 1. Node Features: Decide the type of node(s) generated at this level.
            # Using argmax to get the most likely node type from the feature vector.
            pred_features = output['pred_features']
            node_types_indices = torch.argmax(pred_features, dim=1)
            
            # 2. Adjacency Matrix: Decide connections
            pred_adjacency = output['pred_adjacency']
            # Use a threshold to determine which connections exist.
            connections = (torch.sigmoid(pred_adjacency) > 0.5).nonzero().tolist()

            # Store the generated nodes and edges
            for idx in node_types_indices:
                node_type_str = idx_to_type.get(idx.item(), 'unknown')
                all_nodes.append({'id': node_offset, 'type': node_type_str, 'children': []})
                node_offset += 1
            
            for src, dest in connections:
                if src < node_offset and dest < node_offset:
                    all_edges.append((src, dest))

        except Exception as e:
            print(f"⚠️ WARNING: Could not generate level {level}: {e}")
            break

    print(f"✅ Generation loop finished. Total nodes: {len(all_nodes)}, Total edges: {len(all_edges)}")

    # --- Reconstruct the AST from the flat list of nodes and edges ---
    if not all_nodes:
        return {"type": "empty", "children": []}

    node_map = {node['id']: node for node in all_nodes}
    
    # Build the tree structure
    child_ids = set()
    for src_id, dest_id in all_edges:
        if src_id in node_map and dest_id in node_map:
            # Prevent a node from being its own parent
            if src_id == dest_id:
                continue
            
            # Add child to parent's children list
            parent_node = node_map[src_id]
            child_node = node_map[dest_id]
            parent_node['children'].append(child_node)
            
            # Keep track of which nodes are children
            child_ids.add(dest_id)

    # Find the root of the tree (a node that is not a child of any other node)
    root_nodes = [node for i, node in node_map.items() if i not in child_ids]

    # For simplicity, we'll return the first root found.
    # A more complex implementation might handle multiple roots (a forest).
    if root_nodes:
        generated_ast = root_nodes[0]
    else:
        # Fallback if no clear root is found (e.g., a circular graph)
        generated_ast = all_nodes[0] if all_nodes else {"type": "empty", "children": []}
    
    # Clean up the temporary 'id' field from all nodes before returning
    for node in all_nodes:
        if 'id' in node:
            del node['id']

    return generated_ast

def main():
    parser = argparse.ArgumentParser(description="Generate Ruby code from a text prompt using a hierarchical AST decoder.")
    parser.add_argument("prompt", type=str, nargs="?", default=None, help="The text prompt to generate code from. If not provided, runs in interactive mode.")
    parser.add_argument("--model_dir", type=str, default="models/hierarchical", help="Directory where the hierarchical models are saved.")
    parser.add_argument("--alignment_model_path", type=str, default="models/best_alignment_model.pt", help="Path to the trained text encoder model.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension for GNNs.")
    parser.add_argument("--node_feature_dim", type=int, default=74, help="Dimension of node features.")
    parser.add_argument("--max_depth", type=int, default=20, help="Maximum AST depth for generation.")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode.")
    args = parser.parse_args()

    if not args.prompt and not args.interactive:
        parser.error("You must provide a prompt or use --interactive mode.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    alignment_model, decoder = load_models(args, device)

    def run_generation(prompt):
        generated_ast = generate_ast(prompt, alignment_model, decoder, args, device)
        print("\nGenerated AST:")
        print(json.dumps(generated_ast, indent=2))

        try:
            ruby_code = convert_ast_to_ruby(generated_ast)
            print("\n✅ Generated Ruby Code:")
            print(ruby_code)
        except Exception as e:
            print(f"\n🚨 ERROR: Failed to convert AST to Ruby code: {e}")

    if args.interactive:
        print("\n--- Interactive Mode ---")
        print("Enter a prompt to generate code, or 'exit' to quit.")
        while True:
            try:
                prompt = input("> ")
                if prompt.lower() == 'exit':
                    break
                run_generation(prompt)
            except KeyboardInterrupt:
                print("\nExiting.")
                break
    else:
        run_generation(args.prompt)

if __name__ == "__main__":
    main()
