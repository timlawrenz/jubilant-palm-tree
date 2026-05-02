import argparse
import json
import torch
import os
import sys
import subprocess
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.models import AlignmentModel
from src.data_processing import PairedDataset
from torch_geometric.data import Data

def find_similar_code(prompt, alignment_model, dataset, device):
    """Finds the most similar code snippet in the dataset to the given prompt."""
    
    # Encode the prompt
    prompt_embedding = alignment_model.encode_text([prompt]).detach().cpu().numpy()
    
    best_similarity = -1
    best_sample = None
    
    for i in range(len(dataset)):
        graph_data, text_description = dataset[i]
        
        # Create a Data object for the code encoder
        data = Data(
            x=torch.tensor(graph_data['x'], dtype=torch.float).to(device),
            edge_index=torch.tensor(graph_data['edge_index'], dtype=torch.long).to(device),
            batch=torch.zeros(graph_data['num_nodes'], dtype=torch.long).to(device)
        )
        
        # Encode the code snippet
        code_embedding = alignment_model.encode_code(data).detach().cpu().numpy()
        
        # Calculate similarity
        similarity = cosine_similarity(prompt_embedding, code_embedding)[0][0]
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_sample = dataset.data[i]
            
    return best_sample, best_similarity

def main():
    parser = argparse.ArgumentParser(description="Find the most similar code snippet in the dataset to a given prompt.")
    parser.add_argument("prompt", type=str, help="The text prompt to search for.")
    parser.add_argument("--alignment_model_path", type=str, default="models/best_alignment_model.pt", help="Path to the trained alignment model.")
    parser.add_argument("--dataset_path", type=str, default="dataset/paired_data.jsonl", help="Path to the paired dataset.")
    parser.add_argument("--embedding_dim", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--node_feature_dim", type=int, default=74, help="Dimension of node features.")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load alignment model
    alignment_model = AlignmentModel(
        input_dim=args.node_feature_dim,
        hidden_dim=args.embedding_dim
    ).to(device)
    
    try:
        checkpoint = torch.load(args.alignment_model_path, map_location=device)
        alignment_model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
        alignment_model.eval()
        print("✅ Alignment model loaded successfully.")
    except FileNotFoundError:
        print(f"🚨 ERROR: Alignment model not found at {args.alignment_model_path}. Cannot proceed.")
        sys.exit(1)

    # Load dataset
    dataset = PairedDataset(args.dataset_path)
    
    # Find similar code
    best_sample, best_similarity = find_similar_code(args.prompt, alignment_model, dataset, device)
    
    if best_sample:
        print(f"\nFound best match with similarity: {best_similarity:.4f}")
        print("\nOriginal Descriptions:")
        for desc in best_sample.get('descriptions', []):
            print(f"- {desc['text']}")
        print("\nGenerated Code:")
        
        # Convert AST to Ruby code
        ast_json = best_sample['ast_json']
        script_path = os.path.join(os.path.dirname(__file__), 'scripts', 'pretty_print_ast.rb')
        ruby_executable = os.environ.get('RUBY_EXECUTABLE', 'ruby')
        
        try:
            result = subprocess.run(
                [ruby_executable, script_path],
                input=ast_json,
                text=True,
                capture_output=True,
                check=True
            )
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error running pretty_print_ast.rb: {e.stderr}")
    else:
        print("Could not find a similar code snippet.")

if __name__ == "__main__":
    main()
