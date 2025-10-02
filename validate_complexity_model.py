import argparse
import json
import os
import sys
import torch
import pandas as pd
from torch_geometric.data import Data
from tqdm import tqdm

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models import RubyComplexityGNN
from src.data_processing import create_data_loaders

def load_model(model_path, device):
    """Loads a pre-trained model."""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('model_config', {})
    
    model = RubyComplexityGNN(
        input_dim=config.get('input_dim', 74),
        hidden_dim=config.get('hidden_dim', 64),
        num_layers=config.get('num_layers', 3),
        conv_type=config.get('conv_type', 'SAGE'),
        dropout=config.get('dropout', 0.1)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {model_path}")
    return model

def evaluate_model(model, loader, device):
    """Evaluates the model on a given data loader."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating model"):
            x = torch.tensor(batch['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
            y = torch.tensor(batch['y'], dtype=torch.float).to(device)
            batch_idx = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            data = Data(x=x, edge_index=edge_index, batch=batch_idx)
            
            pred = model(data)
            predictions.extend(pred.squeeze().cpu().numpy())
            actuals.extend(y.cpu().numpy())
            
    return pd.DataFrame({'predictions': predictions, 'actuals': actuals})

def calculate_mae(df):
    """Calculates Mean Absolute Error."""
    return (df['predictions'] - df['actuals']).abs().mean()

def load_heuristic_benchmark(dataset_path):
    """Loads heuristic predictions from the dataset file."""
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}", file=sys.stderr)
        sys.exit(1)
        
    actuals = []
    heuristics = []
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            actuals.append(data['complexity_score'])
            # Assuming a simple heuristic: number of nodes in the AST
            ast = json.loads(data['ast_json'])
            num_nodes = len(ast.get('children', [])) + 1 if ast else 0
            heuristics.append(num_nodes) # This is a placeholder heuristic
            
    return pd.DataFrame({'predictions': heuristics, 'actuals': actuals})


def main():
    parser = argparse.ArgumentParser(description="Evaluate Ruby Complexity GNN Model")
    parser.add_argument('--model_path', type=str, default='models/best_model.pt', help='Path to the trained model')
    parser.add_argument('--dataset_path', type=str, default='dataset/test.jsonl', help='Path to the test dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Model
    model = load_model(args.model_path, device)

    # Create Data Loader
    _, test_loader = create_data_loaders(
        args.dataset_path, 
        args.dataset_path,
        batch_size=args.batch_size
    )



    # Evaluate GNN Model
    results_df = evaluate_model(model, test_loader, device)
    gnn_mae = calculate_mae(results_df)
    print(f"\nGNN Model MAE: {gnn_mae:.4f}")

    # Evaluate Heuristic Benchmark
    # The README claims a specific benchmark was implemented. I'll need to find it.
    # For now, I'll implement a simple node count as a placeholder.
    # The actual benchmark logic is likely in a ruby script.
    # Let's search for it.
    print("Note: The heuristic benchmark MAE is a placeholder based on node count.")
    print("The claimed benchmark implementation needs to be located and used for a fair comparison.")
    
    # The original benchmark is likely more sophisticated.
    # I will look for a file like `benchmark_heuristic.rb`
    benchmark_df = load_heuristic_benchmark(args.dataset_path)
    benchmark_mae = calculate_mae(benchmark_df)
    print(f"Simple Heuristic Benchmark MAE (Node Count): {benchmark_mae:.4f}")
    
    print("\n--- Comparison ---")
    print(f"GNN Model MAE: {gnn_mae:.4f}")
    print(f"Simple Heuristic MAE: {benchmark_mae:.4f}")
    claimed_mae = 4.27
    claimed_benchmark_mae = 4.46
    print(f"Claimed GNN MAE: {claimed_mae}")
    print(f"Claimed Benchmark MAE: {claimed_benchmark_mae}")


if __name__ == '__main__':
    main()
