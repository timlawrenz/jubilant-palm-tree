#!/usr/bin/env python3
"""
Model Evaluation Script for Ruby Complexity Prediction

This script loads the best saved model from Phase 2 (best_model.pt) and the test dataset
(dataset/test.jsonl). It predicts the complexity for each method in the test set and
calculates the final performance metrics (MAE and RMSE).

The results are compared against the heuristic benchmark from Ticket 9.
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import math

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from data_processing import RubyASTDataset, SimpleDataLoader
from models import RubyComplexityGNN


def load_best_model(model_path: str, device: torch.device) -> RubyComplexityGNN:
    """
    Load the best saved model from checkpoint.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Loaded GNN model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model configuration
    config = checkpoint['model_config']
    
    # Initialize model with saved configuration
    model = RubyComplexityGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        conv_type=config['conv_type'],
        dropout=config['dropout']
    ).to(device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Set to evaluation mode
    model.eval()
    
    print(f"✅ Loaded model: {model.get_model_info()}")
    print(f"   Trained for {checkpoint['epoch']} epochs")
    print(f"   Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return model


def evaluate_model(model: RubyComplexityGNN, test_loader: SimpleDataLoader, device: torch.device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model: The trained GNN model
        test_loader: DataLoader for test dataset
        device: Device to run evaluation on
        
    Returns:
        Tuple of (predictions, true_values, mae, rmse)
    """
    model.eval()
    all_predictions = []
    all_true_values = []
    
    print("🔍 Running model evaluation on test set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Convert to PyTorch tensors and move to device
            x = torch.tensor(batch['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
            y = torch.tensor(batch['y'], dtype=torch.float).to(device)
            batch_tensor = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_tensor)
            
            # Forward pass
            predictions = model(data)
            
            # Store predictions and true values
            all_predictions.extend(predictions.squeeze().cpu().numpy())
            all_true_values.extend(y.cpu().numpy())
            
            if batch_idx == 0:
                print(f"   Batch {batch_idx + 1}: {len(y)} samples processed")
    
    # Convert to tensors for metric calculation
    pred_tensor = torch.tensor(all_predictions)
    true_tensor = torch.tensor(all_true_values)
    
    # Calculate metrics
    mae = torch.mean(torch.abs(pred_tensor - true_tensor)).item()
    rmse = torch.sqrt(torch.mean((pred_tensor - true_tensor) ** 2)).item()
    
    return all_predictions, all_true_values, mae, rmse


def print_evaluation_results(predictions, true_values, mae, rmse, heuristic_mae=4.4617):
    """
    Print detailed evaluation results and comparison to heuristic benchmark.
    
    Args:
        predictions: List of predicted complexity values
        true_values: List of true complexity values
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        heuristic_mae: Heuristic benchmark MAE for comparison
    """
    print("\n" + "=" * 60)
    print("📊 MODEL EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"📈 Performance Metrics:")
    print(f"   Mean Absolute Error (MAE):  {mae:.4f}")
    print(f"   Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"   Test samples evaluated: {len(predictions)}")
    
    print(f"\n📋 Data Range:")
    print(f"   True complexity range: {min(true_values):.2f} - {max(true_values):.2f}")
    print(f"   Predicted range: {min(predictions):.2f} - {max(predictions):.2f}")
    
    print(f"\n🎯 Benchmark Comparison:")
    print(f"   Heuristic Benchmark MAE: {heuristic_mae:.4f}")
    print(f"   GNN Model MAE:          {mae:.4f}")
    
    if mae < heuristic_mae:
        improvement = ((heuristic_mae - mae) / heuristic_mae) * 100
        print(f"   🎉 SUCCESS! Model beats heuristic by {improvement:.1f}%")
    else:
        difference = ((mae - heuristic_mae) / heuristic_mae) * 100
        print(f"   ⚠️  Model underperforms heuristic by {difference:.1f}%")
    
    print(f"\n📝 Sample Predictions (first 5):")
    for i in range(min(5, len(predictions))):
        error = abs(predictions[i] - true_values[i])
        print(f"   Sample {i+1}: True={true_values[i]:.2f}, Pred={predictions[i]:.2f}, Error={error:.2f}")
    
    print("=" * 60)


def main():
    """Main evaluation function."""
    print("🔬 Ruby Complexity GNN Model Evaluation")
    print("=" * 60)
    
    # Configuration
    model_path = "best_model.pt"
    test_dataset_path = "dataset/test.jsonl"
    batch_size = 32
    heuristic_baseline_mae = 4.4617  # From Ticket 9 heuristic benchmark
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    try:
        # Load test dataset
        print(f"📂 Loading test dataset: {test_dataset_path}")
        test_dataset = RubyASTDataset(test_dataset_path)
        test_loader = SimpleDataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False  # Keep deterministic order for evaluation
        )
        print(f"   Test dataset size: {len(test_dataset)} samples")
        print(f"   Test batches: {len(test_loader)}")
        
        # Load best model
        print(f"\n🧠 Loading best model: {model_path}")
        model = load_best_model(model_path, device)
        
        # Run evaluation
        print(f"\n⚡ Running evaluation...")
        predictions, true_values, mae, rmse = evaluate_model(model, test_loader, device)
        
        # Print results
        print_evaluation_results(predictions, true_values, mae, rmse, heuristic_baseline_mae)
        
        print(f"\n✅ Evaluation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("   Make sure to run training first to generate best_model.pt")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()