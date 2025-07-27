#!/usr/bin/env python3
"""
Generate t-SNE visualization of learned embeddings for the final report.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import RubyASTDataset, SimpleDataLoader
from models import RubyComplexityGNN


def load_model(model_path: str, device: torch.device) -> RubyComplexityGNN:
    """Load the trained model."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['model_config']
    
    model = RubyComplexityGNN(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        conv_type=config['conv_type'],
        dropout=config['dropout']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


def extract_embeddings(model: RubyComplexityGNN, test_loader: SimpleDataLoader, device: torch.device):
    """Extract graph-level embeddings before the final prediction layer."""
    model.eval()
    all_embeddings = []
    all_complexities = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Convert to PyTorch tensors
            x = torch.tensor(batch['x'], dtype=torch.float).to(device)
            edge_index = torch.tensor(batch['edge_index'], dtype=torch.long).to(device)
            y = torch.tensor(batch['y'], dtype=torch.float).to(device)
            batch_tensor = torch.tensor(batch['batch'], dtype=torch.long).to(device)
            
            # Create PyTorch Geometric Data object
            data = Data(x=x, edge_index=edge_index, batch=batch_tensor)
            
            # Forward pass through GNN layers but stop before final prediction
            h = x
            for i, conv in enumerate(model.convs):
                h = conv(h, edge_index)
                if i < len(model.convs) - 1:  # No activation after last layer
                    h = torch.relu(h)
                    h = torch.nn.functional.dropout(h, p=model.dropout, training=model.training)
            
            # Apply global mean pooling to get graph-level embeddings
            graph_embeddings = global_mean_pool(h, batch_tensor)
            
            # Store embeddings and complexity scores
            all_embeddings.append(graph_embeddings.cpu().numpy())
            all_complexities.extend(y.cpu().numpy())
    
    # Concatenate all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    complexity_array = np.array(all_complexities)
    
    return embeddings_matrix, complexity_array


def create_tsne_visualization(embeddings, complexities, save_path='embedding_visualization.png'):
    """Create t-SNE visualization of embeddings colored by complexity."""
    print("🔄 Computing t-SNE projection...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create complexity categories for better visualization
    complexity_categories = np.digitize(complexities, bins=[0, 5, 10, 20, 100])
    category_labels = ['Very Low (≤5)', 'Low (5-10)', 'Medium (10-20)', 'High (>20)']
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Create a scatter plot with complexity as color
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=complexities, cmap='viridis', s=50, alpha=0.7)
    
    # Add colorbar
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Complexity Score', fontsize=12)
    
    # Add labels and title
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('t-SNE Visualization of Ruby Method Complexity Embeddings\n' + 
              'Graph Neural Network Learned Representations', fontsize=14, pad=20)
    
    # Add statistics text
    stats_text = f'Dataset: {len(complexities)} test samples\n'
    stats_text += f'Complexity range: {complexities.min():.1f} - {complexities.max():.1f}\n'
    stats_text += f'Mean complexity: {complexities.mean():.1f} ± {complexities.std():.1f}'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Visualization saved to: {save_path}")
    
    return embeddings_2d


def main():
    """Main function to generate the visualization."""
    print("🎨 Generating Ruby Complexity Embedding Visualization")
    print("=" * 60)
    
    # Configuration
    model_path = "models/best_model.pt"
    test_dataset_path = "dataset/test.jsonl"
    output_path = "final_embedding_visualization.png"
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    
    # Load test dataset
    print(f"📂 Loading test dataset: {test_dataset_path}")
    test_dataset = RubyASTDataset(test_dataset_path)
    test_loader = SimpleDataLoader(test_dataset, batch_size=32, shuffle=False)
    print(f"   Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"🧠 Loading model: {model_path}")
    model = load_model(model_path, device)
    print(f"   Model loaded successfully")
    
    # Extract embeddings
    print("🔍 Extracting graph embeddings...")
    embeddings, complexities = extract_embeddings(model, test_loader, device)
    print(f"   Extracted embeddings shape: {embeddings.shape}")
    print(f"   Complexity range: {complexities.min():.2f} - {complexities.max():.2f}")
    
    # Create visualization
    print("🎨 Creating t-SNE visualization...")
    embeddings_2d = create_tsne_visualization(embeddings, complexities, output_path)
    
    print("\n✅ Visualization generation completed!")
    print(f"📊 Final plot saved as: {output_path}")


if __name__ == "__main__":
    main()