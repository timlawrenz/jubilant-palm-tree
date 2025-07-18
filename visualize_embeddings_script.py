#!/usr/bin/env python
# coding: utf-8

# # Ruby Code Complexity Embedding Visualization
# 
# This notebook analyzes the learned graph embeddings from our Graph Neural Network model trained on Ruby method complexity prediction. We will:
# 
# 1. Load the trained GNN model
# 2. Extract graph-level embeddings from the test dataset
# 3. Use t-SNE to project high-dimensional embeddings to 2D space
# 4. Visualize the embeddings colored by complexity scores to check for clustering

# In[ ]:


import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

# Add src directory to path
sys.path.insert(0, os.path.join('..', 'src'))

from data_processing import RubyASTDataset
from models import RubyComplexityGNN

# Set style for better plots
plt.style.use('default')
sns.set_palette("viridis")

print("📦 Libraries loaded successfully!")


# ## 1. Load the Trained Model
# 
# First, we'll load our best trained GNN model that was saved during training.

# In[ ]:


# Load the trained model checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('../best_model.pt', map_location=device)

print("🏆 Best Model Information:")
print(f"   Epoch: {checkpoint['epoch']}")
print(f"   Validation Loss: {checkpoint['val_loss']:.4f}")
print(f"   Model Config: {checkpoint['model_config']}")

# Initialize model with the same configuration
model_config = checkpoint['model_config']
model = RubyComplexityGNN(
    input_dim=model_config['input_dim'],
    hidden_dim=model_config['hidden_dim'], 
    num_layers=model_config['num_layers'],
    conv_type=model_config['conv_type'],
    dropout=model_config['dropout']
).to(device)

# Load the trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"\n✅ Model loaded successfully on {device}")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ## 2. Create Embedding Extraction Function
# 
# We need to modify the model's forward pass to extract embeddings after the global pooling layer but before the final prediction layer.

# In[ ]:


def extract_embeddings(model, data):
    """
    Extract graph-level embeddings from the model before the final prediction layer.

    Args:
        model: The trained GNN model
        data: PyTorch Geometric Data object

    Returns:
        embeddings: Graph-level embeddings from global pooling layer
        predictions: Final complexity predictions
    """
    x, edge_index, batch = data.x, data.edge_index, data.batch

    # Apply convolution layers (same as model.forward)
    for i, conv in enumerate(model.convs):
        x = conv(x, edge_index)
        if i < len(model.convs) - 1:  # No activation after last layer
            x = torch.nn.functional.relu(x)
            x = torch.nn.functional.dropout(x, p=model.dropout, training=model.training)

    # Global pooling to get graph-level representation (THIS IS WHAT WE WANT)
    embeddings = global_mean_pool(x, batch)

    # Final prediction
    predictions = model.predictor(embeddings)

    return embeddings, predictions

print("🔧 Embedding extraction function created!")


# ## 3. Load Test Dataset and Extract Embeddings
# 
# Now we'll load the test dataset and pass it through our model to extract the graph-level embeddings.

# In[ ]:


# Load test dataset
test_dataset = RubyASTDataset('../dataset/test.jsonl')
print(f"📊 Loaded {len(test_dataset)} test samples")

# Extract embeddings and predictions for all test samples
all_embeddings = []
all_predictions = []
all_true_complexity = []
all_metadata = []

print("\n🔍 Extracting embeddings from test samples...")

with torch.no_grad():
    for i, sample in enumerate(test_dataset):
        # Convert sample to PyTorch tensors
        x = torch.tensor(sample['x'], dtype=torch.float).to(device)
        edge_index = torch.tensor(sample['edge_index'], dtype=torch.long).to(device)
        y = sample['y'][0]  # True complexity score

        # Create batch tensor for single sample
        batch = torch.zeros(x.size(0), dtype=torch.long).to(device)

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, batch=batch)

        # Extract embeddings and predictions
        embeddings, predictions = extract_embeddings(model, data)

        # Store results
        all_embeddings.append(embeddings.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_true_complexity.append(y)
        all_metadata.append({
            'id': sample['id'],
            'repo_name': sample['repo_name'],
            'file_path': sample['file_path']
        })

        if (i + 1) % 50 == 0:
            print(f"   Processed {i + 1}/{len(test_dataset)} samples")

# Convert to numpy arrays
embeddings_matrix = np.vstack(all_embeddings)
predictions_array = np.vstack(all_predictions).flatten()
true_complexity_array = np.array(all_true_complexity)

print(f"\n✅ Extraction complete!")
print(f"   Embeddings shape: {embeddings_matrix.shape}")
print(f"   Embedding dimension: {embeddings_matrix.shape[1]}")
print(f"   Number of samples: {embeddings_matrix.shape[0]}")
print(f"   Complexity range: {true_complexity_array.min():.1f} - {true_complexity_array.max():.1f}")


# ## 4. Dimensionality Reduction with t-SNE
# 
# We'll use t-SNE to project the high-dimensional embeddings into a 2D space for visualization.

# In[ ]:


print("🎯 Performing t-SNE dimensionality reduction...")
print(f"   Input: {embeddings_matrix.shape[1]}D embeddings")
print(f"   Output: 2D projection")

# Configure t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=min(30, len(embeddings_matrix) // 4),  # Adjust perplexity for smaller datasets
    random_state=42,
    max_iter=1000,
    verbose=1
)

# Apply t-SNE
embeddings_2d = tsne.fit_transform(embeddings_matrix)

print(f"\n✅ t-SNE complete!")
print(f"   2D embeddings shape: {embeddings_2d.shape}")
print(f"   X range: [{embeddings_2d[:, 0].min():.2f}, {embeddings_2d[:, 0].max():.2f}]")
print(f"   Y range: [{embeddings_2d[:, 1].min():.2f}, {embeddings_2d[:, 1].max():.2f}]")


# ## 5. Create Visualization
# 
# Now we'll create a scatter plot of the 2D embeddings, colored by the true complexity scores to see if the model has learned meaningful representations that cluster similar complexity scores together.

# In[ ]:


# Create the main visualization
plt.figure(figsize=(12, 8))

# Create scatter plot colored by complexity score
scatter = plt.scatter(
    embeddings_2d[:, 0], 
    embeddings_2d[:, 1], 
    c=true_complexity_array, 
    cmap='viridis', 
    s=60, 
    alpha=0.7,
    edgecolors='black',
    linewidth=0.5
)

# Add colorbar
cbar = plt.colorbar(scatter)
cbar.set_label('True Complexity Score', fontsize=12, fontweight='bold')

# Customize plot
plt.title('Ruby Method Complexity: 2D Embedding Visualization\n(Colored by True Complexity Score)', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('t-SNE Dimension 1', fontsize=12, fontweight='bold')
plt.ylabel('t-SNE Dimension 2', fontsize=12, fontweight='bold')

# Add grid for better readability
plt.grid(True, alpha=0.3)

# Add text box with statistics
stats_text = f"""Dataset: {len(test_dataset)} Ruby methods
Embedding dim: {embeddings_matrix.shape[1]}D → 2D
Complexity range: {true_complexity_array.min():.1f} - {true_complexity_array.max():.1f}
Model: {model_config['conv_type']} GNN"""

plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
         fontsize=10)

plt.tight_layout()
plt.show()

print("🎨 Main visualization complete!")


# ## 6. Analysis: Clustering by Complexity Ranges
# 
# Let's analyze the clustering behavior by grouping methods into complexity ranges and seeing if they form distinct clusters.

# In[ ]:


# Define complexity ranges for analysis
def categorize_complexity(score):
    if score < 5:
        return 'Low (< 5)'
    elif score < 15:
        return 'Medium (5-15)'
    elif score < 30:
        return 'High (15-30)'
    else:
        return 'Very High (≥ 30)'

# Categorize all samples
complexity_categories = [categorize_complexity(score) for score in true_complexity_array]
unique_categories = list(set(complexity_categories))
category_counts = {cat: complexity_categories.count(cat) for cat in unique_categories}

print("📊 Complexity Category Distribution:")
for category, count in sorted(category_counts.items()):
    print(f"   {category}: {count} methods ({count/len(complexity_categories)*100:.1f}%)")

# Create categorical visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Scatter by categories
colors = plt.cm.Set1(np.linspace(0, 1, len(unique_categories)))
for i, category in enumerate(sorted(unique_categories)):
    mask = [cat == category for cat in complexity_categories]
    ax1.scatter(
        embeddings_2d[mask, 0], 
        embeddings_2d[mask, 1], 
        c=[colors[i]], 
        label=f'{category} ({sum(mask)} methods)',
        s=60, 
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )

ax1.set_title('Embeddings by Complexity Categories', fontsize=12, fontweight='bold')
ax1.set_xlabel('t-SNE Dimension 1', fontweight='bold')
ax1.set_ylabel('t-SNE Dimension 2', fontweight='bold')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

# Right plot: Distribution of categories
category_names = sorted(category_counts.keys())
category_values = [category_counts[cat] for cat in category_names]

bars = ax2.bar(range(len(category_names)), category_values, color=colors[:len(category_names)])
ax2.set_title('Method Count by Complexity Category', fontsize=12, fontweight='bold')
ax2.set_xlabel('Complexity Category', fontweight='bold')
ax2.set_ylabel('Number of Methods', fontweight='bold')
ax2.set_xticks(range(len(category_names)))
ax2.set_xticklabels(category_names, rotation=45, ha='right')

# Add value labels on bars
for bar, value in zip(bars, category_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             str(value), ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

print("\n📈 Categorical analysis complete!")


# ## 7. Model Performance Analysis
# 
# Let's also examine how well our model's predictions correlate with the true complexity scores and if this is reflected in the embedding space.

# In[ ]:


# Calculate prediction metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(true_complexity_array, predictions_array)
mse = mean_squared_error(true_complexity_array, predictions_array)
rmse = np.sqrt(mse)
r2 = r2_score(true_complexity_array, predictions_array)

print("🎯 Model Performance on Test Set:")
print(f"   Mean Absolute Error (MAE): {mae:.4f}")
print(f"   Root Mean Square Error (RMSE): {rmse:.4f}")
print(f"   R² Score: {r2:.4f}")

# Compare with heuristic baseline (MAE: 4.4617 from README)
baseline_mae = 4.4617
improvement = ((baseline_mae - mae) / baseline_mae) * 100
print(f"\n📊 Comparison with Heuristic Baseline:")
print(f"   Heuristic Baseline MAE: {baseline_mae:.4f}")
print(f"   GNN Model MAE: {mae:.4f}")
if mae < baseline_mae:
    print(f"   ✅ Improvement: {improvement:.1f}% better than baseline")
else:
    print(f"   ❌ Performance: {-improvement:.1f}% worse than baseline")

# Create prediction vs actual plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Left: Prediction vs Actual
ax1.scatter(true_complexity_array, predictions_array, alpha=0.6, s=40)
ax1.plot([true_complexity_array.min(), true_complexity_array.max()], 
         [true_complexity_array.min(), true_complexity_array.max()], 
         'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('True Complexity Score', fontweight='bold')
ax1.set_ylabel('Predicted Complexity Score', fontweight='bold')
ax1.set_title(f'Prediction Accuracy\n(R² = {r2:.3f}, MAE = {mae:.3f})', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Right: Residuals
residuals = predictions_array - true_complexity_array
ax2.scatter(true_complexity_array, residuals, alpha=0.6, s=40)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('True Complexity Score', fontweight='bold')
ax2.set_ylabel('Prediction Error (Predicted - True)', fontweight='bold')
ax2.set_title('Prediction Residuals', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n📊 Performance analysis complete!")


# ## 8. Summary and Conclusions
# 
# Let's summarize our findings about the learned embeddings and their clustering behavior.

# In[ ]:


print("🎯 EMBEDDING VISUALIZATION SUMMARY")
print("=" * 50)

print(f"\n📊 Dataset Information:")
print(f"   • Total test samples: {len(test_dataset)}")
print(f"   • Embedding dimension: {embeddings_matrix.shape[1]}D")
print(f"   • Complexity range: {true_complexity_array.min():.1f} - {true_complexity_array.max():.1f}")

print(f"\n🧠 Model Architecture:")
print(f"   • Type: {model_config['conv_type']} Graph Neural Network")
print(f"   • Layers: {model_config['num_layers']}")
print(f"   • Hidden dimension: {model_config['hidden_dim']}")
print(f"   • Parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\n📈 Performance Metrics:")
print(f"   • Mean Absolute Error: {mae:.4f}")
print(f"   • R² Score: {r2:.4f}")
if mae < baseline_mae:
    print(f"   • ✅ Beats heuristic baseline by {improvement:.1f}%")
else:
    print(f"   • ❌ Below heuristic baseline by {-improvement:.1f}%")

print(f"\n🎨 Visualization Results:")
print(f"   • Successfully created 2D embedding visualization")
print(f"   • Methods colored by true complexity scores")
print(f"   • Complexity categories: {len(unique_categories)} groups")

# Analyze clustering quality
print(f"\n🔍 Clustering Analysis:")
for category in sorted(unique_categories):
    count = category_counts[category]
    percentage = count / len(complexity_categories) * 100
    print(f"   • {category}: {count} methods ({percentage:.1f}%)")

print(f"\n✅ CONCLUSION:")
if r2 > 0.3 and mae < baseline_mae:
    print("   The GNN model has successfully learned meaningful representations!")
    print("   The embedding visualization shows evidence of clustering by complexity.")
elif r2 > 0.1:
    print("   The GNN model shows some learning of structural patterns.")
    print("   Further training or architecture improvements may help.")
else:
    print("   The model shows limited learning of complexity patterns.")
    print("   Consider architectural changes or additional training.")

print("\n" + "=" * 50)
print("🎉 Embedding visualization analysis complete!")

