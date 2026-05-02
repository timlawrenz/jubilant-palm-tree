import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from src.models.dataset import ExecutionGraphDataset

def render_matrix_as_image():
    # Load a single, real 128-node graph from our dataset
    dataset = ExecutionGraphDataset("dataset/compressed/compressed_motifs.jsonl", max_nodes=128, augment_permutation=False)
    
    # Grab the largest graph we can find to make it look interesting
    best_idx = 0
    max_nodes = 0
    for i in range(len(dataset)):
        # Calculate actual nodes (not padded)
        n = dataset[i]["padding_mask"].sum().item() ** 0.5
        if n > max_nodes:
            max_nodes = int(n)
            best_idx = i
            
    print(f"Rendering graph with {max_nodes} nodes.")
    sample = dataset[best_idx]
    
    # Extract the Adjacency Tensor [3, 128, 128]
    # Channel 0: Presence (1 or 0)
    # Channel 1: Type (0=Exec, 1=Data)
    # Channel 2: Input Index (0, 1, 2, 3)
    adj = sample["adjacency"].numpy()
    
    # We will map this to an RGB image
    # Red = Execution Edges (Presence=1, Type=0)
    # Green = Data Edges (Presence=1, Type=1)
    # Blue = Index intensity (brighter for higher argument indexes)
    
    rgb = np.zeros((128, 128, 3), dtype=np.float32)
    
    presence = adj[0]
    edge_type = adj[1]
    input_index = adj[2]
    
    # Red channel (Execution)
    rgb[:, :, 0] = presence * (1 - edge_type)
    
    # Green channel (Data)
    rgb[:, :, 1] = presence * edge_type
    
    # Blue channel (Input Index)
    # Scale index (0 to 3) to (0.25 to 1.0) for visibility, only where edges exist
    rgb[:, :, 2] = presence * ((input_index + 1) / 4.0)
    
    # Make the background dark blue/grey instead of harsh black for aesthetic
    background_mask = (presence == 0)
    rgb[background_mask, 0] = 0.05
    rgb[background_mask, 1] = 0.05
    rgb[background_mask, 2] = 0.1
    
    # Plot it!
    plt.figure(figsize=(10, 10))
    plt.imshow(rgb)
    plt.title(f"Neural Universal Machine Matrix\n{max_nodes}-Node Execution Graph")
    plt.axis('off')
    
    os.makedirs("ablation_plots", exist_ok=True)
    out_path = "ablation_plots/matrix_visualization.png"
    plt.savefig(out_path, bbox_inches='tight', dpi=300)
    print(f"Saved visualization to {out_path}")

if __name__ == "__main__":
    render_matrix_as_image()
