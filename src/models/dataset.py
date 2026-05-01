import json
import torch
from torch.utils.data import Dataset, DataLoader
import random

class ExecutionGraphDataset(Dataset):
    def __init__(self, jsonl_path: str, max_nodes: int = 128, augment_permutation: bool = True):
        self.max_nodes = max_nodes
        self.augment_permutation = augment_permutation
        self.graphs = []
        
        # Load all graphs into memory (13MB is trivial)
        with open(jsonl_path, 'r') as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                
                # Filter out graphs that exceed our node limit
                if len(data["compressed_graph"]["nodes"]) <= self.max_nodes:
                    self.graphs.append(data["compressed_graph"])

        print(f"Loaded {len(self.graphs)} valid graphs (max_nodes <= {max_nodes}).")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        nodes = graph["nodes"]
        edges = graph["edges"]
        
        num_nodes = len(nodes)
        
        # 1. Node Permutation Augmentation
        # We create a mapping from original node_id to a new shuffled index
        original_indices = [n["node_id"] for n in nodes]
        shuffled_indices = list(range(num_nodes))
        
        if self.augment_permutation:
            random.shuffle(shuffled_indices)
            
        # Map old ID -> new shuffled position
        id_to_idx = {old_id: new_idx for old_id, new_idx in zip(original_indices, shuffled_indices)}
        
        # 2. Initialize Tensors
        # [3, N, N] - Channel 0: Presence, Channel 1: EdgeType, Channel 2: InputIndex
        adjacency = torch.zeros((3, self.max_nodes, self.max_nodes), dtype=torch.float32)
        
        # Padding mask: True means real node intersection, False means padding void
        padding_mask = torch.zeros((self.max_nodes, self.max_nodes), dtype=torch.bool)
        
        # We also need to encode the Motif Types.
        # We can attach this as a 1D tensor [MAX_NODES] to feed into the DiT conditioning,
        # or we could embed it into the diagonal of the adjacency matrix. 
        # For now, let's keep it separate: 0 is padding, 1-6 are the Motifs.
        motif_map = {"Boundary": 1, "Sequence": 2, "Condition": 3, "Loop": 4, "State": 5, "Message": 6}
        motifs = torch.zeros((self.max_nodes,), dtype=torch.long)
        
        # Populate Motifs
        for n in nodes:
            new_idx = id_to_idx[n["node_id"]]
            motifs[new_idx] = motif_map.get(n["motif"], 0)
            
        # 3. Populate Edges
        for e in edges:
            source_idx = id_to_idx[e["source_node"]]
            target_idx = id_to_idx[e["target_node"]]
            
            # Channel 0: Presence = 1
            adjacency[0, source_idx, target_idx] = 1.0
            # Channel 1: Edge Type (0 for Exec, 1 for Data)
            adjacency[1, source_idx, target_idx] = float(e["edge_type"])
            # The original edges list contains inputs indices that can theoretically exceed num_index_classes
            # e.g., if a Method Call has 5 arguments, we get index 0,1,2,3,4. 
            # To prevent CUDA CrossEntropy bounds crashes, we clamp the index to our class limit (K-1).
            index_clamped = min(int(e["input_index"]), 3) # self.num_index_classes is 4
            
            # Channel 2: Input Index (Clamped for CE Loss)
            adjacency[2, source_idx, target_idx] = float(index_clamped)
            
        # 4. Build Padding Mask (NxN grid of valid intersections)
        padding_mask[:num_nodes, :num_nodes] = True
        
        return {
            "adjacency": adjacency,      # [3, 128, 128]
            "motifs": motifs,            # [128]
            "padding_mask": padding_mask # [128, 128]
        }

if __name__ == "__main__":
    dataset = ExecutionGraphDataset("dataset/compressed/compressed_motifs.jsonl", max_nodes=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    batch = next(iter(dataloader))
    print("Batch loaded successfully!")
    print(f"Adjacency Tensor Shape: {batch['adjacency'].shape}")
    print(f"Motifs Tensor Shape: {batch['motifs'].shape}")
    print(f"Padding Mask Shape: {batch['padding_mask'].shape}")
    
    # Verify the padding mask boolean count
    valid_cells = batch['padding_mask'][0].sum().item()
    print(f"Sample 0 has {valid_cells} valid N*N intersections ({(valid_cells**0.5):.0f} nodes).")
