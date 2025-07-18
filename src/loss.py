"""
Loss functions for AST reconstruction tasks.

This module provides loss functions for measuring the difference between 
original and reconstructed Abstract Syntax Trees in torch_geometric format.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Any, Union


def ast_reconstruction_loss(original: Data, reconstructed: Dict[str, Any], 
                          node_weight: float = 1.0, edge_weight: float = 0.5) -> torch.Tensor:
    """
    Compute the reconstruction loss between original and reconstructed AST.
    
    This loss function combines:
    1. Node Type Loss: Cross-entropy loss for predicting correct node types
    2. Edge Prediction Loss: Loss for predicting correct graph connectivity
    
    Args:
        original: Original AST as torch_geometric.data.Data object
        reconstructed: Reconstructed AST from decoder containing:
            - 'node_features': Tensor of shape [batch_size, num_nodes, feature_dim]
            - 'edge_index': Edge connectivity (optional, for edge loss)
            - 'batch': Batch indices
            - 'num_nodes_per_graph': List of node counts per graph
        node_weight: Weight for node type loss component
        edge_weight: Weight for edge prediction loss component
        
    Returns:
        Scalar tensor representing the total reconstruction loss
    """
    # Extract original data
    original_x = original.x  # [total_nodes, feature_dim]
    original_edge_index = original.edge_index  # [2, total_edges]
    original_batch = original.batch  # [total_nodes]
    
    # Extract reconstructed data
    recon_node_features = reconstructed['node_features']  # [batch_size, max_nodes, feature_dim]
    batch_size = recon_node_features.size(0)
    max_nodes = recon_node_features.size(1)
    feature_dim = recon_node_features.size(2)
    
    # Compute node type loss
    node_loss = compute_node_type_loss(original_x, recon_node_features, original_batch)
    
    # Compute edge prediction loss (simplified version)
    edge_loss = compute_edge_prediction_loss(original_edge_index, original_batch, 
                                           reconstructed, batch_size)
    
    # Combine losses
    total_loss = node_weight * node_loss + edge_weight * edge_loss
    
    return total_loss


def compute_node_type_loss(original_x: torch.Tensor, 
                          recon_node_features: torch.Tensor,
                          original_batch: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss for node type prediction.
    
    Args:
        original_x: Original node features [total_nodes, feature_dim] (one-hot encoded)
        recon_node_features: Reconstructed features [batch_size, max_nodes, feature_dim] (logits)
        original_batch: Batch indices for original nodes [total_nodes]
        
    Returns:
        Average cross-entropy loss across all nodes
    """
    batch_size = recon_node_features.size(0)
    max_nodes = recon_node_features.size(1)
    feature_dim = recon_node_features.size(2)
    
    total_loss = 0.0
    total_nodes = 0
    
    # Process each graph in the batch
    for batch_idx in range(batch_size):
        # Get original nodes for this graph
        mask = (original_batch == batch_idx)
        if not mask.any():
            continue
            
        original_nodes = original_x[mask]  # [num_nodes_in_graph, feature_dim]
        num_original_nodes = original_nodes.size(0)
        
        # Get reconstructed nodes for this graph (up to actual node count)
        # Handle case where reconstruction has fewer nodes than original
        num_recon_nodes = min(num_original_nodes, max_nodes)
        recon_nodes = recon_node_features[batch_idx, :num_recon_nodes, :]  # [num_recon_nodes, feature_dim]
        
        # Only use original nodes up to the number of reconstructed nodes
        original_nodes_subset = original_nodes[:num_recon_nodes, :]  # [num_recon_nodes, feature_dim]
        
        # Convert one-hot original to class indices for cross-entropy
        # Assumes original_x is one-hot encoded
        original_classes = torch.argmax(original_nodes_subset, dim=1)  # [num_recon_nodes]
        
        # Compute cross-entropy loss
        # recon_nodes are logits, original_classes are target class indices
        loss = F.cross_entropy(recon_nodes, original_classes, reduction='sum')
        
        total_loss += loss
        total_nodes += num_recon_nodes
    
    # Return average loss per node
    if total_nodes > 0:
        return total_loss / total_nodes
    else:
        return torch.tensor(0.0, device=original_x.device, requires_grad=True)


def compute_edge_prediction_loss(original_edge_index: torch.Tensor,
                                original_batch: torch.Tensor,
                                reconstructed: Dict[str, Any],
                                batch_size: int) -> torch.Tensor:
    """
    Compute edge prediction loss based on graph connectivity.
    
    This is a simplified version that compares the number of edges per graph
    rather than exact edge-to-edge matching, which would be more complex.
    
    Args:
        original_edge_index: Original edges [2, total_edges]
        original_batch: Batch indices for original nodes [total_nodes]
        reconstructed: Dictionary containing reconstruction info
        batch_size: Number of graphs in batch
        
    Returns:
        Loss based on edge count differences
    """
    if original_edge_index.size(1) == 0:
        # No edges in original, return zero loss
        return torch.tensor(0.0, device=original_edge_index.device, requires_grad=True)
    
    total_loss = 0.0
    
    # Get reconstructed edge information if available
    recon_edge_index = reconstructed.get('edge_index', None)
    
    for batch_idx in range(batch_size):
        # Count original edges for this graph
        # Get nodes belonging to this graph
        node_mask = (original_batch == batch_idx)
        if not node_mask.any():
            continue
            
        node_indices = torch.where(node_mask)[0]
        node_set = set(node_indices.cpu().numpy())
        
        # Count edges where both source and target are in this graph
        original_edge_count = 0
        for i in range(original_edge_index.size(1)):
            src, dst = original_edge_index[0, i].item(), original_edge_index[1, i].item()
            if src in node_set and dst in node_set:
                original_edge_count += 1
        
        # For reconstructed edges, use a simple heuristic based on node count
        # In a more sophisticated implementation, you'd have actual edge predictions
        num_nodes = node_mask.sum().item()
        
        if recon_edge_index is not None and recon_edge_index.size(1) > 0:
            # Count reconstructed edges for this graph
            # This is a simplification - in practice you'd need better edge tracking
            recon_edge_count = min(recon_edge_index.size(1) // batch_size, num_nodes * 2)
        else:
            # Estimate based on typical AST structure (tree-like with some additional edges)
            recon_edge_count = max(0, num_nodes - 1)  # Tree has n-1 edges
        
        # Compute loss as squared difference in edge counts
        edge_diff = abs(original_edge_count - recon_edge_count)
        total_loss += edge_diff ** 2
    
    # Normalize and return as tensor
    return torch.tensor(total_loss / batch_size, device=original_edge_index.device, requires_grad=True)


def ast_reconstruction_loss_simple(original: Data, reconstructed: Dict[str, Any]) -> torch.Tensor:
    """
    Simplified version of AST reconstruction loss focusing primarily on node prediction.
    
    This version is easier to use and debug, focusing on the core node type prediction
    task which is the most important component for AST reconstruction.
    
    Args:
        original: Original AST as torch_geometric.data.Data object
        reconstructed: Reconstructed AST from decoder
        
    Returns:
        Scalar tensor representing the node type reconstruction loss
    """
    return compute_node_type_loss(original.x, reconstructed['node_features'], original.batch)