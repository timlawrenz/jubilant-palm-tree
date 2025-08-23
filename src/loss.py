"""
Loss functions for AST reconstruction tasks and contrastive learning.

This module provides loss functions for:
1. Measuring the difference between original and reconstructed Abstract Syntax Trees
2. Contrastive learning between code and text embeddings for alignment
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict, Any, Union


def ast_reconstruction_loss_comprehensive(original: Data, reconstructed: Dict[str, Any], 
                                        node_weight: float = 1.0, parent_weight: float = 1.0) -> torch.Tensor:
    """
    Computes a comprehensive reconstruction loss for an AST.

    This loss combines:
    1. Node Type Loss: Cross-entropy for predicting the correct node types.
    2. Parent Prediction Loss: Cross-entropy for predicting the correct parent for each node.
    """
    # --- Node Type Loss ---
    recon_node_logits = reconstructed['node_features'].squeeze(0)
    
    # Numerical stability: Clamp values to a reasonable range to prevent overflow
    recon_node_logits = torch.clamp(recon_node_logits, min=-100, max=100)
    
    true_node_types = original.x.argmax(dim=1)
    
    num_nodes = min(recon_node_logits.size(0), true_node_types.size(0))
    if num_nodes == 0:
        return torch.tensor(0.0, device=original.x.device, requires_grad=True)
        
    node_loss = F.cross_entropy(
        recon_node_logits[:num_nodes], 
        true_node_types[:num_nodes]
    )

    # --- Parent Prediction Loss ---
    recon_parent_logits = reconstructed['parent_logits'].squeeze(0) # [num_nodes, max_nodes]
    
    # Numerical stability: Clamp values
    recon_parent_logits = torch.clamp(recon_parent_logits, min=-100, max=100)
    
    max_nodes = recon_parent_logits.size(1)
    
    # Create the true parent labels
    num_true_nodes = original.num_nodes
    # Initialize with an ignore_index
    ignore_index = -100 
    true_parents = torch.full((num_true_nodes,), ignore_index, dtype=torch.long, device=original.x.device)
    
    # Edge index is [parent, child], so edge_index[0] are parents and edge_index[1] are children
    children = original.edge_index[1]
    parents = original.edge_index[0]

    # Clamp parent indices to be within the prediction range [0, max_nodes-1]
    valid_parents = torch.clamp(parents, 0, max_nodes - 1)
    true_parents[children] = valid_parents

    # We only care about the first num_nodes predictions and labels
    num_nodes = min(recon_parent_logits.size(0), true_parents.size(0))

    # Check if there are any valid parent labels to compute loss on
    if (true_parents[:num_nodes] != ignore_index).any():
        parent_loss = F.cross_entropy(
            recon_parent_logits[:num_nodes],
            true_parents[:num_nodes],
            ignore_index=ignore_index
        )
    else:
        # No valid parents to compute loss on (e.g., single-node graph)
        parent_loss = torch.tensor(0.0, device=original.x.device, requires_grad=True)

    # --- Total Loss ---
    total_loss = (node_weight * node_loss) + (parent_weight * parent_loss)
    return total_loss


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
        
        # Numerical stability: Check for and handle NaN/Inf values in reconstructed logits
        if torch.isnan(recon_nodes).any() or torch.isinf(recon_nodes).any():
            # Replace NaN/Inf with safe values to prevent loss explosion
            recon_nodes = torch.where(torch.isnan(recon_nodes), torch.zeros_like(recon_nodes), recon_nodes)
            recon_nodes = torch.clamp(recon_nodes, min=-100, max=100)  # Clamp to reasonable range
        
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
    
    # --- Vectorized implementation to avoid CPU bottlenecks ---
    
    # 1. Get the batch index for the source node of each edge
    edge_batch_indices = original_batch[original_edge_index[0]]

    # 2. Count the number of edges for each graph in the batch
    # `bincount` is a highly optimized way to count occurrences of each index
    original_edge_counts = torch.bincount(edge_batch_indices, minlength=batch_size).float()

    # 3. Estimate reconstructed edge counts (maintaining original logic)
    # Get the number of nodes in each graph of the batch
    num_nodes_per_graph = torch.bincount(original_batch, minlength=batch_size).float()
    # Estimate edge count as num_nodes - 1 (for a tree-like structure)
    recon_edge_counts = torch.clamp(num_nodes_per_graph - 1, min=0)

    # 4. Compute the loss as the mean squared error between the counts
    # This is a single, fast, vectorized operation
    loss = F.mse_loss(recon_edge_counts, original_edge_counts)
    
    return loss


def ast_reconstruction_loss_improved(original: Data, reconstructed: Dict[str, Any],
                                   type_weight: float = 1.0, 
                                   parent_weight: float = 1.0) -> torch.Tensor:
    """
    Improved AST reconstruction loss with explicit parent prediction.
    
    This loss function provides a strong structural learning signal by combining
    node type prediction with explicit parent prediction for each node.
    
    Args:
        original: Original AST as a single torch_geometric.data.Data object (for one graph).
        reconstructed: Reconstructed AST from decoder containing 'node_features' and 'parent_logits'.
        type_weight: Weight for the node type prediction loss.
        parent_weight: Weight for the parent prediction loss.
        
    Returns:
        Scalar tensor representing the total weighted reconstruction loss.
    """
    # --- Component 1: Node Type Loss ---
    # Squeeze to remove the batch dimension of 1
    recon_node_logits = reconstructed['node_features'].squeeze(0)
    true_node_types = original.x.argmax(dim=1)
    
    # The decoder might predict a different number of nodes than the original.
    # We compute the loss only for the minimum number of nodes between the two.
    num_nodes = min(recon_node_logits.size(0), true_node_types.size(0))
    if num_nodes == 0:
        return torch.tensor(0.0, device=original.x.device, requires_grad=True)
        
    type_loss = F.cross_entropy(
        recon_node_logits[:num_nodes], 
        true_node_types[:num_nodes]
    )

    # --- Component 2: Parent Prediction Loss ---
    recon_parent_logits = reconstructed['parent_logits'].squeeze(0) # Shape: [num_nodes, max_nodes]
    max_nodes = recon_parent_logits.size(1)
    
    # Create the ground truth parent labels for the original graph
    num_true_nodes = original.num_nodes
    ignore_index = -100 # A standard ignore index for cross-entropy
    true_parents = torch.full((num_true_nodes,), ignore_index, dtype=torch.long, device=original.x.device)
    
    # The root node has no parent, so it will be ignored in the loss calculation.
    # For all other nodes, find their parent from the edge index.
    # original.edge_index is [parent_indices, child_indices]
    children = original.edge_index[1]
    parents = original.edge_index[0]

    # Populate the true_parents tensor, clamping indices to be within the prediction range.
    valid_parents = torch.clamp(parents, 0, max_nodes - 1)
    true_parents[children] = valid_parents

    # We only compute the loss for the nodes that were actually reconstructed.
    num_nodes_for_loss = min(recon_parent_logits.size(0), true_parents.size(0))

    # Check if there are any valid parent-child relationships to compute loss on.
    if (true_parents[:num_nodes_for_loss] != ignore_index).any():
        parent_loss = F.cross_entropy(
            recon_parent_logits[:num_nodes_for_loss],
            true_parents[:num_nodes_for_loss],
            ignore_index=ignore_index
        )
    else:
        # This case handles single-node graphs (which have no parents).
        parent_loss = torch.tensor(0.0, device=original.x.device)

    # --- Total Loss ---
    total_loss = (type_weight * type_loss) + (parent_weight * parent_loss)
    return total_loss


def _compute_role_loss(original: Data, reconstructed: Dict[str, Any]) -> torch.Tensor:
    """
    Compute role loss component for improved AST reconstruction.
    
    This function computes a loss that encourages the model to understand the 
    functional role of identifiers (e.g., method argument, local variable).
    
    For backward compatibility with current one-hot node features, this implements
    a simplified role-aware loss based on node types and graph structure.
    In the future, this will use dedicated role embeddings.
    
    Args:
        original: Original AST data
        reconstructed: Reconstructed AST data
        
    Returns:
        Scalar tensor representing the role loss
    """
    recon_node_features = reconstructed['node_features']
    batch_size = recon_node_features.size(0)
    
    # For backward compatibility, derive role information from node types and graph structure
    # This is a simplified approach until dedicated role features are implemented
    
    total_loss = 0.0
    total_nodes = 0
    
    for batch_idx in range(batch_size):
        # Get original nodes for this graph
        mask = (original.batch == batch_idx)
        if not mask.any():
            continue
            
        original_nodes = original.x[mask]  # [num_nodes_in_graph, feature_dim]
        num_original_nodes = original_nodes.size(0)
        
        # Get node types for role inference
        original_node_types = torch.argmax(original_nodes, dim=1)
        
        # Simple role-based loss: encourage consistency in how similar node types are handled
        # This approximates role understanding until full role features are available
        if num_original_nodes > 1:
            # Create a simple role similarity matrix based on node types
            type_similarity = (original_node_types.unsqueeze(0) == original_node_types.unsqueeze(1)).float()
            
            # Get reconstructed features for this batch
            max_nodes = min(num_original_nodes, recon_node_features.size(1))
            recon_features = recon_node_features[batch_idx, :max_nodes, :]
            
            # Compute pairwise similarities in reconstructed space
            recon_normalized = F.normalize(recon_features, p=2, dim=1)
            recon_similarity = torch.matmul(recon_normalized, recon_normalized.t())
            
            # Encourage similar node types to have similar representations (role consistency)
            role_consistency_loss = F.mse_loss(recon_similarity, type_similarity[:max_nodes, :max_nodes])
            total_loss += role_consistency_loss
            total_nodes += 1
    
    # Return average loss
    if total_nodes > 0:
        avg_loss = total_loss / total_nodes
        if isinstance(avg_loss, torch.Tensor):
            return avg_loss.requires_grad_(True)
        else:
            return torch.tensor(avg_loss, device=original.x.device, requires_grad=True)
    else:
        return torch.tensor(0.0, device=original.x.device, requires_grad=True)


def _compute_name_loss(original: Data, reconstructed: Dict[str, Any]) -> torch.Tensor:
    """
    Compute name loss component for improved AST reconstruction.
    
    This function computes a loss that lightly encourages the model to use
    appropriate names while not penalizing heavily for choosing different
    but valid names.
    
    For backward compatibility with current features, this implements a 
    placeholder loss that encourages semantic consistency.
    In the future, this will use dedicated name embeddings.
    
    Args:
        original: Original AST data
        reconstructed: Reconstructed AST data
        
    Returns:
        Scalar tensor representing the name loss
    """
    recon_node_features = reconstructed['node_features']
    batch_size = recon_node_features.size(0)
    
    # For backward compatibility, implement a lightweight semantic consistency loss
    # This will be replaced with proper name embedding loss in the future
    
    total_loss = 0.0
    total_nodes = 0
    
    for batch_idx in range(batch_size):
        # Get original nodes for this graph
        mask = (original.batch == batch_idx)
        if not mask.any():
            continue
            
        original_nodes = original.x[mask]
        num_original_nodes = original_nodes.size(0)
        
        # Get reconstructed features
        max_nodes = min(num_original_nodes, recon_node_features.size(1))
        recon_features = recon_node_features[batch_idx, :max_nodes, :]
        
        # Lightweight semantic consistency: encourage reconstructed features to maintain
        # relative relationships present in original (approximates name consistency)
        if max_nodes > 1:
            # Compute cosine similarities in both spaces
            orig_normalized = F.normalize(original_nodes[:max_nodes], p=2, dim=1)
            recon_normalized = F.normalize(recon_features, p=2, dim=1)
            
            orig_similarities = torch.matmul(orig_normalized, orig_normalized.t())
            recon_similarities = torch.matmul(recon_normalized, recon_normalized.t())
            
            # Light penalty for changing semantic relationships (low weight applied externally)
            semantic_consistency_loss = F.mse_loss(recon_similarities, orig_similarities)
            total_loss += semantic_consistency_loss
            total_nodes += 1
    
    # Return average loss
    if total_nodes > 0:
        avg_loss = total_loss / total_nodes
        if isinstance(avg_loss, torch.Tensor):
            return avg_loss.requires_grad_(True)
        else:
            return torch.tensor(avg_loss, device=original.x.device, requires_grad=True)
    else:
        return torch.tensor(0.0, device=original.x.device, requires_grad=True)


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


# ============================================================================
# Contrastive Loss Functions for Code-Text Alignment (Phase 5)
# ============================================================================

def info_nce_loss(code_embeddings: torch.Tensor, text_embeddings: torch.Tensor, 
                  temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE (Information Noise Contrastive Estimation) loss for contrastive learning.
    
    This loss encourages correct (code, text) pairs to have high similarity while
    pushing incorrect pairs to have low similarity. It's commonly used in 
    contrastive learning and multimodal alignment.
    
    Args:
        code_embeddings: Code embeddings tensor of shape [batch_size, embedding_dim]
        text_embeddings: Text embeddings tensor of shape [batch_size, embedding_dim]
        temperature: Temperature parameter for scaling similarities (higher = softer)
        
    Returns:
        Scalar tensor representing the InfoNCE loss
        
    Note:
        Assumes that code_embeddings[i] and text_embeddings[i] form a positive pair,
        while all other combinations are negative pairs.
    """
    batch_size = code_embeddings.size(0)
    
    # Normalize embeddings to unit vectors for stable cosine similarity
    code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Compute similarity matrix: [batch_size, batch_size]
    # similarity[i, j] = similarity between code[i] and text[j]
    similarity_matrix = torch.matmul(code_embeddings, text_embeddings.t()) / temperature
    
    # Create labels: positive pairs are on the diagonal
    labels = torch.arange(batch_size, device=code_embeddings.device)
    
    # InfoNCE loss is cross-entropy between similarity scores and correct indices
    # For each code embedding, we want the corresponding text embedding to have highest similarity
    loss_code_to_text = F.cross_entropy(similarity_matrix, labels)
    
    # Symmetric loss: for each text embedding, we want the corresponding code embedding to have highest similarity
    loss_text_to_code = F.cross_entropy(similarity_matrix.t(), labels)
    
    # Return average of both directions
    return (loss_code_to_text + loss_text_to_code) / 2.0


def cosine_embedding_loss(code_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                         margin: float = 0.2) -> torch.Tensor:
    """
    Simple cosine embedding loss for contrastive learning.
    
    This loss encourages positive pairs to have high cosine similarity (close to 1)
    and negative pairs to have low cosine similarity (below margin).
    
    Args:
        code_embeddings: Code embeddings tensor of shape [batch_size, embedding_dim]
        text_embeddings: Text embeddings tensor of shape [batch_size, embedding_dim]
        margin: Margin for negative pairs (similarity should be below this)
        
    Returns:
        Scalar tensor representing the cosine embedding loss
    """
    batch_size = code_embeddings.size(0)
    
    # Normalize embeddings for stable cosine similarity
    code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Compute cosine similarities for all pairs
    similarity_matrix = torch.matmul(code_embeddings, text_embeddings.t())
    
    # Positive pairs: diagonal elements (code[i] with text[i])
    positive_similarities = torch.diag(similarity_matrix)
    
    # Loss for positive pairs: encourage high similarity (target = 1)
    positive_loss = F.mse_loss(positive_similarities, torch.ones_like(positive_similarities))
    
    # For negative pairs, only apply if we have more than one sample
    if batch_size > 1:
        # Negative pairs: off-diagonal elements
        mask = torch.eye(batch_size, device=code_embeddings.device).bool()
        negative_similarities = similarity_matrix[~mask]
        
        # Loss for negative pairs: encourage low similarity (below margin)
        # Only penalize if similarity is above margin
        negative_loss = F.relu(negative_similarities - margin).mean()
    else:
        # No negative pairs when batch size is 1
        negative_loss = torch.tensor(0.0, device=code_embeddings.device)
    
    # Combine losses
    return positive_loss + negative_loss


def simple_contrastive_loss(code_embeddings: torch.Tensor, text_embeddings: torch.Tensor,
                           temperature: float = 0.1) -> torch.Tensor:
    """
    Simplified contrastive loss using cosine similarity.
    
    This is a straightforward implementation that maximizes cosine similarity
    between correct pairs and minimizes it for incorrect pairs.
    
    Args:
        code_embeddings: Code embeddings tensor of shape [batch_size, embedding_dim]
        text_embeddings: Text embeddings tensor of shape [batch_size, embedding_dim]
        temperature: Temperature for scaling similarities
        
    Returns:
        Scalar tensor representing the contrastive loss
    """
    # Normalize embeddings
    code_embeddings = F.normalize(code_embeddings, p=2, dim=1)
    text_embeddings = F.normalize(text_embeddings, p=2, dim=1)
    
    # Compute cosine similarities
    similarities = F.cosine_similarity(code_embeddings, text_embeddings, dim=1)
    
    # Loss is simply negative mean similarity (we want to maximize similarity)
    # Scale by temperature for better gradient flow
    return -similarities.mean() / temperature
