"""
Naive Threshold Discretizer for RLAIF.

Replaces the JudicialConstraintSolver during RLAIF training.
No arity snapping, no collision resolution — raw model output only.
The model must learn to produce legal graphs without post-hoc repair.
"""

import torch


def naive_discretize(continuous_matrix: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
    """
    Applies simple thresholds to convert the continuous ODE output into
    a discrete adjacency matrix. No topological repair.

    Args:
        continuous_matrix: [B, 6, N, N] — final integrated state from Euler solver.
            Channels 0-1: continuous presence and edge type values.
            Channels 2-5: categorical logits for input_index.
        motifs: [B, N] — motif type IDs (used only for padding mask).

    Returns:
        [B, 3, N, N] discrete adjacency matrix:
            Channel 0: presence (0 or 1)
            Channel 1: edge type (0=Exec, 1=Data)
            Channel 2: input_index (0-3)
    """
    B, C, N, _ = continuous_matrix.shape

    # Threshold presence via sigmoid > 0.5
    presence = (torch.sigmoid(continuous_matrix[:, 0, :, :]) > 0.5).int()

    # Threshold edge type via sigmoid > 0.5
    edge_type = (torch.sigmoid(continuous_matrix[:, 1, :, :]) > 0.5).int()

    # Argmax over the 4 categorical index logits
    index_logits = continuous_matrix[:, 2:, :, :]  # [B, 4, N, N]
    input_index = torch.argmax(index_logits, dim=1).int()  # [B, N, N]

    # Apply padding mask: zero out any edge involving a padding node
    padding_mask = (motifs != 0).float()  # [B, N]
    mask_2d = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)  # [B, N, N]

    presence = presence * mask_2d.int()
    edge_type = edge_type * mask_2d.int()
    input_index = input_index * mask_2d.int()

    # Stack into [B, 3, N, N]
    discrete_adjacency = torch.stack([presence, edge_type, input_index], dim=1)

    return discrete_adjacency
