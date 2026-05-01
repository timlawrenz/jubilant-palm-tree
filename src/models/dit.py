import torch
import torch.nn as nn

class InputConditioner(nn.Module):
    """
    Cross-Hatch Embedding Injection
    Fuses the 1D Motif identities into the 2D adjacency matrix space so the DiT 
    knows the structural identity of the source and target nodes for every edge.
    """
    def __init__(self, num_motifs: int = 7, motif_dim: int = 16, in_channels: int = 3):
        super().__init__()
        # 0=Padding, 1=Boundary, 2=Sequence, 3=Condition, 4=Loop, 5=State, 6=Message
        self.embedding = nn.Embedding(num_embeddings=num_motifs, embedding_dim=motif_dim)
        self.out_channels = in_channels + (motif_dim * 2) # e.g. 3 + 16 + 16 = 35

    def forward(self, noisy_adj: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        """
        noisy_adj: [B, 3, N, N]
        motifs: [B, N]
        Returns: [B, 35, N, N]
        """
        B, C, N, _ = noisy_adj.shape
        
        # [B, N, 16]
        motif_embeds = self.embedding(motifs)
        
        # 1. Expand to represent Source Nodes (rows)
        # Shape: [B, 16, N, 1] -> broadcast to -> [B, 16, N, N]
        source_grid = motif_embeds.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, N)
        
        # 2. Expand to represent Target Nodes (columns)
        # Shape: [B, 16, 1, N] -> broadcast to -> [B, 16, N, N]
        target_grid = motif_embeds.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, N, -1)
        
        # 3. Fuse the structural noise with the semantic node identities
        # Final Shape: [B, 35, N, N]
        dit_input = torch.cat([noisy_adj, source_grid, target_grid], dim=1)
        
        return dit_input

if __name__ == "__main__":
    # Test the cross-hatch injection
    B, N = 4, 128
    noisy_adj = torch.randn(B, 3, N, N)
    motifs = torch.randint(0, 7, (B, N))
    
    conditioner = InputConditioner()
    dit_input = conditioner(noisy_adj, motifs)
    
    print(f"Noisy Adjacency: {noisy_adj.shape}")
    print(f"Motifs: {motifs.shape}")
    print(f"Cross-Hatched DiT Input: {dit_input.shape}")
    assert dit_input.shape == (B, 35, N, N)
    print("Cross-Hatch injection successful!")
