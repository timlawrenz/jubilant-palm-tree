import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class InputConditioner(nn.Module):
    """
    Cross-Hatch Embedding Injection
    Fuses the 1D Motif identities into the 2D adjacency matrix space so the DiT 
    knows the structural identity of the source and target nodes for every edge.

    Optional degree-conditioning branch (Exp 1.9): per-node expected in-degree
    scalars perturb the motif embeddings BEFORE grid expansion, so the enrichment
    signal is baked into the conditioning path at training time. Shapes downstream
    are unchanged, so pretrained checkpoints load with strict=False.
    """
    def __init__(self, num_motifs: int = 7, motif_dim: int = 16, in_channels: int = 3,
                 use_degree: bool = False, degree_dim: int = 1):
        super().__init__()
        # 0=Padding, 1=Boundary, 2=Sequence, 3=Condition, 4=Loop, 5=State, 6=Message
        self.embedding = nn.Embedding(num_embeddings=num_motifs, embedding_dim=motif_dim)
        self.out_channels = in_channels + (motif_dim * 2) # e.g. 6 + 16 + 16 = 38
        self.use_degree = use_degree
        if use_degree:
            # Maps per-node scalar degree -> motif_dim perturbation added to embeddings
            self.degree_proj = nn.Linear(degree_dim, motif_dim)

    def forward(self, noisy_adj: torch.Tensor, motifs: torch.Tensor,
                degrees: torch.Tensor | None = None) -> torch.Tensor:
        """
        noisy_adj: [B, in_channels, N, N]
        motifs: [B, N]
        degrees: [B, N] optional per-node scalar enrichment (Exp 1.9)
        Returns: [B, out_channels, N, N]
        """
        B, C, N, _ = noisy_adj.shape
        
        # [B, N, 16]
        motif_embeds = self.embedding(motifs)
        
        if self.use_degree and degrees is not None:
            # [B, N] -> [B, N, 1] -> [B, N, 16] additive perturbation
            motif_embeds = motif_embeds + self.degree_proj(degrees.unsqueeze(-1))
        
        # 1. Expand to represent Source Nodes (rows)
        source_grid = motif_embeds.permute(0, 2, 1).unsqueeze(-1).expand(-1, -1, -1, N)
        
        # 2. Expand to represent Target Nodes (columns)
        target_grid = motif_embeds.permute(0, 2, 1).unsqueeze(2).expand(-1, -1, N, -1)
        
        # 3. Fuse
        dit_input = torch.cat([noisy_adj, source_grid, target_grid], dim=1)
        return dit_input


class AdaLN(nn.Module):
    def __init__(self, hidden_dim: int, timestep_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_dim, hidden_dim * 2)
        )
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H, W], we need to apply AdaLN to the channel dim
        # Let's permute to [B, H, W, C] for LayerNorm
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm)
        
        # t_emb shape: [B, timestep_dim] -> proj -> [B, 2*C]
        scale_shift = self.proj(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        
        # Reshape to [B, 1, 1, C] for broadcasting
        scale = scale.unsqueeze(1).unsqueeze(1)
        shift = shift.unsqueeze(1).unsqueeze(1)
        
        x_out = x_norm * (1 + scale) + shift
        # Permute back to [B, C, H, W]
        return x_out.permute(0, 3, 1, 2)


class Mlp(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, int(hidden_dim * mlp_ratio))
        self.act = nn.GELU()
        self.fc2 = nn.Linear(int(hidden_dim * mlp_ratio), hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class GraphDiTBlock(nn.Module):
    """
    Message-Passing DiT Block using Axial (Row-Column) Attention.
    """
    def __init__(self, hidden_dim: int, num_heads: int, timestep_dim: int = 256):
        super().__init__()
        # AdaLN for injecting the Diffusion Timestep (t)
        self.adaln = AdaLN(hidden_dim, timestep_dim) 
        
        self.row_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.col_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        self.mlp = Mlp(hidden_dim)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, 128, 128]
        B, C, H, W = x.shape
        
        # 1. Condition on Timestep
        x_conditioned = self.adaln(x, t_emb)
        
        def run_row_attn(x_in):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_r = x_in.permute(0, 2, 3, 1).reshape(B * H, W, C)
                x_r_out, _ = self.row_attn(x_r, x_r, x_r, need_weights=False)
                return x_r_out.reshape(B, H, W, C).permute(0, 3, 1, 2)
            
        def run_col_attn(x_in):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                x_c = x_in.permute(0, 3, 2, 1).reshape(B * W, H, C)
                x_c_out, _ = self.col_attn(x_c, x_c, x_c, need_weights=False)
                return x_c_out.reshape(B, W, H, C).permute(0, 3, 2, 1)
            
        def run_mlp(x_in):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                return self.mlp(x_in.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # 2-4. Checkpointed blocks — use_reentrant=True correctly inherits ambient autocast
        #      context during backward recompute, avoiding shape-mismatch with bfloat16 AMP.
        if x.requires_grad:
            x = x + checkpoint(run_row_attn, x_conditioned, use_reentrant=True)
            x = x + checkpoint(run_col_attn, x, use_reentrant=True)
            x = x + checkpoint(run_mlp, x, use_reentrant=True)
        else:
            x = x + run_row_attn(x_conditioned)
            x = x + run_col_attn(x)
            x = x + run_mlp(x)
        
        return x

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
    
    # Test GraphDiTBlock
    block = GraphDiTBlock(hidden_dim=35, num_heads=7) # 35 is divisible by 7
    t_emb = torch.randn(B, 256)
    out = block(dit_input, t_emb)
    print(f"GraphDiTBlock output: {out.shape}")
    assert out.shape == (B, 35, N, N)
    print("GraphDiTBlock successful!")
