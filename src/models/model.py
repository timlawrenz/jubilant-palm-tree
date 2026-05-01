import torch
from src.models.dit import InputConditioner, GraphDiTBlock

class NeuralUniversalMachineDiT(torch.nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, depth=12, num_motifs=7, motif_dim=16, in_channels=6):
        super().__init__()
        self.conditioner = InputConditioner(num_motifs=num_motifs, motif_dim=motif_dim, in_channels=in_channels)
        
        # 3 + 16 + 16 = 35 channels
        conditioned_channels = self.conditioner.out_channels
        
        # Project up to hidden_dim
        self.proj_in = torch.nn.Conv2d(conditioned_channels, hidden_dim, kernel_size=1)
        
        # Timestep embedding (standard sinusoidal can be used, we'll assume an external t_emb generator or simple linear for MVP)
        self.t_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256)
        )
        
        # The Axial Attention Blocks
        self.blocks = torch.nn.ModuleList([
            GraphDiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, timestep_dim=256)
            for _ in range(depth)
        ])
        
        # Project back down to output channels
        # Channels 0 and 1 are continuous velocity for Flow Matching (Presence, EdgeType)
        # Channels 2 to 2+K are the Categorical Classification logits for input_index
        # So total output channels = 2 + num_index_classes
        self.num_index_classes = 4 # 0, 1, 2, 3
        self.proj_out = torch.nn.Conv2d(hidden_dim, 2 + self.num_index_classes, kernel_size=1)
        
    def forward(self, x_t, t, motifs):
        """
        x_t: [B, 3, N, N]
        t: [B]
        motifs: [B, N]
        """
        # Embed timestep
        t_emb = self.t_mlp(t.unsqueeze(1)) # [B, 256]
        
        # Fuse Motifs into spatial matrix: [B, 35, N, N]
        x = self.conditioner(x_t, motifs)
        
        # [B, hidden_dim, N, N]
        x = self.proj_in(x)
        
        # Message Passing
        for block in self.blocks:
            x = block(x, t_emb)
            
        # Predict Velocity: [B, 3, N, N]
        v_pred = self.proj_out(x)
        return v_pred
