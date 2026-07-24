import torch
from src.models.dit import InputConditioner, GraphDiTBlock


class NeuralUniversalMachineDiT(torch.nn.Module):
    def __init__(self, hidden_dim=256, num_heads=8, depth=12, num_motifs=7, motif_dim=16, in_channels=6):
        super().__init__()
        self.conditioner = InputConditioner(num_motifs=num_motifs, motif_dim=motif_dim, in_channels=in_channels)
        conditioned_channels = self.conditioner.out_channels
        
        self.proj_in = torch.nn.Conv2d(conditioned_channels, hidden_dim, kernel_size=1)
        
        self.t_mlp = torch.nn.Sequential(
            torch.nn.Linear(1, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256)
        )
        
        self.blocks = torch.nn.ModuleList([
            GraphDiTBlock(hidden_dim=hidden_dim, num_heads=num_heads, timestep_dim=256)
            for _ in range(depth)
        ])
        
        self.num_index_classes = 4
        self.proj_out = torch.nn.Conv2d(hidden_dim, 2 + self.num_index_classes, kernel_size=1)
        
    def forward(self, x_t, t, motifs):
        t_emb = self.t_mlp(t.unsqueeze(1))
        x = self.conditioner(x_t, motifs)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x, t_emb)
        v_pred = self.proj_out(x)
        return v_pred


def ode_generate_with_density_bias(model, motifs, edge_density, device, num_steps=20,
                                   density_bias_scale=1.0):
    """Run the 20-step ODE with edge-density bias injected into the presence channel.
    
    At each ODE step, the presence channel (x[:, 0]) receives a small additive
    bias toward the target density, computed as:
        bias = density_bias_scale * (target_density - current_density)
    where current_density is estimated from the current sigmoid(presence) values.
    
    model: NeuralUniversalMachineDiT (original checkpoint, no modification needed)
    motifs: [B, N]
    edge_density: [B] — scalar target density (e.g., k / (num_nodes * (num_nodes-1)))
    """
    B, N = motifs.shape
    x = torch.randn((B, 6, N, N), device=device)
    dt = 1.0 / num_steps
    
    # Broadcast density to [B, 1, 1, 1] for channel-0 bias
    density_b = edge_density.view(B, 1, 1, 1)
    
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device)
            v = model(x, t_tensor, motifs)
            
            # ODE step
            x_cont = x[:, :2] + v[:, :2] * dt
            x_cat = v[:, 2:]
            x = torch.cat([x_cont, x_cat], dim=1)
            
            # FiLM-style density bias on the presence channel (channel 0)
            if density_bias_scale > 0:
                current_presence = torch.sigmoid(x[:, 0:1])
                current_density = current_presence.mean(dim=(2, 3), keepdim=True)  # [B, 1, 1, 1]
                bias = density_bias_scale * (density_b - current_density)
                x[:, 0:1] = x[:, 0:1] + bias
    
    from src.models.constraint_solver import ConstraintSolver
    discrete = ConstraintSolver.discretize_and_repair(x, motifs)
    return discrete
