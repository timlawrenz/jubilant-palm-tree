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


INV_SIGMOID_ONE = 10.0  # sigmoid(10) ≈ 0.99995 — used for edge clamping


def ode_generate_with_partial_seed(model, motifs, gt_adjacency, num_nodes,
                                    device, num_steps=20, seed_fraction=0.1,
                                    generator=None):
    """Run the ODE with a fraction of ground-truth edges clamped throughout.

    At each ODE step, seeded edges (presence + edge_type) are clamped to their
    ground-truth values while all other edges follow the normal denoising trajectory.

    model: NeuralUniversalMachineDiT
    motifs: [B, N]
    gt_adjacency: [3, N, N] — ground-truth discrete adjacency
    num_nodes: int — real nodes (scoring block)
    seed_fraction: float — fraction of gt edges to seed (0.0 = vanilla)
    generator: optional torch.Generator for reproducibility
    Returns: (discrete [B,3,N,N], seed_positions [n_seed,2])
    """
    B, N = motifs.shape
    gen = generator or torch.Generator(device=device)
    x = torch.randn((B, 6, N, N), device=device, generator=gen)
    dt = 1.0 / num_steps

    # Build seed mask from ground-truth edges in the valid node block
    gt_pres = gt_adjacency[0, :num_nodes, :num_nodes] > 0.5
    gt_edges = torch.nonzero(gt_pres, as_tuple=False)  # [K, 2] on CPU
    K = gt_edges.shape[0]
    n_seed = max(1, int(K * seed_fraction)) if seed_fraction > 0 else 0

    if n_seed > 0:
        # Sample on CPU, then move positions to device for clamping
        perm = torch.randperm(K)
        seed_positions = gt_edges[perm[:n_seed]].to(device)  # [n_seed, 2]
        seed_presence_val = INV_SIGMOID_ONE
        seed_type_raw = gt_adjacency[1, seed_positions[:, 0].cpu(), seed_positions[:, 1].cpu()].to(device)
        seed_type_ode = torch.where(seed_type_raw > 0.5,
                                     torch.tensor(INV_SIGMOID_ONE, device=device),
                                     torch.tensor(-INV_SIGMOID_ONE, device=device))
    else:
        seed_positions = torch.empty((0, 2), dtype=torch.long, device=device)
        seed_presence_val = 0.0
        seed_type_ode = torch.empty(0, device=device)

    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device)
            v = model(x, t_tensor, motifs)

            x = torch.cat([x[:, :2] + v[:, :2] * dt, v[:, 2:]], dim=1)

            if n_seed > 0:
                x[:, 0, seed_positions[:, 0], seed_positions[:, 1]] = seed_presence_val
                x[:, 1, seed_positions[:, 0], seed_positions[:, 1]] = seed_type_ode

    from src.models.constraint_solver import ConstraintSolver
    discrete = ConstraintSolver.discretize_and_repair(x, motifs)
    return discrete, seed_positions

# Per-motif expected degree statistics (computed from 3,951 compressed Ruby ASTs)
# Motif IDs: 1=BOUNDARY, 2=SEQUENCE, 3=CONDITION, 4=LOOP, 5=STATE, 6=MESSAGE
EXPECTED_EXEC_OUT = {1: 1.001, 2: 1.077, 3: 0.289, 4: 0.500, 5: 0.093, 6: 0.169}
EXPECTED_DATA_IN  = {1: 0.000, 2: 0.000, 3: 2.339, 4: 2.000, 5: 0.220, 6: 0.964}


def ode_generate_with_degree_bias(model, motifs, device, num_steps=20,
                                   out_scale=0.0, in_scale=0.0):
    """Run the ODE with per-node degree bias on the presence channel.

    At each step, each node i receives a FiLM-style bias on its row (out-degree)
    and column (in-degree) of the presence channel, nudging toward the expected
    degree for its motif type.

    model: NeuralUniversalMachineDiT (original checkpoint, no modification)
    motifs: [B, N]
    out_scale, in_scale: bias strength for out-degree and in-degree respectively
    """
    B, N = motifs.shape
    x = torch.randn((B, 6, N, N), device=device)
    dt = 1.0 / num_steps

    # Build per-node expected degree tensors from motif lookup
    expected_out = torch.zeros(B, N, device=device)
    expected_in  = torch.zeros(B, N, device=device)
    for mid, val in EXPECTED_EXEC_OUT.items():
        expected_out += val * (motifs == mid).float()
    for mid, val in EXPECTED_DATA_IN.items():
        expected_in  += val * (motifs == mid).float()

    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device)
            v = model(x, t_tensor, motifs)

            x_cont = x[:, :2] + v[:, :2] * dt
            x_cat = v[:, 2:]
            x = torch.cat([x_cont, x_cat], dim=1)

            # Per-node degree bias on the presence channel (channel 0)
            if out_scale > 0 or in_scale > 0:
                presence = torch.sigmoid(x[:, 0])  # [B, N, N]

                if out_scale > 0:
                    current_out = presence.mean(dim=2)  # [B, N] — avg presence per row
                    out_bias = out_scale * (expected_out - current_out)
                    x[:, 0] = x[:, 0] + out_bias.unsqueeze(2)  # add to each row

                if in_scale > 0:
                    current_in = presence.mean(dim=1)  # [B, N] — avg presence per column
                    in_bias = in_scale * (expected_in - current_in)
                    x[:, 0] = x[:, 0] + in_bias.unsqueeze(1)  # add to each column

    from src.models.constraint_solver import ConstraintSolver
    discrete = ConstraintSolver.discretize_and_repair(x, motifs)
    return discrete
