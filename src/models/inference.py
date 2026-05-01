import torch
from src.models.model import NeuralUniversalMachineDiT

@torch.no_grad()
def sample_graph(model: NeuralUniversalMachineDiT, motifs: torch.Tensor, num_steps: int = 20) -> torch.Tensor:
    """
    Generates a discrete execution graph adjacency matrix using Euler ODE solver.
    
    Args:
        model: Trained NeuralUniversalMachineDiT
        motifs: [B, MAX_NODES] tensor of motif integer classes (the ingredients)
        num_steps: Number of integration steps for the Euler solver
        
    Returns: 
        [B, 3, MAX_NODES, MAX_NODES] discrete adjacency matrix (0s and 1s)
    """
    model.eval()
    B, MAX_NODES = motifs.shape
    device = motifs.device
    
    # 1. Start at pure noise (t = 0)
    # The channels are: 0=Presence, 1=EdgeType, 2=InputIndex
    x = torch.randn((B, 3, MAX_NODES, MAX_NODES), device=device)
    
    # 2. Time steps
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        # Current time t
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=device)
        
        # Predict the constant velocity vector field
        velocity = model(x, t_tensor, motifs)
        
        # Take an Euler step: x_{t+dt} = x_t + v * dt
        x = x + (velocity * dt)
    
    # 3. Discretize the Continuous Output
    # The output 'x' is now a continuous approximation.
    # Channel 0 (Presence) and Channel 1 (Type) are basically binary probabilities.
    # Channel 2 (InputIndex) is ordinal, but we'll round it cleanly.
    
    # Separate the channels for careful discretization
    presence_continuous = x[:, 0:1, :, :]
    type_continuous = x[:, 1:2, :, :]
    index_continuous = x[:, 2:3, :, :]
    
    # Squash and threshold binary channels
    presence_discrete = (torch.sigmoid(presence_continuous) > 0.5).int()
    type_discrete = (torch.sigmoid(type_continuous) > 0.5).int()
    
    # Round the ordinal index channel (e.g., 0.9 -> 1, 0.1 -> 0)
    # We only care about this where presence is 1, but we can round the whole thing safely
    index_discrete = torch.round(index_continuous).int()
    # Ensure index doesn't go below 0
    index_discrete = torch.clamp(index_discrete, min=0)
    
    # Recombine
    discrete_adjacency = torch.cat([presence_discrete, type_discrete, index_discrete], dim=1)
    
    # 4. Clean up the padding
    # Ensure any rows/cols where motifs == 0 (padding) remain 0
    padding_mask = (motifs != 0).float() # [B, N]
    mask_2d = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1) # [B, N, N]
    
    # Apply to all channels [B, 3, N, N]
    discrete_adjacency = discrete_adjacency * mask_2d.unsqueeze(1).int()
    
    return discrete_adjacency

if __name__ == "__main__":
    # Test the inference loop
    model = NeuralUniversalMachineDiT(hidden_dim=64, num_heads=4, depth=2)
    dummy_motifs = torch.tensor([[1, 5, 5, 4, 6, 6, 1] + [0]*121]) # [1, 128]
    
    print("Running 20-step Euler ODE solver on DiT...")
    out_matrix = sample_graph(model, dummy_motifs, num_steps=20)
    
    print(f"Output shape: {out_matrix.shape}")
    print(f"Discrete types present in Presence channel: {torch.unique(out_matrix[:, 0, :, :])}")
    print("Inference loop successful!")
