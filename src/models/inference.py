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
    # Channels: 0=Presence, 1=EdgeType, plus 4 noise channels for the categorical logits input
    x = torch.randn((B, 6, MAX_NODES, MAX_NODES), device=device)
    
    # 2. Time steps
    dt = 1.0 / num_steps
    
    for step in range(num_steps):
        # Current time t
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=device)
        
        # Predict the constant velocity vector field
        velocity = model(x, t_tensor, motifs)
        
        # Extract only the 2 continuous velocity channels (Presence, Type) to integrate
        velocity_continuous = velocity[:, :2, :, :]
        x_continuous = x[:, :2, :, :]
        x_continuous = x_continuous + (velocity_continuous * dt)
        
        # We also need to keep track of the categorical logits, so we just take the latest prediction
        # Since logits aren't integrated like velocity, we just hold the raw prediction for Step 3
        velocity_categorical_logits = velocity[:, 2:, :, :]
        
        # Update our x tensor
        x = torch.cat([x_continuous, velocity_categorical_logits], dim=1)
    
    # 3. Discretize the Continuous Output
    # The output 'x' is now a continuous approximation.
    # Channel 0 (Presence) and Channel 1 (Type) are basically binary probabilities.
    # Channel 2 is now categorical classification logits [B, K, N, N]
    
    presence_continuous = x[:, 0:1, :, :]
    type_continuous = x[:, 1:2, :, :]
    
    presence_discrete = (torch.sigmoid(presence_continuous) > 0.5).int()
    type_discrete = (torch.sigmoid(type_continuous) > 0.5).int()
    
    index_logits = x[:, 2:, :, :]
    index_discrete = torch.argmax(index_logits, dim=1).int()
    
    # Recombine
    discrete_adjacency = torch.cat([presence_discrete, type_discrete, index_discrete.unsqueeze(1)], dim=1)
    
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
