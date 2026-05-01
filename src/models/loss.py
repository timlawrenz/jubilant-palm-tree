import torch

def compute_flow_matching_loss(model, batch):
    """
    Computes the masked Conditional Flow Matching (Optimal Transport) Loss.
    """
    x_1 = batch["adjacency"]       # [B, 3, N, N]
    motifs = batch["motifs"]       # [B, N]
    mask = batch["padding_mask"]   # [B, N, N]
    
    B, C, N, _ = x_1.shape
    device = x_1.device
    
    # 2. Sample random timesteps t ~ Uniform(0, 1)
    # Shape: [B, 1, 1, 1] for broadcasting
    t = torch.rand((B,), device=device)
    t_expand = t.view(B, 1, 1, 1)
    
    # 3. Generate pure noise
    x_0 = torch.randn_like(x_1)
    
    # 4. Calculate the interpolated matrix x_t
    x_t = (1 - t_expand) * x_0 + t_expand * x_1
    
    # 5. Calculate the True Velocity (The Target)
    target_velocity = x_1 - x_0
    
    # 6. Forward Pass through the DiT model
    # Model should encapsulate InputConditioner internally, or we call it explicitly
    predicted_velocity = model(x_t, t, motifs) 
    
    # 7. The Masked MSE Loss
    # Expand mask to cover the 3 edge channels: [B, 3, N, N]
    mask_expanded = mask.unsqueeze(1).expand(-1, C, -1, -1)
    
    # Calculate squared error
    squared_error = (predicted_velocity - target_velocity) ** 2
    
    # Zero out the error in the padded regions!
    masked_error = squared_error * mask_expanded
    
    # Average the loss ONLY over the valid (unmasked) elements
    loss = masked_error.sum() / max(mask_expanded.sum(), 1)
    
    return loss
