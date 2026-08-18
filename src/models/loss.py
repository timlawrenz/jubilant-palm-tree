import torch
import torch.nn.functional as F

def compute_flow_matching_loss(model, batch, degrees=None):
    """
    Computes the masked Conditional Flow Matching Loss for Channels 0,1 
    AND CrossEntropyLoss for Channel 2 (input_index).

    degrees: optional [B, N] per-node enrichment scalars forwarded to the
             model's conditioning branch (Exp 1.9 training-time enrichment).
    """
    x_1 = batch["adjacency"]       # [B, 3, N, N]
    motifs = batch["motifs"]       # [B, N]
    mask = batch["padding_mask"]   # [B, N, N]
    
    B, C, N, _ = x_1.shape
    device = x_1.device
    
    # 1. Separate Continuous targets and Categorical targets
    x_1_continuous = x_1[:, :2, :, :] # [B, 2, N, N] (Presence, EdgeType)
    x_1_categorical = x_1[:, 2, :, :].long() # [B, N, N] (InputIndex classes)
    
    # 2. Sample random timesteps t ~ Uniform(0, 1)
    t = torch.rand((B,), device=device)
    t_expand = t.view(B, 1, 1, 1)
    
    # 3. Generate pure noise for the continuous channels
    x_0_continuous = torch.randn_like(x_1_continuous)
    
    # 4. Interpolate continuous matrix
    x_t_continuous = (1 - t_expand) * x_0_continuous + t_expand * x_1_continuous
    
    # For the categorical channel, the DiT now accepts 6 channels:
    # [Presence, EdgeType, Logit0, Logit1, Logit2, Logit3]
    # We pass pure noise for the 4 categorical channels so it learns to predict it from scratch.
    x_t_categorical_noise = torch.randn((B, 4, N, N), device=device)
    x_t = torch.cat([x_t_continuous, x_t_categorical_noise], dim=1) # [B, 6, N, N]
    
    # 5. Target Velocity for continuous channels
    target_velocity = x_1_continuous - x_0_continuous
    
    # 6. Forward Pass
    predictions = model(x_t, t, motifs, degrees=degrees) 
    
    # 7. Split predictions
    pred_velocity = predictions[:, :2, :, :] # [B, 2, N, N]
    pred_logits = predictions[:, 2:, :, :]   # [B, 4, N, N] (assuming 4 classes)
    
    # 8. Masked MSE Loss (Channels 0 & 1)
    mask_expanded_mse = mask.unsqueeze(1).expand(-1, 2, -1, -1)
    squared_error = (pred_velocity - target_velocity) ** 2
    masked_mse = squared_error * mask_expanded_mse
    loss_mse = masked_mse.sum() / max(mask_expanded_mse.sum(), 1)
    
    # 9. Masked CrossEntropy Loss (Channel 2)
    # pred_logits: [B, C, N, N], x_1_categorical: [B, N, N]
    loss_ce_unreduced = F.cross_entropy(pred_logits, x_1_categorical, reduction='none')
    masked_ce = loss_ce_unreduced * mask.float()
    loss_ce = masked_ce.sum() / max(mask.float().sum(), 1)
    
    # Total Hybrid Loss
    loss = loss_mse + loss_ce
    
    return loss
