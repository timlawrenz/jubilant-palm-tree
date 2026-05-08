"""
Differentiable structural loss for RLAIF.

Instead of REINFORCE (sample + scalar reward), we compute differentiable
versions of the 5 structural constraints directly on the continuous ODE output.
This gives dense, per-element gradient signal.

The loss operates on the soft (pre-threshold) model output, so gradients
flow directly to the model parameters through the last ODE step.
"""

import torch
import torch.nn.functional as F


# Expected execution out-degree per motif type (from GraphValidator._check_out_degree)
# Motif IDs: 0=pad, 1=BOUNDARY, 2=SEQUENCE, 3=CONDITION, 4=LOOP, 5=STATE, 6=MESSAGE
# BOUNDARY: 0 or 1 (entry=1, exit=0), we target 0.5 as compromise
# SEQUENCE/STATE/MESSAGE: 0 or 1 (mostly 1)
# CONDITION/LOOP: 0 or 2 (mostly 2 if functional)
EXPECTED_EXEC_OUT = {
    0: 0.0,   # padding — no edges
    1: 0.5,   # BOUNDARY — 0 or 1 (soft target)
    2: 1.0,   # SEQUENCE — exactly 1
    3: 2.0,   # CONDITION — exactly 2
    4: 2.0,   # LOOP — exactly 2
    5: 1.0,   # STATE — 0 or 1 (target 1 if in exec path)
    6: 1.0,   # MESSAGE — 0 or 1 (target 1 if in exec path)
}


def compute_structural_loss(x_final: torch.Tensor, motifs: torch.Tensor) -> dict:
    """
    Compute differentiable structural losses on the continuous ODE output.

    Args:
        x_final: [B, 6, N, N] — continuous model output (not discretized).
            Channel 0: presence logit
            Channel 1: edge type logit (0=exec, 1=data)
            Channels 2-5: input_index logits
        motifs: [B, N] — motif type IDs (0 = padding).

    Returns:
        Dictionary with individual loss components and total loss (all [B]).
    """
    B, C, N, _ = x_final.shape
    device = x_final.device

    # Padding mask
    padding_mask = (motifs != 0).float()  # [B, N]
    n_valid = padding_mask.sum(dim=1).clamp(min=1)  # [B]
    mask_2d = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)  # [B, N, N]

    # Soft edge probabilities
    soft_presence = torch.sigmoid(x_final[:, 0])  # [B, N, N]
    soft_data_type = torch.sigmoid(x_final[:, 1])  # [B, N, N] prob of being DATA
    soft_exec_type = 1.0 - soft_data_type  # prob of being EXEC

    # Effective soft edge weights (presence × type)
    soft_exec = soft_presence * soft_exec_type * mask_2d  # [B, N, N]
    soft_data = soft_presence * soft_data_type * mask_2d  # [B, N, N]

    # === 1. EXECUTION OUT-DEGREE LOSS ===
    # Each node's exec out-degree should match its motif type's expected value
    exec_out_degree = soft_exec.sum(dim=2)  # [B, N] — sum over targets

    # Build target out-degree from motif types
    target_out = torch.zeros_like(exec_out_degree)
    for motif_id, expected in EXPECTED_EXEC_OUT.items():
        target_out[motifs == motif_id] = expected

    out_degree_loss = ((exec_out_degree - target_out) ** 2 * padding_mask).sum(dim=1) / n_valid

    # === 2. EDGE SHARPNESS LOSS ===
    # Only penalize ambiguity on edges that EXIST (presence > 0.5).
    # This prevents the model from collapsing to "no edges" to minimize sharpness.
    present_edges = (soft_presence > 0.5).float() * mask_2d
    sharpness = soft_presence * (1.0 - soft_presence) * present_edges
    sharpness_loss = sharpness.sum(dim=(1, 2)) / present_edges.sum(dim=(1, 2)).clamp(min=1)

    # Same for edge type — only on present edges
    type_sharpness = soft_data_type * soft_exec_type * present_edges * soft_presence
    type_sharpness_loss = type_sharpness.sum(dim=(1, 2)) / present_edges.sum(dim=(1, 2)).clamp(min=1)

    # === 3. TERMINAL SINK LOSS ===
    # At least one valid node should have near-zero exec out-degree (= exit node)
    # Use soft-min: the minimum exec out-degree among valid nodes should be ≈ 0
    # Replace padding with large value so it doesn't affect min
    exec_out_masked = exec_out_degree + (1 - padding_mask) * 100.0
    min_exec_out = exec_out_masked.min(dim=1).values  # [B]
    terminal_loss = min_exec_out ** 2  # Push min towards 0

    # === 4. NO ORPHAN LOSS ===
    # Each valid node should have total degree > 0 (at least one edge)
    total_out = (soft_exec + soft_data).sum(dim=2)  # [B, N]
    total_in = (soft_exec + soft_data).sum(dim=1)  # [B, N]
    total_degree = total_out + total_in  # [B, N]
    # Penalize nodes with degree < 1 using soft hinge
    orphan_deficit = F.relu(1.0 - total_degree) * padding_mask
    orphan_loss = orphan_deficit.sum(dim=1) / n_valid

    # === 5. DATA IN-DEGREE LOSS ===
    # CONDITION/LOOP (3,4) should have exactly 1 data input
    data_in_degree = soft_data.sum(dim=1)  # [B, N] — sum over sources
    cond_loop_mask = ((motifs == 3) | (motifs == 4)).float()
    cond_loop_in_loss = ((data_in_degree - 1.0) ** 2 * cond_loop_mask).sum(dim=1) / cond_loop_mask.sum(dim=1).clamp(min=1)

    # === 6. EDGE DENSITY LOSS ===
    # Prevent collapse to empty graphs. The expected edge count is derived from motifs:
    # each valid non-exit node should produce ~1 exec edge, conditions/loops ~2.
    # Minimum target: roughly 1 edge per valid node.
    total_soft_edges = (soft_presence * mask_2d).sum(dim=(1, 2))  # [B]
    target_edges = n_valid  # At least 1 edge per valid node
    density_deficit = F.relu(target_edges - total_soft_edges)  # Only penalize too few
    density_loss = (density_deficit / n_valid) ** 2  # Normalized squared deficit

    # === COMBINE ===
    total = (
        1.0 * out_degree_loss
        + 0.1 * sharpness_loss
        + 0.1 * type_sharpness_loss
        + 0.5 * terminal_loss
        + 1.0 * orphan_loss
        + 0.5 * cond_loop_in_loss
        + 2.0 * density_loss
    )

    return {
        "total": total,           # [B]
        "out_degree": out_degree_loss,
        "sharpness": sharpness_loss,
        "type_sharpness": type_sharpness_loss,
        "terminal": terminal_loss,
        "orphan": orphan_loss,
        "data_in": cond_loop_in_loss,
        "density": density_loss,
    }
