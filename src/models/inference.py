import torch
from src.models.model import NeuralUniversalMachineDiT
from src.execution_engine.schema import MotifType

class JudicialConstraintSolver:
    """
    Acts as the 'Judicial Branch' of the Neural Universal Machine.
    Takes the continuous probability matrix from the DiT (Executive Branch) and 
    deterministically snaps it into a mathematically legal discrete DAG by enforcing 
    arity constraints and resolving index collisions.
    """
    MOTIF_MAP = {
        1: MotifType.BOUNDARY,
        2: MotifType.SEQUENCE,
        3: MotifType.CONDITION,
        4: MotifType.LOOP,
        5: MotifType.STATE,
        6: MotifType.MESSAGE
    }

    @classmethod
    def apply_constraints(cls, continuous_matrix: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        """
        continuous_matrix: [B, 6, N, N]
        motifs: [B, N]
        Returns: [B, 3, N, N] discrete adjacency matrix
        """
        B, C, N, _ = continuous_matrix.shape
        device = continuous_matrix.device
        
        # 1. Squash Presence and Type to probabilities [0, 1]
        presence_probs = torch.sigmoid(continuous_matrix[:, 0, :, :])
        type_probs = torch.sigmoid(continuous_matrix[:, 1, :, :])
        
        # 2. Extract index logits
        index_logits = continuous_matrix[:, 2:, :, :] # [B, 4, N, N]
        index_discrete = torch.argmax(index_logits, dim=1).int() # [B, N, N]
        
        # Initialize the output discrete tensor with zeros
        discrete_presence = torch.zeros_like(presence_probs, dtype=torch.int32)
        discrete_type = (type_probs > 0.5).int() # We still naively threshold Type, as it's just Exec vs Data
        
        # 3. Apply Motif-Specific Arity Laws row-by-row
        for b in range(B):
            for i in range(N):
                motif_id = motifs[b, i].item()
                if motif_id == 0:
                    continue # Skip padding
                    
                motif = cls.MOTIF_MAP.get(motif_id)
                row_probs = presence_probs[b, i, :]
                
                # Rule 1: Out-Degree (Execution Edges)
                # How many outgoing edges is this motif legally allowed to have?
                allowed_edges = 0
                if motif in (MotifType.SEQUENCE, MotifType.STATE, MotifType.MESSAGE):
                    allowed_edges = 1
                elif motif in (MotifType.CONDITION, MotifType.LOOP):
                    allowed_edges = 2
                    
                # To enforce the law, we take the Top-K highest probabilities in the row
                if allowed_edges > 0:
                    # We only want to select edges where the DiT actually intended an EXECUTION type (type_probs < 0.5)
                    # We mask out Data edges so we don't accidentally snap a Data edge into an Execution slot
                    exec_mask = (type_probs[b, i, :] < 0.5).float()
                    masked_probs = row_probs * exec_mask
                    
                    # Get the Top-K indices
                    top_k_vals, top_k_indices = torch.topk(masked_probs, allowed_edges)
                    
                    for k in range(allowed_edges):
                        # Only snap to 1 if the probability isn't completely dead noise
                        if top_k_vals[k] > 0.1:
                            target_idx = top_k_indices[k]
                            discrete_presence[b, i, target_idx] = 1
                            discrete_type[b, i, target_idx] = 0 # Force to Execution
                            
                            # If it's a Condition/Loop, force the indices to be distinct (0 and 1)
                            if allowed_edges == 2:
                                index_discrete[b, i, target_idx] = k
                                
                # Rule 2: In-Degree (Data Edges)
                # For data edges, the arity rule is based on INCOMING edges (columns), not rows.
                # However, Data dependencies are complex because a node can provide data to infinite targets.
                # We will natively threshold the data edges, and resolve collisions on the column pass next.
                data_mask = (type_probs[b, i, :] >= 0.5).float()
                data_edges = (row_probs * data_mask) > 0.5
                discrete_presence[b, i, :] = discrete_presence[b, i, :] | data_edges.int()
                
            # 4. Resolve Data Index Collisions (Column Pass)
            for j in range(N):
                motif_id = motifs[b, j].item()
                if motif_id == 0: continue
                motif = cls.MOTIF_MAP.get(motif_id)
                
                # Find all incoming DATA edges to node j
                incoming_data_sources = [i for i in range(N) if discrete_presence[b, i, j] == 1 and discrete_type[b, i, j] == 1]
                
                if motif in (MotifType.CONDITION, MotifType.LOOP, MotifType.STATE):
                    # These can only accept exactly 1 incoming data edge.
                    if len(incoming_data_sources) > 1:
                        # Collision! Keep the one with the highest Presence probability, kill the rest
                        best_source = max(incoming_data_sources, key=lambda s: presence_probs[b, s, j].item())
                        for s in incoming_data_sources:
                            if s != best_source:
                                discrete_presence[b, s, j] = 0 # Kill the illegal ghost edge
                                
                elif motif == MotifType.MESSAGE:
                    # Messages can accept multiple arguments, but they must have distinct input_indices!
                    claimed_indices = {}
                    for s in incoming_data_sources:
                        idx_val = index_discrete[b, s, j].item()
                        if idx_val in claimed_indices:
                            # Collision! Two arguments claim the same slot.
                            # The edge with the higher categorical logit for that slot wins.
                            competitor_s = claimed_indices[idx_val]
                            my_logit = index_logits[b, idx_val, s, j].item()
                            comp_logit = index_logits[b, idx_val, competitor_s, j].item()
                            
                            if my_logit > comp_logit:
                                # I win. Kill the competitor.
                                discrete_presence[b, competitor_s, j] = 0
                                claimed_indices[idx_val] = s
                            else:
                                # I lose. Kill me.
                                discrete_presence[b, s, j] = 0
                        else:
                            claimed_indices[idx_val] = s

        # 5. Clean up padding
        padding_mask = (motifs != 0).float()
        mask_2d = padding_mask.unsqueeze(2) * padding_mask.unsqueeze(1)
        
        discrete_adjacency = torch.cat([
            discrete_presence.unsqueeze(1), 
            discrete_type.unsqueeze(1), 
            index_discrete.unsqueeze(1)
        ], dim=1)
        
        discrete_adjacency = discrete_adjacency * mask_2d.unsqueeze(1).int()
        
        return discrete_adjacency

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
    
    # 3. Discretize the Continuous Output using the Judicial Constraint Solver
    discrete_adjacency = JudicialConstraintSolver.apply_constraints(x, motifs)
    
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
