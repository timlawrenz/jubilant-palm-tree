import torch
from src.rlaif.naive_discretizer import naive_discretize

class ConstraintSolver:
    @classmethod
    def discretize_and_repair(cls, continuous_matrix: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
        """
        Discretizes the continuous heatmap and applies deterministic topological repairs.
        """
        # 1. Start with the naive thresholding
        discrete = naive_discretize(continuous_matrix, motifs)
        
        # 2. Extract soft probabilities to resolve conflicts
        soft_presence = torch.sigmoid(continuous_matrix[:, 0]) # [B, N, N]
        soft_data_type = torch.sigmoid(continuous_matrix[:, 1]) # [B, N, N]
        index_logits = continuous_matrix[:, 2:6] # [B, 4, N, N]
        
        B = discrete.shape[0]
        
        # Apply repairs per graph
        for b in range(B):
            valid_nodes = [n for n, m in enumerate(motifs[b].tolist()) if m != 0]
            if not valid_nodes:
                continue
                
            discrete[b] = cls._repair_exec_out_degree(discrete[b], soft_presence[b], valid_nodes, motifs[b].tolist())
            discrete[b] = cls._repair_data_in_degree(discrete[b], soft_presence[b], soft_data_type[b], valid_nodes, motifs[b].tolist())
            discrete[b] = cls._repair_index_collisions(discrete[b], index_logits[b], soft_presence[b], valid_nodes)
            discrete[b] = cls._repair_acyclicity(discrete[b], soft_presence[b], valid_nodes)
            
        return discrete

    @classmethod
    def _repair_exec_out_degree(cls, adj: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list, motifs: list) -> torch.Tensor:
        """
        Enforces execution out-degree limits based on Motif type.
        adj is [3, N, N]. Channel 0 = presence, Channel 1 = type (0=Exec).
        """
        from src.rlaif.structural_loss import EXPECTED_EXEC_OUT
        
        # We only look at execution edges (type == 0)
        is_exec = (adj[1] == 0) & (adj[0] == 1)
        
        for u in valid_nodes:
            motif_id = motifs[u]
            target_out = EXPECTED_EXEC_OUT.get(motif_id, 0.0)
            
            # For this MVP, we enforce a strict max cap based on the motif.
            # Condition/Loop: max 2. Sequence/State/Message: max 1. Boundary: max 1.
            max_allowed = int(target_out)
            if max_allowed == 0 and motif_id != 1: 
                # If target is 0.0 but it's not a boundary, it's padded/invalid, skip
                continue
            if motif_id == 1:
                max_allowed = 1 # Boundary can be 0 or 1
                
            current_out_edges = [v for v in valid_nodes if is_exec[u, v]]
            
            # If over-degree, keep the top-K highest probability edges and prune the rest
            if len(current_out_edges) > max_allowed:
                # Sort by soft probability descending
                current_out_edges.sort(key=lambda v: soft_presence[u, v].item(), reverse=True)
                
                # Prune edges beyond the allowed max
                for v_to_prune in current_out_edges[max_allowed:]:
                    adj[0, u, v_to_prune] = 0 # Drop presence
                    
        return adj

    @classmethod
    def _repair_data_in_degree(cls, adj: torch.Tensor, soft_presence: torch.Tensor, soft_data_type: torch.Tensor, valid_nodes: list, motifs: list) -> torch.Tensor:
        """
        Enforces data in-degree minimums.
        Condition/Loop/State-Writes need >= 1 data input.
        If they have 0, we find the highest probability incoming data edge in the continuous map and force it to 1.
        """
        is_data = (adj[1] == 1) & (adj[0] == 1)
        is_exec = (adj[1] == 0) & (adj[0] == 1)
        
        for v in valid_nodes:
            motif_id = motifs[v]
            
            # Check if this node REQUIRES a data input
            requires_data = False
            if motif_id in (3, 4): # Condition, Loop
                requires_data = True
            elif motif_id == 5: # State (Write)
                # It's a write if it has any outgoing exec edges
                if any(is_exec[v, tgt] for tgt in valid_nodes):
                    requires_data = True
                    
            if not requires_data:
                continue
                
            current_in_edges = [u for u in valid_nodes if is_data[u, v]]
            
            # If under-degree (0 inputs), find the strongest candidate and force it
            if len(current_in_edges) == 0:
                best_u = None
                best_prob = -1.0
                
                for u in valid_nodes:
                    if u == v: continue # Prevent self-loops trivially
                    # The probability of it being a data edge is presence * data_type
                    prob = (soft_presence[u, v] * soft_data_type[u, v]).item()
                    if prob > best_prob:
                        best_prob = prob
                        best_u = u
                        
                if best_u is not None:
                    # Force the edge into existence
                    adj[0, best_u, v] = 1 # Presence
                    adj[1, best_u, v] = 1 # Type = Data
                    # Let the argmax index logits stay as they are in adj[2]
                    
        return adj

    @classmethod
    def _repair_index_collisions(cls, adj: torch.Tensor, index_logits: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list) -> torch.Tensor:
        """
        Resolves input_index collisions for execution and data edges using categorical logits.
        adj is [3, N, N]. Channel 0 = presence, Channel 1 = type, Channel 2 = input_index.
        index_logits is [4, N, N].
        """
        presence = adj[0].bool()
        edge_type = adj[1].int()
        input_index = adj[2].int()

        is_exec = presence & (edge_type == 0)
        is_data = presence & (edge_type == 1)

        def resolve_collisions(edges, is_source_grouped=True):
            # Group edges by source (for exec out) or by target (for data in)
            groups = {n: [] for n in valid_nodes}
            for u in valid_nodes:
                for v in valid_nodes:
                    if edges[u, v]:
                        key = u if is_source_grouped else v
                        groups[key].append((u, v))

            for key, edge_list in groups.items():
                if len(edge_list) <= 1:
                    continue

                # Check for collisions
                used_indices = {}
                for u, v in edge_list:
                    idx = input_index[u, v].item()
                    if idx not in used_indices:
                        used_indices[idx] = []
                    used_indices[idx].append((u, v))

                for idx, colliding_edges in used_indices.items():
                    if len(colliding_edges) <= 1:
                        continue

                    # Collision detected! Sort by presence probability descending
                    colliding_edges.sort(key=lambda e: soft_presence[e[0], e[1]].item(), reverse=True)
                    
                    # The strongest edge gets to keep the index
                    winner = colliding_edges[0]
                    losers = colliding_edges[1:]

                    # Reassign losers using their next best logits
                    occupied = set(input_index[u, v].item() for u, v in edge_list)
                    
                    for lu, lv in losers:
                        logits = index_logits[:, lu, lv]
                        # Sort indices by logit score descending
                        sorted_indices = torch.argsort(logits, descending=True).tolist()
                        
                        # Find the highest scored index that isn't occupied
                        new_idx = None
                        for candidate_idx in sorted_indices:
                            if candidate_idx not in occupied:
                                new_idx = candidate_idx
                                break
                                
                        # If all 4 slots are occupied (rare), force an arbitrary unused one
                        if new_idx is None:
                            new_idx = max(occupied) + 1 if occupied else 0
                            
                        input_index[lu, lv] = new_idx
                        occupied.add(new_idx)

        # Apply to Execution Out-Edges (grouped by source node)
        resolve_collisions(is_exec, is_source_grouped=True)
        # Apply to Data In-Edges (grouped by target node)
        resolve_collisions(is_data, is_source_grouped=False)

        adj[2] = input_index
        return adj

    @classmethod
    def _repair_acyclicity(cls, adj: torch.Tensor, soft_presence: torch.Tensor, valid_nodes: list) -> torch.Tensor:
        """
        Detects data-plane cycles and breaks them by removing the weakest edge.
        adj is [3, N, N] where channel 0 is presence, channel 1 is edge_type (1=Data).
        """
        while True:
            cycle = cls._find_cycle(adj, valid_nodes)
            if not cycle:
                break # DAG achieved!
                
            # Find the weakest link in the cycle
            min_prob = float('inf')
            weakest_edge = None
            
            for u, v in cycle:
                prob = soft_presence[u, v].item()
                if prob < min_prob:
                    min_prob = prob
                    weakest_edge = (u, v)
                    
            # Break the cycle
            u, v = weakest_edge
            adj[0, u, v] = 0 # Remove presence
            
        return adj

    @classmethod
    def _find_cycle(cls, adj: torch.Tensor, valid_nodes: list) -> list | None:
        """Returns a list of edges (u, v) forming a cycle, or None."""
        presence = adj[0].bool().tolist()
        edge_type = adj[1].int().tolist()
        
        # Build adjacency list for DATA edges only (edge_type == 1)
        data_out = {n: [] for n in valid_nodes}
        for u in valid_nodes:
            for v in valid_nodes:
                if presence[u][v] and edge_type[u][v] == 1:
                    data_out[u].append(v)
                    
        visited = set()
        rec_stack = set()
        parent = {}
        
        def dfs(u):
            visited.add(u)
            rec_stack.add(u)
            
            for v in data_out[u]:
                if v not in visited:
                    parent[v] = u
                    cycle = dfs(v)
                    if cycle: return cycle
                elif v in rec_stack:
                    # Cycle detected. Trace back to build the path.
                    path = [(u, v)]
                    curr = u
                    while curr != v:
                        p = parent.get(curr)
                        if p is None:
                            break
                        path.append((p, curr))
                        curr = p
                    return path[::-1]
                    
            rec_stack.remove(u)
            return None
            
        for n in valid_nodes:
            if n not in visited:
                cycle = dfs(n)
                if cycle: return cycle
                
        return None
