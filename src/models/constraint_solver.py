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
        
        B = discrete.shape[0]
        
        # Apply repairs per graph
        for b in range(B):
            valid_nodes = [n for n, m in enumerate(motifs[b].tolist()) if m != 0]
            if not valid_nodes:
                continue
                
            discrete[b] = cls._repair_acyclicity(discrete[b], soft_presence[b], valid_nodes)
            
        return discrete

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
