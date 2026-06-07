import torch
from typing import List
from src.execution_engine.schema import ExecutionGraph, Node, Edge, MotifType, EdgeType

class DummySemanticCustodian:
    """
    Acts as a stand-in for the Semantic LLM.
    Translates raw tensor topologies into ExecutionGraphs with safe, dummy literals.
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
    def tensor_to_graph(cls, adj: torch.Tensor, motifs: torch.Tensor) -> ExecutionGraph:
        """
        adj: [3, N, N] discrete adjacency
        motifs: [N] 1D tensor
        """
        valid_nodes = [n for n, m in enumerate(motifs.tolist()) if m != 0]
        
        presence = adj[0].bool().tolist()
        edge_type = adj[1].int().tolist()
        input_idx = adj[2].int().tolist()
        
        nodes: List[Node] = []
        edges: List[Edge] = []
        literal_pool = {
            0: "dummy_var",
            1: "safe_op",
            2: 1 # Safe constant
        }
        
        for n in valid_nodes:
            m_id = int(motifs[n].item())
            m_type = cls.MOTIF_MAP[m_id]
            
            ptr = None
            if m_type == MotifType.STATE:
                ptr = 0 # Points to "dummy_var"
            elif m_type == MotifType.MESSAGE:
                # Check if it has incoming data. If not, it's a constant.
                has_in = any(presence[src][n] and edge_type[src][n] == 1 for src in valid_nodes)
                ptr = 1 if has_in else 2 # "safe_op" or constant 1
                
            nodes.append(Node(node_id=n, motif=m_type, literal_pointer=ptr))
            
            # Extract edges originating from n
            for tgt in valid_nodes:
                if presence[n][tgt]:
                    edges.append(Edge(
                        source_node=n,
                        target_node=tgt,
                        edge_type=EdgeType(edge_type[n][tgt]),
                        input_index=input_idx[n][tgt]
                    ))
                    
        return ExecutionGraph(nodes=nodes, edges=edges, literal_pool=literal_pool)