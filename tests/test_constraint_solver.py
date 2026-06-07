import torch
import pytest
from src.models.constraint_solver import ConstraintSolver

def test_find_and_repair_cycle():
    # 3 nodes forming a cycle: 0 -> 1 -> 2 -> 0
    adj = torch.zeros(3, 3, 3)
    # Set presence
    adj[0, 0, 1] = 1
    adj[0, 1, 2] = 1
    adj[0, 2, 0] = 1
    # Set edge_type to 1 (Data)
    adj[1, 0, 1] = 1
    adj[1, 1, 2] = 1
    adj[1, 2, 0] = 1
    
    valid_nodes = [0, 1, 2]
    
    # 1->2 is the weakest edge (0.3)
    soft_presence = torch.zeros(3, 3)
    soft_presence[0, 1] = 0.9
    soft_presence[1, 2] = 0.3
    soft_presence[2, 0] = 0.8
    
    repaired_adj = ConstraintSolver._repair_acyclicity(adj, soft_presence, valid_nodes)
    
    # The weakest edge should be removed
    assert repaired_adj[0, 1, 2].item() == 0
    
    # The strong edges should remain
    assert repaired_adj[0, 0, 1].item() == 1
    assert repaired_adj[0, 2, 0].item() == 1
    
    # Verify no cycles remain
    assert ConstraintSolver._find_cycle(repaired_adj, valid_nodes) is None

def test_repair_exec_out_degree():
    # 4 nodes. Node 0 is SEQUENCE (motif=2). It should have MAX 1 exec out.
    adj = torch.zeros(3, 4, 4)
    # Node 0 has 3 outgoing exec edges to 1, 2, 3
    adj[0, 0, 1] = 1; adj[1, 0, 1] = 0  # Exec
    adj[0, 0, 2] = 1; adj[1, 0, 2] = 0  # Exec
    adj[0, 0, 3] = 1; adj[1, 0, 3] = 0  # Exec
    
    valid_nodes = [0, 1, 2, 3]
    motifs = [2, 1, 1, 1] # 0 is Sequence, others Boundary
    
    # 0->2 is the strongest edge (0.95)
    soft_presence = torch.zeros(4, 4)
    soft_presence[0, 1] = 0.8
    soft_presence[0, 2] = 0.95
    soft_presence[0, 3] = 0.6
    
    repaired_adj = ConstraintSolver._repair_exec_out_degree(adj, soft_presence, valid_nodes, motifs)
    
    # The two weaker edges should be pruned
    assert repaired_adj[0, 0, 1].item() == 0
    assert repaired_adj[0, 0, 3].item() == 0
    # The strongest edge should remain
    assert repaired_adj[0, 0, 2].item() == 1

def test_repair_data_in_degree():
    # 4 nodes. Node 1 is CONDITION (motif=3). It NEEDS >=1 data in.
    adj = torch.zeros(3, 4, 4)
    # Node 1 currently has 0 inputs
    
    valid_nodes = [0, 1, 2, 3]
    motifs = [1, 3, 1, 1] # 1 is Condition
    
    soft_presence = torch.zeros(4, 4)
    soft_data_type = torch.zeros(4, 4)
    
    # Candidate 0->1: presence 0.4, data 0.9 => 0.36
    soft_presence[0, 1] = 0.4; soft_data_type[0, 1] = 0.9
    # Candidate 2->1: presence 0.7, data 0.8 => 0.56 (Strongest)
    soft_presence[2, 1] = 0.7; soft_data_type[2, 1] = 0.8
    # Candidate 3->1: presence 0.6, data 0.2 => 0.12
    soft_presence[3, 1] = 0.6; soft_data_type[3, 1] = 0.2
    
    repaired_adj = ConstraintSolver._repair_data_in_degree(adj, soft_presence, soft_data_type, valid_nodes, motifs)
    
    # The strongest candidate (2->1) should be forced to 1
    assert repaired_adj[0, 2, 1].item() == 1
    assert repaired_adj[1, 2, 1].item() == 1
    
    # The others remain 0
    assert repaired_adj[0, 0, 1].item() == 0
    assert repaired_adj[0, 3, 1].item() == 0

def test_repair_index_collisions():
    # Node 0 has 2 outgoing exec edges to 1 and 2. Both currently claim index 0.
    adj = torch.zeros(3, 3, 3)
    adj[0, 0, 1] = 1; adj[1, 0, 1] = 0; adj[2, 0, 1] = 0
    adj[0, 0, 2] = 1; adj[1, 0, 2] = 0; adj[2, 0, 2] = 0
    
    valid_nodes = [0, 1, 2]
    
    # 0->1 is the stronger edge (0.9 vs 0.6)
    soft_presence = torch.zeros(3, 3)
    soft_presence[0, 1] = 0.9
    soft_presence[0, 2] = 0.6
    
    # Provide categorical logits (4 channels)
    index_logits = torch.zeros(4, 3, 3)
    # Edge 0->1 strongly prefers index 0
    index_logits[:, 0, 1] = torch.tensor([5.0, 1.0, 0.0, 0.0])
    # Edge 0->2 prefers index 0, but its second choice is index 1
    index_logits[:, 0, 2] = torch.tensor([4.0, 3.0, 0.0, 0.0])
    
    repaired_adj = ConstraintSolver._repair_index_collisions(adj, index_logits, soft_presence, valid_nodes)
    
    # The stronger edge (0->1) keeps index 0
    assert repaired_adj[2, 0, 1].item() == 0
    # The weaker edge (0->2) should be bumped to its second choice, index 1
    assert repaired_adj[2, 0, 2].item() == 1