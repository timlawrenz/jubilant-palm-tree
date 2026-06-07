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
