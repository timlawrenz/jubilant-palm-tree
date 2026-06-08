import pytest
import torch
from src.models.validation import GraphValidator
from src.execution_engine.schema import MotifType

def get_motif_seq(*motifs):
    """Helper to create a motif_seq tensor for a given list of motif IDs."""
    return torch.tensor(list(motifs))

def test_motif_map():
    assert GraphValidator.MOTIF_MAP[1] == MotifType.BOUNDARY
    assert GraphValidator.MOTIF_MAP[2] == MotifType.SEQUENCE
    assert GraphValidator.MOTIF_MAP[3] == MotifType.CONDITION

def test_check_out_degree():
    valid_nodes = [0, 1, 2]
    # 0: BOUNDARY, 1: SEQUENCE, 2: CONDITION
    motif_seq = get_motif_seq(1, 2, 3) 
    
    # Valid: BOUNDARY has 1, SEQUENCE has 1, CONDITION has 2 distinct
    exec_out_valid = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 1), (1, 0)]
    }
    assert GraphValidator._check_out_degree(valid_nodes, motif_seq, exec_out_valid) is True

    # Invalid: BOUNDARY has > 1
    exec_out_invalid_bound = {
        0: [(0, 1), (1, 2)],
        1: [(0, 2)],
        2: [(0, 1), (1, 0)]
    }
    assert GraphValidator._check_out_degree(valid_nodes, motif_seq, exec_out_invalid_bound) is False

    # Invalid: CONDITION has same index twice
    exec_out_invalid_cond = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 1), (0, 0)] # indices are identical -> len(indices) != 2
    }
    assert GraphValidator._check_out_degree(valid_nodes, motif_seq, exec_out_invalid_cond) is False

def test_check_in_degree():
    valid_nodes = [0, 1]
    # 0: CONDITION, 1: STATE
    motif_seq = get_motif_seq(3, 5) 

    # Condition has 1 data_in, State has 1 data_in
    data_in_valid = {
        0: [(0, 1)],
        1: [(0, 0)]
    }
    # State has 1 exec_out, making it a "write"
    exec_out_write = {
        0: [(0, 1), (1, 1)],
        1: [(0, 0)]
    }
    assert GraphValidator._check_in_degree(valid_nodes, motif_seq, data_in_valid, exec_out_write) is True

    # Invalid: Condition lacks data_in
    data_in_invalid_cond = {
        0: [],
        1: [(0, 0)]
    }
    assert GraphValidator._check_in_degree(valid_nodes, motif_seq, data_in_invalid_cond, exec_out_write) is False

    # Invalid: State (write) lacks data_in
    data_in_invalid_state = {
        0: [(0, 1)],
        1: []
    }
    assert GraphValidator._check_in_degree(valid_nodes, motif_seq, data_in_invalid_state, exec_out_write) is False

def test_check_no_orphans():
    valid_nodes = [0, 1, 2]
    
    # Fully connected graph
    exec_out = {0: [(0, 1)], 1: [(0, 2)], 2: []}
    data_in = {0: [], 1: [], 2: []}
    assert GraphValidator._check_no_orphans(valid_nodes, exec_out, data_in) is True

    # Node 2 is orphaned
    exec_out_orphan = {0: [(0, 1)], 1: [], 2: []}
    data_in_orphan = {0: [], 1: [], 2: []}
    assert GraphValidator._check_no_orphans(valid_nodes, exec_out_orphan, data_in_orphan) is False

    # Two disconnected subgraphs (0->1, 2 alone)
    data_in_sub = {0: [], 1: [(0, 0)], 2: []}
    assert GraphValidator._check_no_orphans(valid_nodes, {0:[], 1:[], 2:[]}, data_in_sub) is False

def test_check_acyclic_data():
    valid_nodes = [0, 1, 2]
    
    # Acyclic: 0 <- 1 <- 2
    data_in_acyclic = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: []
    }
    assert GraphValidator._check_acyclic_data(valid_nodes, data_in_acyclic) is True

    # Cyclic: 0 <- 1 <- 2 <- 0
    data_in_cyclic = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 0)]
    }
    assert GraphValidator._check_acyclic_data(valid_nodes, data_in_cyclic) is False

    # Self-loop: 0 <- 0
    data_in_self = {
        0: [(0, 0)],
        1: [],
        2: []
    }
    assert GraphValidator._check_acyclic_data(valid_nodes, data_in_self) is False

def test_check_terminal_sink():
    valid_nodes = [0, 1, 2]
    motif_seq = get_motif_seq(2, 2, 1) # Motif doesn't deeply matter for this rule, but passed
    
    # Path with sink: 0 -> 1 -> 2 (2 has no exec out)
    exec_out_sink = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: []
    }
    assert GraphValidator._check_terminal_sink(valid_nodes, motif_seq, exec_out_sink) is True

    # Infinite loop without escape: 0 -> 1 -> 2 -> 0
    exec_out_inf = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [(0, 0)]
    }
    assert GraphValidator._check_terminal_sink(valid_nodes, motif_seq, exec_out_inf) is False

def test_evaluate_batch_perfect_graph():
    # Construct a simple valid graph: BOUNDARY -> SEQUENCE -> BOUNDARY(exit)
    # Motifs: 1, 2, 1, 0(padding)
    motifs = torch.tensor([[1, 2, 1, 0]])
    B, N = 1, 4
    
    adj = torch.zeros(B, 3, N, N)
    
    # Edge 0: 0 -> 1 (Exec, index 0)
    adj[0, 0, 0, 1] = 1 # presence
    adj[0, 1, 0, 1] = 0 # type = Exec
    adj[0, 2, 0, 1] = 0 # index = 0
    
    # Edge 1: 1 -> 2 (Exec, index 0)
    adj[0, 0, 1, 2] = 1 # presence
    adj[0, 1, 1, 2] = 0 # type = Exec
    adj[0, 2, 1, 2] = 0 # index = 0

    results = GraphValidator.evaluate_batch(adj, motifs)
    
    assert results["out_degree_pass"] == 100.0
    assert results["in_degree_pass"] == 100.0
    assert results["no_orphan_pass"] == 100.0
    assert results["acyclic_data_pass"] == 100.0
    assert results["terminal_sink_pass"] == 100.0
    assert results["global_spine_pass"] == 100.0
    assert results["perfect_graphs"] == 100.0

def test_check_global_spine():
    valid_nodes = [0, 1, 2, 3]
    # 0: Boundary, 1: Sequence, 2: Boundary, 3: Sequence
    motif_seq = get_motif_seq(1, 2, 1, 2)
    
    # Valid: 0 -> 1 -> 2 (3 is not in exec plane)
    exec_out_valid = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [],
        3: []
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_valid) is True

    # Invalid: Two entries (0 and 2 are boundaries with 0 in-edges)
    exec_out_two_entries = {
        0: [(0, 1)],
        1: [],
        2: [(0, 3)],
        3: []
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_two_entries) is False

    # Invalid: Valid entry, but disconnected loop (3 -> 3)
    exec_out_disconnected = {
        0: [(0, 1)],
        1: [(0, 2)],
        2: [],
        3: [(0, 3)]
    }
    assert GraphValidator._check_global_spine(valid_nodes, motif_seq, exec_out_disconnected) is False

def test_evaluate_batch_empty_graph_is_skipped():
    motifs = torch.tensor([[0, 0, 0, 0]])
    adj = torch.zeros(1, 3, 4, 4)
    results = GraphValidator.evaluate_batch(adj, motifs)
    assert results["perfect_graphs"] == 0.0 # B=1, but skipped, so 0/1 = 0%
