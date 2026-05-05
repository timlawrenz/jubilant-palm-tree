"""
Per-graph reward function for RLAIF.

Wraps GraphValidator to return per-graph scalar rewards (not batch percentages).
Implements asymmetric weighting with a 2.5× jackpot for perfect graphs.
"""

import torch
from typing import Dict, List
from src.models.validation import GraphValidator


# Reward weights per law
REWARD_WEIGHTS = {
    "out_degree": {"pass": 0.4, "fail": -0.2},
    "in_degree": {"pass": 0.2, "fail": -0.1},
    "no_orphan": {"pass": 0.2, "fail": -0.1},
    "acyclic_data": {"pass": 0.2, "fail": -0.1},
    "terminal_sink": {"pass": 0.4, "fail": -0.4},
}

JACKPOT_MULTIPLIER = 2.5


def compute_reward(discrete_adjacency: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
    """
    Computes per-graph scalar rewards based on the 5 Laws of Physics.

    Args:
        discrete_adjacency: [B, 3, N, N] — discrete adjacency from naive_discretize.
        motifs: [B, N] — motif type IDs.

    Returns:
        rewards: [B] tensor of scalar rewards per graph.
    """
    B = motifs.shape[0]
    rewards = torch.zeros(B, device=motifs.device)

    for i in range(B):
        adj = discrete_adjacency[i]  # [3, N, N]
        motif_seq = motifs[i]  # [N]

        reward = _score_single_graph(adj, motif_seq)
        rewards[i] = reward

    return rewards


def _score_single_graph(adj: torch.Tensor, motif_seq: torch.Tensor) -> float:
    """Score a single graph against all 5 laws."""
    from src.execution_engine.schema import MotifType

    MOTIF_MAP = GraphValidator.MOTIF_MAP

    # Extract valid nodes
    valid_nodes = [n for n, m in enumerate(motif_seq.tolist()) if m != 0]
    if not valid_nodes:
        return -1.0  # Empty graph penalty

    # Build edge structures (same as GraphValidator)
    presence = adj[0].bool().tolist()
    edge_type = adj[1].int().tolist()
    input_index = adj[2].int().tolist()

    exec_out = {n: [] for n in valid_nodes}
    data_in = {n: [] for n in valid_nodes}

    for src in valid_nodes:
        for tgt in valid_nodes:
            if presence[src][tgt]:
                e_type = edge_type[src][tgt]
                idx = input_index[src][tgt]
                if e_type == 0:  # EXECUTION
                    exec_out[src].append((idx, tgt))
                else:  # DATA
                    data_in[tgt].append((idx, src))

    # Evaluate each law
    r1 = GraphValidator._check_out_degree(valid_nodes, motif_seq, exec_out)
    r2 = GraphValidator._check_in_degree(valid_nodes, motif_seq, data_in, exec_out)
    r3 = GraphValidator._check_no_orphans(valid_nodes, exec_out, data_in)
    # _check_acyclic_data returns True if cycle EXISTS (= fail), so we invert
    r4 = not GraphValidator._check_acyclic_data(valid_nodes, data_in)
    r5 = GraphValidator._check_terminal_sink(valid_nodes, motif_seq, exec_out)

    # Compute weighted reward
    reward = 0.0
    laws = [
        ("out_degree", r1),
        ("in_degree", r2),
        ("no_orphan", r3),
        ("acyclic_data", r4),
        ("terminal_sink", r5),
    ]

    all_pass = True
    for law_name, passed in laws:
        if passed:
            reward += REWARD_WEIGHTS[law_name]["pass"]
        else:
            reward += REWARD_WEIGHTS[law_name]["fail"]
            all_pass = False

    # Jackpot multiplier for perfect graphs
    if all_pass:
        reward *= JACKPOT_MULTIPLIER

    return reward
