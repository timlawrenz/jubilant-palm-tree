"""Tests for the RLAIF reward function."""

import torch
import pytest
from src.rlaif.reward import compute_reward, _score_single_graph, JACKPOT_MULTIPLIER


class TestRewardComputation:
    """Test the per-graph reward function."""

    def _make_empty_adj(self, n=10):
        """Create an empty discrete adjacency [3, N, N]."""
        return torch.zeros((3, n, n), dtype=torch.int32)

    def _make_motifs(self, types, max_n=10):
        """Create a motif tensor [N] with padding."""
        motifs = torch.zeros(max_n, dtype=torch.long)
        for i, t in enumerate(types):
            motifs[i] = t
        return motifs

    def test_empty_graph_penalty(self):
        """Empty graph (all padding) should get penalty."""
        adj = self._make_empty_adj()
        motifs = torch.zeros(10, dtype=torch.long)  # all padding
        reward = _score_single_graph(adj, motifs)
        assert reward == -1.0

    def test_all_pass_gets_jackpot(self):
        """When all 5 laws pass, reward is multiplied by 2.5."""
        # A minimal valid graph: Boundary(entry) -> Boundary(exit)
        # Motifs: [Boundary=1, Boundary=1, 0, 0, ...]
        motifs = self._make_motifs([1, 1])
        adj = self._make_empty_adj()
        # Execution edge from node 0 to node 1
        adj[0, 0, 1] = 1  # presence
        adj[1, 0, 1] = 0  # exec type
        adj[2, 0, 1] = 0  # index 0

        reward = _score_single_graph(adj, motifs)
        # All 5 laws should pass for this minimal graph:
        # (0.4 + 0.2 + 0.2 + 0.2 + 0.4) * 2.5 = 1.4 * 2.5 = 3.5
        assert reward == pytest.approx(1.4 * JACKPOT_MULTIPLIER, rel=1e-5)

    def test_batch_returns_correct_shape(self):
        """compute_reward should return [B] tensor."""
        B = 4
        adj = torch.zeros((B, 3, 10, 10), dtype=torch.int32)
        motifs = torch.zeros((B, 10), dtype=torch.long)
        motifs[:, 0] = 1  # At least one non-padding node

        rewards = compute_reward(adj, motifs)
        assert rewards.shape == (B,)

    def test_failing_laws_give_penalties(self):
        """A graph that violates laws should get lower reward than a perfect graph."""
        # Condition node with 3 outgoing exec edges (violates out-degree)
        # Also violates in-degree (Condition needs 1 data in, has 0)
        motifs = self._make_motifs([3, 2, 2, 2])  # Condition, Seq, Seq, Seq
        adj = self._make_empty_adj()
        adj[0, 0, 1] = 1  # exec edge 1
        adj[0, 0, 2] = 1  # exec edge 2
        adj[0, 0, 3] = 1  # exec edge 3 (illegal for Condition!)
        adj[1, 0, 1] = 0  # exec
        adj[1, 0, 2] = 0  # exec
        adj[1, 0, 3] = 0  # exec

        reward = _score_single_graph(adj, motifs)
        # Out-degree fails (-0.2), in-degree fails (-0.1), others may pass
        # Reward should be less than the jackpot (3.5)
        assert reward < 1.4 * JACKPOT_MULTIPLIER
