"""Tests for the naive threshold discretizer."""

import torch
import pytest
from src.rlaif.naive_discretizer import naive_discretize


class TestNaiveDiscretizer:
    """Test that naive thresholding behaves correctly."""

    def test_output_shape(self):
        """Output should be [B, 3, N, N]."""
        B, N = 2, 16
        continuous = torch.randn(B, 6, N, N)
        motifs = torch.ones(B, N, dtype=torch.long)  # all valid
        result = naive_discretize(continuous, motifs)
        assert result.shape == (B, 3, N, N)

    def test_presence_thresholding(self):
        """Values above 0 in logit space (sigmoid > 0.5) should become 1."""
        B, N = 1, 4
        continuous = torch.zeros(B, 6, N, N)
        motifs = torch.ones(B, N, dtype=torch.long)

        # Set presence channel to high value (sigmoid will be > 0.5)
        continuous[0, 0, 0, 1] = 5.0  # should be 1
        continuous[0, 0, 0, 2] = -5.0  # should be 0

        result = naive_discretize(continuous, motifs)
        assert result[0, 0, 0, 1].item() == 1
        assert result[0, 0, 0, 2].item() == 0

    def test_edge_type_thresholding(self):
        """Type channel: sigmoid > 0.5 → 1 (Data), else 0 (Exec)."""
        B, N = 1, 4
        continuous = torch.zeros(B, 6, N, N)
        motifs = torch.ones(B, N, dtype=torch.long)

        continuous[0, 1, 0, 1] = 5.0  # Data edge
        continuous[0, 1, 0, 2] = -5.0  # Exec edge

        result = naive_discretize(continuous, motifs)
        assert result[0, 1, 0, 1].item() == 1  # Data
        assert result[0, 1, 0, 2].item() == 0  # Exec

    def test_index_argmax(self):
        """Index channel should be argmax of logits 2-5."""
        B, N = 1, 4
        continuous = torch.zeros(B, 6, N, N)
        motifs = torch.ones(B, N, dtype=torch.long)

        # Make channel 4 (index=2) the highest at position (0,1)
        continuous[0, 4, 0, 1] = 10.0  # index 2 wins

        result = naive_discretize(continuous, motifs)
        assert result[0, 2, 0, 1].item() == 2

    def test_padding_mask_zeros_out_padding(self):
        """Padding nodes (motif=0) should have no edges."""
        B, N = 1, 8
        continuous = torch.ones(B, 6, N, N) * 5.0  # Everything would be active
        motifs = torch.zeros(B, N, dtype=torch.long)
        motifs[0, :3] = 1  # Only first 3 nodes are valid

        result = naive_discretize(continuous, motifs)

        # Edges involving padding nodes should be zero
        assert result[0, 0, 0, 5].item() == 0  # valid → padding
        assert result[0, 0, 5, 0].item() == 0  # padding → valid
        assert result[0, 0, 5, 6].item() == 0  # padding → padding

        # Edges between valid nodes should be active
        assert result[0, 0, 0, 1].item() == 1  # valid → valid

    def test_all_channels_are_integers(self):
        """All output values should be integers."""
        B, N = 2, 16
        continuous = torch.randn(B, 6, N, N)
        motifs = torch.ones(B, N, dtype=torch.long)
        result = naive_discretize(continuous, motifs)

        assert result.dtype == torch.int32
