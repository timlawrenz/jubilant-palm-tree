"""Tests for the PPO module."""

import torch
import pytest
from src.models.model import NeuralUniversalMachineDiT
from src.rlaif.ppo import (
    _continuous_log_prob, _categorical_log_prob,
    _continuous_kl, _categorical_kl, ppo_update
)
from src.rlaif.rollout_buffer import RolloutBuffer


class TestLogProbFunctions:
    """Test log-prob helper functions."""

    def test_continuous_log_prob_is_negative(self):
        """Gaussian log-prob should be ≤ 0."""
        v_sampled = torch.randn(2, 2, 8, 8)
        v_mean = torch.randn(2, 2, 8, 8)
        result = _continuous_log_prob(v_sampled, v_mean, sigma=1.0, padding_mask=None)
        assert (result <= 0).all()

    def test_continuous_log_prob_zero_when_equal(self):
        """log_prob = 0 when v_sampled == v_mean."""
        v = torch.randn(2, 2, 8, 8)
        result = _continuous_log_prob(v, v, sigma=1.0, padding_mask=None)
        assert result.abs().max() < 1e-5

    def test_continuous_log_prob_decreases_with_distance(self):
        """Further from mean → lower log-prob."""
        v_mean = torch.zeros(1, 2, 4, 4)
        v_close = torch.ones(1, 2, 4, 4) * 0.1
        v_far = torch.ones(1, 2, 4, 4) * 5.0

        lp_close = _continuous_log_prob(v_close, v_mean, sigma=1.0, padding_mask=None)
        lp_far = _continuous_log_prob(v_far, v_mean, sigma=1.0, padding_mask=None)
        assert lp_close.item() > lp_far.item()

    def test_categorical_log_prob_shape(self):
        """Should return [B] tensor."""
        v_sampled = torch.randn(3, 4, 8, 8)
        v_mean = torch.randn(3, 4, 8, 8)
        result = _categorical_log_prob(v_sampled, v_mean, padding_mask=None)
        assert result.shape == (3,)


class TestKLFunctions:
    """Test KL divergence helpers."""

    def test_continuous_kl_zero_for_identical(self):
        """KL should be 0 when policy == reference."""
        v = torch.randn(2, 2, 8, 8)
        kl = _continuous_kl(v, v, sigma=1.0, padding_mask=None)
        assert kl.abs().max() < 1e-5

    def test_continuous_kl_positive_for_different(self):
        """KL should be positive when policy != reference."""
        v_policy = torch.randn(2, 2, 8, 8)
        v_ref = torch.randn(2, 2, 8, 8)
        kl = _continuous_kl(v_policy, v_ref, sigma=1.0, padding_mask=None)
        assert (kl > 0).all()

    def test_categorical_kl_zero_for_identical(self):
        """KL should be ~0 when logits are identical."""
        logits = torch.randn(2, 4, 8, 8)
        kl = _categorical_kl(logits, logits, padding_mask=None)
        assert kl.abs().max() < 1e-4

    def test_categorical_kl_positive_for_different(self):
        """KL should be positive for different distributions."""
        logits_p = torch.randn(2, 4, 8, 8)
        logits_q = torch.randn(2, 4, 8, 8)
        kl = _categorical_kl(logits_p, logits_q, padding_mask=None)
        assert (kl >= 0).all()


class TestPPOUpdate:
    """Integration test for the PPO update step."""

    def _make_small_model(self):
        """Create a tiny DiT for testing."""
        return NeuralUniversalMachineDiT(hidden_dim=32, num_heads=2, depth=2)

    def test_ppo_update_runs_without_error(self):
        """PPO update should complete on a small model + buffer."""
        B, N = 2, 8
        device = torch.device("cpu")

        policy = self._make_small_model().to(device)
        ref = self._make_small_model().to(device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

        # Build a small buffer
        buffer = RolloutBuffer(sigma=1.0)
        motifs = torch.randint(1, 7, (B, N))
        # Pad to 128 for the model
        motifs_padded = torch.zeros(B, 128, dtype=torch.long)
        motifs_padded[:, :N] = motifs
        padding_mask = torch.zeros(B, 128, 128)
        padding_mask[:, :N, :N] = 1.0

        buffer.set_context(motifs_padded, padding_mask)

        # Store 3 fake steps
        for step in range(3):
            x_t = torch.randn(B, 6, 128, 128)
            v_mean = policy(x_t, torch.full((B,), step * 0.05), motifs_padded)
            v_sampled = v_mean + torch.randn_like(v_mean)
            buffer.store_step(x_t, v_mean.detach(), v_sampled.detach(), step * 0.05)

        buffer.assign_rewards(torch.tensor([0.5, -0.2]), baseline=0.0)

        # Run PPO
        metrics = ppo_update(
            policy_model=policy,
            ref_model=ref,
            buffer=buffer,
            optimizer=optimizer,
            device=device,
            sigma=1.0,
            ppo_epochs=2,
        )

        assert "ppo_loss" in metrics
        assert metrics["epochs_completed"] > 0

    def test_ratio_is_one_before_update(self):
        """Before any update, policy == old_policy so ratio should be ~1."""
        B, N = 1, 8
        device = torch.device("cpu")

        model = self._make_small_model().to(device)

        buffer = RolloutBuffer(sigma=1.0)
        motifs = torch.randint(1, 7, (B, N))
        motifs_padded = torch.zeros(B, 128, dtype=torch.long)
        motifs_padded[:, :N] = motifs
        padding_mask = torch.zeros(B, 128, 128)
        padding_mask[:, :N, :N] = 1.0

        buffer.set_context(motifs_padded, padding_mask)

        x_t = torch.randn(B, 6, 128, 128)
        with torch.no_grad():
            v_mean = model(x_t, torch.full((B,), 0.0), motifs_padded)
        v_sampled = v_mean + torch.randn_like(v_mean) * 1.0
        buffer.store_step(x_t, v_mean, v_sampled, 0.0)

        # Recompute v_new with the same (unmodified) model
        with torch.no_grad():
            v_new = model(x_t, torch.full((B,), 0.0), motifs_padded)

        # Log-probs should be very close
        lp_old = buffer.steps[0].log_prob_old
        lp_new = _continuous_log_prob(
            v_sampled[:, :2], v_new[:, :2], sigma=1.0, padding_mask=padding_mask
        )

        ratio = torch.exp(lp_new - lp_old)
        # Ratio should be very close to 1
        assert (ratio - 1.0).abs().max() < 0.01
