"""Tests for the PPO rollout buffer."""

import torch
import pytest
from src.rlaif.rollout_buffer import RolloutBuffer


class TestRolloutBuffer:
    """Test buffer storage, retrieval, and log_prob computation."""

    def _make_buffer(self, sigma=1.0, B=2, N=8, num_steps=3):
        """Create a buffer with a few steps stored."""
        buffer = RolloutBuffer(sigma=sigma)
        motifs = torch.ones(B, N, dtype=torch.long)
        padding_mask = torch.ones(B, N, N)
        buffer.set_context(motifs, padding_mask)

        for step in range(num_steps):
            x_t = torch.randn(B, 6, N, N)
            v_mean = torch.randn(B, 6, N, N)
            noise = torch.randn(B, 6, N, N)
            v_sampled = v_mean + sigma * noise
            buffer.store_step(x_t, v_mean, v_sampled, step * 0.05)

        return buffer

    def test_stores_correct_number_of_steps(self):
        """Buffer should store the right number of steps."""
        buffer = self._make_buffer(num_steps=5)
        assert buffer.num_steps() == 5

    def test_batch_size(self):
        """Buffer should report correct batch size."""
        buffer = self._make_buffer(B=4)
        assert buffer.batch_size() == 4

    def test_log_prob_old_is_scalar_per_graph(self):
        """log_prob_old should be [B], not a full tensor."""
        buffer = self._make_buffer(B=3)
        step = buffer.steps[0]
        assert step.log_prob_old.shape == (3,)

    def test_log_prob_old_is_negative(self):
        """Log-prob should always be ≤ 0 (Gaussian log-prob is negative)."""
        buffer = self._make_buffer()
        for step in buffer.steps:
            assert (step.log_prob_old <= 0).all()

    def test_log_prob_zero_when_v_sampled_equals_mean(self):
        """If v_sampled == v_mean (no noise), log_prob should be 0."""
        B, N = 2, 8
        buffer = RolloutBuffer(sigma=1.0)
        motifs = torch.ones(B, N, dtype=torch.long)
        padding_mask = torch.ones(B, N, N)
        buffer.set_context(motifs, padding_mask)

        v_mean = torch.randn(B, 6, N, N)
        v_sampled = v_mean.clone()  # No noise!
        x_t = torch.randn(B, 6, N, N)

        buffer.store_step(x_t, v_mean, v_sampled, 0.0)

        # Continuous log_prob should be 0 (MSE = 0)
        assert buffer.steps[0].log_prob_old.abs().max() < 1e-5

    def test_assign_rewards_and_advantages(self):
        """Rewards and advantages should be correctly stored."""
        buffer = self._make_buffer(B=3, num_steps=2)
        rewards = torch.tensor([1.0, -0.5, 2.0])
        baseline = 0.5

        buffer.assign_rewards(rewards, baseline)

        assert buffer.rewards is not None
        assert buffer.advantages is not None
        expected_adv = rewards - baseline
        assert torch.allclose(buffer.advantages, expected_adv)

    def test_get_step_returns_correct_keys(self):
        """get_step should return all required fields."""
        buffer = self._make_buffer(num_steps=2)
        buffer.assign_rewards(torch.tensor([0.5, 0.5]), baseline=0.0)

        step_data = buffer.get_step(0, torch.device("cpu"))
        required_keys = {"x_t", "v_sampled", "log_prob_old", "log_prob_cat_old",
                         "t_val", "motifs", "advantages"}
        assert required_keys == set(step_data.keys())

    def test_reset_clears_buffer(self):
        """reset() should clear all stored data."""
        buffer = self._make_buffer(num_steps=5)
        buffer.assign_rewards(torch.tensor([1.0, 1.0]), baseline=0.0)

        buffer.reset()

        assert buffer.num_steps() == 0
        assert buffer.motifs is None
        assert buffer.rewards is None

    def test_iterate_steps_yields_all_steps(self):
        """iterate_steps should yield all stored steps."""
        buffer = self._make_buffer(num_steps=7)
        buffer.assign_rewards(torch.tensor([0.0, 0.0]), baseline=0.0)

        steps_yielded = list(buffer.iterate_steps(torch.device("cpu"), shuffle=False))
        assert len(steps_yielded) == 7
