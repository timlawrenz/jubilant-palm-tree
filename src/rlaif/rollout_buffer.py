"""
PPO Rollout Buffer for DDPO-style RLAIF.

Stores step-wise trajectory data from the 20-step Euler ODE rollout:
- x_t: noisy state at each step (needed to recompute v_new with grad)
- v_sampled: the actual velocity used (needed for log_prob_new)
- log_prob_old: scalar per step, computed at collection time, FIXED across PPO epochs
- t: timestep value
- motifs: conditioning signal

The log_prob_old is stored as a scalar (not the full v_mean_old tensor) for memory efficiency.
This is the standard DDPO approach: ~300MB total vs ~600MB if storing v_mean_old.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepData:
    """Data for a single Euler step across the batch."""
    x_t: torch.Tensor          # [B, 6, N, N]
    v_sampled: torch.Tensor    # [B, 6, N, N]
    log_prob_old: torch.Tensor  # [B] scalar per graph
    log_prob_cat_old: torch.Tensor  # [B] categorical log-prob
    t_val: float               # scalar timestep


class RolloutBuffer:
    """
    Collects trajectory data during the rollout phase and serves it
    during the PPO update phase.
    """

    def __init__(self, sigma: float = 1.0):
        self.sigma = sigma
        self.steps: list[StepData] = []
        self.motifs: Optional[torch.Tensor] = None  # [B, N]
        self.padding_mask: Optional[torch.Tensor] = None  # [B, N, N]
        self.rewards: Optional[torch.Tensor] = None  # [B]
        self.advantages: Optional[torch.Tensor] = None  # [B]

    def reset(self):
        """Clear buffer for next rollout."""
        self.steps = []
        self.motifs = None
        self.padding_mask = None
        self.rewards = None
        self.advantages = None

    def set_context(self, motifs: torch.Tensor, padding_mask: torch.Tensor):
        """Store the conditioning context for this rollout batch."""
        self.motifs = motifs.detach().cpu()
        self.padding_mask = padding_mask.detach().cpu()

    def store_step(
        self,
        x_t: torch.Tensor,
        v_mean_old: torch.Tensor,
        v_sampled: torch.Tensor,
        t_val: float,
    ):
        """
        Store one Euler step's data. Computes log_prob_old from v_mean_old
        and v_sampled, then discards v_mean_old to save memory.

        Args:
            x_t: [B, 6, N, N] noisy state
            v_mean_old: [B, 6, N, N] policy's deterministic prediction (will be discarded)
            v_sampled: [B, 6, N, N] actual sampled velocity (v_mean_old + σε)
            t_val: scalar timestep
        """
        # Compute log_prob_old for continuous channels (0-1)
        log_prob_cont = self._compute_continuous_log_prob(
            v_sampled[:, :2, :, :], v_mean_old[:, :2, :, :]
        )

        # Compute log_prob_old for categorical channels (2-5)
        log_prob_cat = self._compute_categorical_log_prob(
            v_sampled[:, 2:, :, :], v_mean_old[:, 2:, :, :]
        )

        self.steps.append(StepData(
            x_t=x_t.detach().cpu(),
            v_sampled=v_sampled.detach().cpu(),
            log_prob_old=log_prob_cont.detach().cpu(),
            log_prob_cat_old=log_prob_cat.detach().cpu(),
            t_val=t_val,
        ))

    def assign_rewards(self, rewards: torch.Tensor, baseline: float):
        """
        Assign terminal rewards and compute advantages.

        Args:
            rewards: [B] per-graph reward from GraphValidator
            baseline: EMA baseline for advantage computation
        """
        self.rewards = rewards.detach().cpu()
        self.advantages = (self.rewards - baseline)

    def num_steps(self) -> int:
        return len(self.steps)

    def batch_size(self) -> int:
        if self.steps:
            return self.steps[0].x_t.shape[0]
        return 0

    def get_step(self, step_idx: int, device: torch.device) -> dict:
        """
        Retrieve a single step's data, moved to the target device.

        Returns dict with: x_t, v_sampled, log_prob_old, log_prob_cat_old,
                          t_val, motifs, advantages
        """
        step = self.steps[step_idx]
        return {
            "x_t": step.x_t.to(device),
            "v_sampled": step.v_sampled.to(device),
            "log_prob_old": step.log_prob_old.to(device),
            "log_prob_cat_old": step.log_prob_cat_old.to(device),
            "t_val": step.t_val,
            "motifs": self.motifs.to(device),
            "advantages": self.advantages.to(device),
        }

    def iterate_steps(self, device: torch.device, shuffle: bool = True):
        """
        Yields step data dicts for PPO update.
        Optionally shuffles step order for better gradient estimates.
        """
        indices = list(range(self.num_steps()))
        if shuffle:
            import random
            random.shuffle(indices)

        for idx in indices:
            yield self.get_step(idx, device)

    def _compute_continuous_log_prob(
        self, v_sampled: torch.Tensor, v_mean: torch.Tensor
    ) -> torch.Tensor:
        """
        Gaussian log-prob for continuous channels.
        log π(v|x,t) ∝ -||v_sampled - v_mean||² / (2σ²)

        Computed per-graph as masked sum over spatial dimensions.
        Returns: [B] scalar log-prob per graph.
        """
        B = v_sampled.shape[0]

        # Squared difference per element
        sq_diff = (v_sampled - v_mean) ** 2  # [B, 2, N, N]

        # Apply padding mask if available
        if self.padding_mask is not None:
            mask = self.padding_mask.unsqueeze(1).expand_as(sq_diff)  # [B, 2, N, N]
            sq_diff = sq_diff * mask

        # Sum over all dimensions except batch, then normalize
        mse_per_graph = sq_diff.sum(dim=(1, 2, 3))  # [B]
        log_prob = -mse_per_graph / (2.0 * self.sigma ** 2)

        return log_prob

    def _compute_categorical_log_prob(
        self, v_sampled_logits: torch.Tensor, v_mean_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Categorical log-prob for index channels.
        The 'action' is the argmax of the sampled logits.
        The log-prob is computed under the policy's softmax distribution.

        Args:
            v_sampled_logits: [B, 4, N, N] — the sampled categorical logits
            v_mean_logits: [B, 4, N, N] — the policy's predicted logits (distribution)

        Returns: [B] scalar log-prob per graph.
        """
        B = v_sampled_logits.shape[0]

        # The "action" taken is the argmax of sampled logits
        actions = torch.argmax(v_sampled_logits, dim=1)  # [B, N, N]

        # Compute log-prob under the policy's distribution
        log_probs = torch.nn.functional.log_softmax(v_mean_logits, dim=1)  # [B, 4, N, N]

        # Gather the log-prob of the chosen action
        actions_expanded = actions.unsqueeze(1)  # [B, 1, N, N]
        chosen_log_probs = torch.gather(log_probs, 1, actions_expanded).squeeze(1)  # [B, N, N]

        # Apply padding mask and sum
        if self.padding_mask is not None:
            chosen_log_probs = chosen_log_probs * self.padding_mask

        log_prob_per_graph = chosen_log_probs.sum(dim=(1, 2))  # [B]

        return log_prob_per_graph
