"""
PPO (Proximal Policy Optimization) with Gaussian log-prob for DDPO-style RLAIF.

The policy is π_θ(v_t | x_t, t) = N(v_θ(x_t, t), σ²I).
Log-prob is just negative MSE: log π = -||v_sampled - v_mean||² / (2σ²).
KL between policy and reference is also MSE: KL = ||v_policy - v_ref||² / (2σ²).

This module handles the PPO update phase: iterating over the rollout buffer,
recomputing log_probs with grad, and applying the clipped surrogate objective.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

from src.models.model import NeuralUniversalMachineDiT
from src.rlaif.rollout_buffer import RolloutBuffer


def ppo_update(
    policy_model: NeuralUniversalMachineDiT,
    ref_model: NeuralUniversalMachineDiT,
    buffer: RolloutBuffer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    sigma: float = 1.0,
    clip_epsilon: float = 0.2,
    beta_kl: float = 0.1,
    ppo_epochs: int = 4,
    max_grad_norm: float = 1.0,
    kl_early_stop: float = 0.05,
) -> Dict[str, float]:
    """
    Performs PPO updates over the collected rollout buffer.

    Args:
        policy_model: The trainable DiT (active policy).
        ref_model: The frozen reference DiT (KL anchor).
        buffer: Filled RolloutBuffer from rollout phase.
        optimizer: AdamW optimizer for policy_model.
        device: CUDA/CPU device.
        sigma: Exploration noise std (shared with rollout).
        clip_epsilon: PPO clip parameter.
        beta_kl: KL penalty coefficient.
        ppo_epochs: Number of passes over the buffer.
        max_grad_norm: Gradient clipping norm.
        kl_early_stop: If mean KL exceeds this, stop PPO early.

    Returns:
        Dict of metrics: loss, clip_fraction, kl_mean, etc.
    """
    policy_model.train()
    ref_model.eval()

    metrics = {
        "ppo_loss": 0.0,
        "clip_fraction": 0.0,
        "kl_mean": 0.0,
        "surrogate_loss": 0.0,
        "kl_loss": 0.0,
        "epochs_completed": 0,
    }

    total_steps = 0
    early_stopped = False

    for ppo_epoch in range(ppo_epochs):
        epoch_kl_sum = 0.0
        epoch_steps = 0

        for step_data in buffer.iterate_steps(device, shuffle=True):
            x_t = step_data["x_t"]
            v_sampled = step_data["v_sampled"]
            log_prob_old = step_data["log_prob_old"]
            log_prob_cat_old = step_data["log_prob_cat_old"]
            t_val = step_data["t_val"]
            motifs = step_data["motifs"]
            advantages = step_data["advantages"]

            B = x_t.shape[0]
            t_tensor = torch.full((B,), t_val, device=device)

            # --- Recompute v_new with gradients ---
            v_new = policy_model(x_t, t_tensor, motifs)

            # --- Compute log_prob_new (continuous channels) ---
            log_prob_cont_new = _continuous_log_prob(
                v_sampled[:, :2, :, :], v_new[:, :2, :, :],
                sigma, buffer.padding_mask.to(device) if buffer.padding_mask is not None else None
            )

            # --- Compute log_prob_new (categorical channels) ---
            log_prob_cat_new = _categorical_log_prob(
                v_sampled[:, 2:, :, :], v_new[:, 2:, :, :],
                buffer.padding_mask.to(device) if buffer.padding_mask is not None else None
            )

            # --- Combined log-prob ---
            log_prob_new_total = log_prob_cont_new + log_prob_cat_new
            log_prob_old_total = log_prob_old + log_prob_cat_old

            # --- Importance sampling ratio ---
            log_ratio = log_prob_new_total - log_prob_old_total
            ratio = torch.exp(log_ratio)

            # --- Clipped surrogate objective ---
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            surrogate_loss = -torch.min(surr1, surr2).mean()

            # --- KL penalty (Gaussian: MSE between policy and reference means) ---
            with torch.no_grad():
                v_ref = ref_model(x_t, t_tensor, motifs)

            kl_continuous = _continuous_kl(v_new[:, :2, :, :], v_ref[:, :2, :, :], sigma,
                                          buffer.padding_mask.to(device) if buffer.padding_mask is not None else None)
            kl_categorical = _categorical_kl(v_new[:, 2:, :, :], v_ref[:, 2:, :, :],
                                             buffer.padding_mask.to(device) if buffer.padding_mask is not None else None)
            kl_total = (kl_continuous + kl_categorical).mean()

            # --- Total loss ---
            loss = surrogate_loss + beta_kl * kl_total

            # --- Backward + step ---
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            optimizer.step()

            # --- Metrics ---
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
            metrics["ppo_loss"] += loss.item()
            metrics["surrogate_loss"] += surrogate_loss.item()
            metrics["kl_loss"] += kl_total.item()
            metrics["clip_fraction"] += clip_fraction

            epoch_kl_sum += kl_total.item()
            epoch_steps += 1
            total_steps += 1

        metrics["epochs_completed"] = ppo_epoch + 1

        # Early stopping on KL
        if epoch_steps > 0:
            epoch_kl_mean = epoch_kl_sum / epoch_steps
            if epoch_kl_mean > kl_early_stop:
                early_stopped = True
                break

    # Normalize metrics
    if total_steps > 0:
        metrics["ppo_loss"] /= total_steps
        metrics["surrogate_loss"] /= total_steps
        metrics["kl_loss"] /= total_steps
        metrics["kl_mean"] = metrics["kl_loss"]
        metrics["clip_fraction"] /= total_steps

    metrics["early_stopped"] = early_stopped

    return metrics


def _continuous_log_prob(
    v_sampled: torch.Tensor,
    v_mean: torch.Tensor,
    sigma: float,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Gaussian log-prob: -||v_sampled - v_mean||² / (2σ²), masked and summed.
    Returns: [B] scalar per graph.
    """
    sq_diff = (v_sampled - v_mean) ** 2  # [B, 2, N, N]
    if padding_mask is not None:
        mask = padding_mask.unsqueeze(1).expand_as(sq_diff)
        sq_diff = sq_diff * mask
    mse_per_graph = sq_diff.sum(dim=(1, 2, 3))  # [B]
    return -mse_per_graph / (2.0 * sigma ** 2)


def _categorical_log_prob(
    v_sampled_logits: torch.Tensor,
    v_mean_logits: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Categorical log-prob of the sampled action under the policy distribution.
    Returns: [B] scalar per graph.
    """
    actions = torch.argmax(v_sampled_logits, dim=1)  # [B, N, N]
    log_probs = F.log_softmax(v_mean_logits, dim=1)  # [B, 4, N, N]
    chosen = torch.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(1)  # [B, N, N]
    if padding_mask is not None:
        chosen = chosen * padding_mask
    return chosen.sum(dim=(1, 2))  # [B]


def _continuous_kl(
    v_policy: torch.Tensor,
    v_ref: torch.Tensor,
    sigma: float,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    KL divergence between two Gaussians with shared σ:
    KL(π_policy || π_ref) = ||v_policy - v_ref||² / (2σ²)
    Returns: [B] scalar per graph.
    """
    sq_diff = (v_policy - v_ref) ** 2
    if padding_mask is not None:
        mask = padding_mask.unsqueeze(1).expand_as(sq_diff)
        sq_diff = sq_diff * mask
    return sq_diff.sum(dim=(1, 2, 3)) / (2.0 * sigma ** 2)


def _categorical_kl(
    logits_policy: torch.Tensor,
    logits_ref: torch.Tensor,
    padding_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    KL divergence between categorical distributions (index logits).
    KL(p || q) = Σ p(x) * (log p(x) - log q(x))
    Returns: [B] scalar per graph.
    """
    p = F.softmax(logits_policy, dim=1)  # [B, 4, N, N]
    log_p = F.log_softmax(logits_policy, dim=1)
    log_q = F.log_softmax(logits_ref, dim=1)

    kl = (p * (log_p - log_q)).sum(dim=1)  # [B, N, N]
    if padding_mask is not None:
        kl = kl * padding_mask
    return kl.sum(dim=(1, 2))  # [B]
