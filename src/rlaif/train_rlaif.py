"""
RLAIF Training Script — DDPO (Denoising Diffusion Policy Optimization).

Treats the 20-step Euler ODE as a 20-step MDP. Each step's velocity prediction
is the mean of a Gaussian policy. The GraphValidator's 5 Laws of Physics provide
the terminal reward. PPO with clipped surrogate + KL anchor optimizes the policy.

Usage:
    python -m src.rlaif.train_rlaif [--checkpoint PATH] [--sigma FLOAT] ...
"""

import torch
import argparse
import os
import time
import copy
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.rlaif.naive_discretizer import naive_discretize
from src.rlaif.reward import compute_reward
from src.rlaif.rollout_buffer import RolloutBuffer
from src.rlaif.ppo import ppo_update


def load_checkpoint(path: str, device: torch.device) -> NeuralUniversalMachineDiT:
    """Load a pre-trained DiT from checkpoint."""
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def rollout(
    policy_model: NeuralUniversalMachineDiT,
    motifs: torch.Tensor,
    padding_mask: torch.Tensor,
    buffer: RolloutBuffer,
    num_steps: int = 20,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Execute the 20-step Euler ODE with Gaussian noise injection.
    Stores trajectory in the rollout buffer.

    Returns:
        x_final: [B, 6, N, N] — the final continuous state for discretization.
    """
    policy_model.eval()
    B, N = motifs.shape
    device = motifs.device
    dt = 1.0 / num_steps

    # Start from pure noise
    x = torch.randn((B, 6, N, N), device=device)

    buffer.set_context(motifs, padding_mask)

    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device)

            # Policy predicts deterministic velocity (mean of Gaussian)
            v_mean_old = policy_model(x, t_tensor, motifs)

            # Sample from Gaussian: v ~ N(v_mean, σ²I)
            noise = torch.randn_like(v_mean_old)
            v_sampled = v_mean_old + sigma * noise

            # Store step in buffer (computes and stores log_prob_old internally)
            buffer.store_step(x, v_mean_old, v_sampled, t_val)

            # Euler integration with sampled velocity
            # Only integrate continuous channels (0-1); categorical channels get latest logits
            x_continuous = x[:, :2, :, :] + v_sampled[:, :2, :, :] * dt
            x_categorical = v_sampled[:, 2:, :, :]  # Take latest categorical logits
            x = torch.cat([x_continuous, x_categorical], dim=1)

    return x


def train_rlaif(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DDPO-RLAIF Training on {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  σ (exploration): {args.sigma}")
    print(f"  β (KL): {args.beta_kl}")
    print(f"  ε (PPO clip): {args.clip_epsilon}")
    print(f"  LR: {args.lr}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  PPO epochs per rollout: {args.ppo_epochs}")

    # --- Load models ---
    print("\nLoading pre-trained checkpoint...")
    policy_model = load_checkpoint(args.checkpoint, device)
    ref_model = load_checkpoint(args.checkpoint, device)

    # Freeze reference model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    print(f"  Policy params: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  Reference frozen: {sum(p.numel() for p in ref_model.parameters()):,}")

    # --- Optimizer ---
    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=1e-5)

    # --- Dataset ---
    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} graphs")

    # --- TensorBoard ---
    run_name = f"rlaif_ddpo_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  TensorBoard: runs/{run_name}")

    # --- Checkpointing ---
    ckpt_dir = "checkpoints/rlaif"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training state ---
    sigma = args.sigma
    baseline_ema = 0.0  # Exponential moving average baseline
    baseline_alpha = 0.99
    global_rollout = 0

    # --- Training loop ---
    print(f"\n{'='*60}")
    print("Starting DDPO-RLAIF Training Loop")
    print(f"{'='*60}\n")

    for epoch in range(1, args.max_epochs + 1):
        epoch_rewards = []
        epoch_svr = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(progress):
            motifs = batch["motifs"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            # === ROLLOUT PHASE ===
            buffer = RolloutBuffer(sigma=sigma)
            x_final = rollout(
                policy_model, motifs, padding_mask, buffer,
                num_steps=args.num_steps, sigma=sigma
            )

            # === REWARD PHASE ===
            discrete_adj = naive_discretize(x_final, motifs)
            rewards = compute_reward(discrete_adj, motifs)

            # Update EMA baseline
            batch_reward_mean = rewards.mean().item()
            if global_rollout == 0:
                baseline_ema = batch_reward_mean
            else:
                baseline_ema = baseline_alpha * baseline_ema + (1 - baseline_alpha) * batch_reward_mean

            buffer.assign_rewards(rewards, baseline_ema)

            # Track SVR (proportion of perfect graphs)
            svr = (rewards > 0).float().mean().item()  # Rough: positive reward ≈ many laws pass

            # === PPO UPDATE PHASE ===
            ppo_metrics = ppo_update(
                policy_model=policy_model,
                ref_model=ref_model,
                buffer=buffer,
                optimizer=optimizer,
                device=device,
                sigma=sigma,
                clip_epsilon=args.clip_epsilon,
                beta_kl=args.beta_kl,
                ppo_epochs=args.ppo_epochs,
                max_grad_norm=1.0,
                kl_early_stop=args.kl_early_stop,
            )

            # === LOGGING ===
            epoch_rewards.append(batch_reward_mean)
            epoch_svr.append(svr)

            writer.add_scalar("Rollout/Reward_Mean", batch_reward_mean, global_rollout)
            writer.add_scalar("Rollout/Reward_Std", rewards.std().item(), global_rollout)
            writer.add_scalar("Rollout/SVR_Approx", svr, global_rollout)
            writer.add_scalar("Rollout/Baseline_EMA", baseline_ema, global_rollout)
            writer.add_scalar("PPO/Loss", ppo_metrics["ppo_loss"], global_rollout)
            writer.add_scalar("PPO/Surrogate_Loss", ppo_metrics["surrogate_loss"], global_rollout)
            writer.add_scalar("PPO/KL_Mean", ppo_metrics["kl_mean"], global_rollout)
            writer.add_scalar("PPO/Clip_Fraction", ppo_metrics["clip_fraction"], global_rollout)
            writer.add_scalar("PPO/Epochs_Completed", ppo_metrics["epochs_completed"], global_rollout)
            writer.add_scalar("Hyperparams/Sigma", sigma, global_rollout)

            progress.set_postfix({
                "reward": f"{batch_reward_mean:.3f}",
                "svr": f"{svr:.1%}",
                "kl": f"{ppo_metrics['kl_mean']:.4f}",
                "clip": f"{ppo_metrics['clip_fraction']:.2f}",
            })

            global_rollout += 1

        # === EPOCH SUMMARY ===
        avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
        avg_svr = sum(epoch_svr) / len(epoch_svr) if epoch_svr else 0
        print(f"\nEpoch {epoch} | Avg Reward: {avg_reward:.4f} | Avg SVR: {avg_svr:.1%} | σ: {sigma:.4f}")

        writer.add_scalar("Epoch/Avg_Reward", avg_reward, epoch)
        writer.add_scalar("Epoch/Avg_SVR", avg_svr, epoch)

        # === SIGMA ANNEALING ===
        if epoch % args.sigma_decay_every == 0:
            sigma = max(sigma * args.sigma_decay_rate, args.sigma_min)
            print(f"  σ annealed to {sigma:.4f}")

        # === CHECKPOINT ===
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"rlaif_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_rollout": global_rollout,
                "model_state_dict": policy_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "sigma": sigma,
                "baseline_ema": baseline_ema,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        # Clean VRAM
        torch.cuda.empty_cache()

    writer.close()
    print("\nRLAIF Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPO-RLAIF Training for Neural Universal Machine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/num_dit_epoch_340.pt",
                        help="Path to pre-trained DiT checkpoint")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Rollout batch size (graphs per rollout)")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of Euler ODE steps")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Initial exploration noise σ")
    parser.add_argument("--sigma-decay-rate", type=float, default=0.95,
                        help="σ multiplicative decay factor")
    parser.add_argument("--sigma-decay-every", type=int, default=5,
                        help="Decay σ every N epochs")
    parser.add_argument("--sigma-min", type=float, default=0.1,
                        help="Minimum σ floor")
    parser.add_argument("--beta-kl", type=float, default=0.5,
                        help="KL penalty coefficient β")
    parser.add_argument("--clip-epsilon", type=float, default=0.2,
                        help="PPO clip parameter ε")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO epochs per rollout")
    parser.add_argument("--kl-early-stop", type=float, default=0.03,
                        help="Stop PPO epoch early if mean KL exceeds this")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    train_rlaif(args)
