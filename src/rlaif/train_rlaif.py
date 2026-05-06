"""
RLAIF Training Script — Reward-Weighted Regression (RWR) / REINFORCE.

Instead of per-step PPO (which has credit assignment issues across 20 ODE steps),
we use reward-weighted regression: run the ODE with no grad, get a reward, then
do a single backward pass where the flow-matching loss at each step is weighted
by the advantage. This is simpler, faster, and avoids KL explosion.

The KL anchor is enforced by blending the reward-weighted loss with the standard
flow-matching loss against the reference policy's targets.

Usage:
    python -m src.rlaif.train_rlaif [--checkpoint PATH] [--sigma FLOAT] ...
"""

import torch
import argparse
import os
import time
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.rlaif.naive_discretizer import naive_discretize
from src.rlaif.reward import compute_reward


def load_checkpoint(path: str, device: torch.device) -> NeuralUniversalMachineDiT:
    """Load a pre-trained DiT from checkpoint."""
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train_rlaif(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RLAIF-RWR Training on {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  σ (exploration): {args.sigma}")
    print(f"  β_kl (KL weight): {args.beta_kl}")
    print(f"  LR: {args.lr}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  Accum rollouts: {args.accum_rollouts}")

    # --- Load models ---
    print("\nLoading pre-trained checkpoint...")
    policy_model = load_checkpoint(args.checkpoint, device)
    ref_model = load_checkpoint(args.checkpoint, device)

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
    run_name = f"rlaif_rwr_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  TensorBoard: runs/{run_name}")

    # --- Checkpointing ---
    ckpt_dir = "checkpoints/rlaif"
    os.makedirs(ckpt_dir, exist_ok=True)

    # --- Training state ---
    sigma = args.sigma
    baseline_ema = 0.0
    baseline_alpha = 0.9
    global_step = 0

    print(f"\n{'='*60}")
    print("Starting RLAIF-RWR Training Loop")
    print(f"{'='*60}\n")

    for epoch in range(1, args.max_epochs + 1):
        epoch_rewards = []
        epoch_svr = []
        epoch_loss = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}")

        # Accumulation state
        accum_trajectories = []  # List of (x_t_list, noise_list, motifs, padding_mask, rewards)
        accum_count = 0

        for batch_idx, batch in enumerate(progress):
            motifs = batch["motifs"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            B, N = motifs.shape
            dt = 1.0 / args.num_steps

            # === ROLLOUT (no grad) — collect trajectory ===
            policy_model.eval()
            x_t_list = []
            noise_list = []
            t_list = []

            x = torch.randn((B, 6, N, N), device=device)
            with torch.no_grad():
                for step in range(args.num_steps):
                    t_val = step * dt
                    t_tensor = torch.full((B,), t_val, device=device)
                    v_mean = policy_model(x, t_tensor, motifs)

                    # Sample noise and store
                    noise = torch.randn_like(v_mean)
                    v_sampled = v_mean + sigma * noise

                    x_t_list.append(x.cpu())
                    noise_list.append(noise.cpu())
                    t_list.append(t_val)

                    # Euler step
                    x_cont = x[:, :2] + v_sampled[:, :2] * dt
                    x_cat = v_sampled[:, 2:]
                    x = torch.cat([x_cont, x_cat], dim=1)

            # === REWARD ===
            discrete_adj = naive_discretize(x, motifs)
            rewards = compute_reward(discrete_adj, motifs)

            # Update baseline
            batch_reward = rewards.mean().item()
            if global_step == 0 and accum_count == 0:
                baseline_ema = batch_reward
            else:
                baseline_ema = baseline_alpha * baseline_ema + (1 - baseline_alpha) * batch_reward

            svr = (rewards > 0).float().mean().item()

            # Store for accumulation
            accum_trajectories.append({
                "x_t_list": x_t_list,
                "noise_list": noise_list,
                "t_list": t_list,
                "motifs": motifs.cpu(),
                "padding_mask": padding_mask.cpu(),
                "rewards": rewards.cpu(),
            })
            accum_count += 1

            epoch_rewards.append(batch_reward)
            epoch_svr.append(svr)

            # === UPDATE (every accum_rollouts) ===
            if accum_count >= args.accum_rollouts:
                policy_model.train()
                optimizer.zero_grad()

                # Compute normalized advantages across all accumulated graphs
                all_rewards = torch.cat([t["rewards"] for t in accum_trajectories])
                raw_adv = all_rewards - baseline_ema
                if raw_adv.std() > 1e-8:
                    norm_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
                else:
                    norm_adv = raw_adv

                total_rwr_loss = 0.0
                total_kl_loss = 0.0
                num_grad_steps = 0

                # Process each trajectory
                adv_offset = 0
                for traj in accum_trajectories:
                    B_traj = traj["rewards"].shape[0]
                    advantages = norm_adv[adv_offset:adv_offset + B_traj].to(device)
                    adv_offset += B_traj
                    traj_motifs = traj["motifs"].to(device)
                    traj_mask = traj["padding_mask"].to(device)

                    # Pick a random subset of steps to update on (saves VRAM + compute)
                    step_indices = torch.randperm(len(traj["t_list"]))[:args.steps_per_update]

                    for step_idx in step_indices:
                        x_t = traj["x_t_list"][step_idx].to(device)
                        noise = traj["noise_list"][step_idx].to(device)
                        t_val = traj["t_list"][step_idx]
                        t_tensor = torch.full((B_traj,), t_val, device=device)

                        # Policy prediction (with grad)
                        v_new = policy_model(x_t, t_tensor, traj_motifs)

                        # Target = v_mean_old + sigma * noise was the sampled action
                        # The "loss" for REINFORCE/RWR is: -advantage * log_prob
                        # log_prob ∝ -||noise||² (since v_sampled = v_mean + σ*noise)
                        # Gradient of log_prob w.r.t. θ = (noise / σ) · ∂v_mean/∂θ
                        # This simplifies to: minimize advantage-weighted MSE to sampled action
                        v_target = (v_new.detach() + sigma * noise)  # The action we took

                        # RWR loss: weight the MSE by advantage
                        # Positive advantage → push towards this action
                        # Negative advantage → push away from this action
                        mse_per_graph = ((v_new - v_target) ** 2)
                        if traj_mask is not None:
                            mask_expanded = traj_mask.unsqueeze(1).expand_as(mse_per_graph)
                            mse_per_graph = mse_per_graph * mask_expanded
                            n_valid = mask_expanded.sum(dim=(1, 2, 3)).clamp(min=1)
                            mse_per_graph = mse_per_graph.sum(dim=(1, 2, 3)) / n_valid
                        else:
                            mse_per_graph = mse_per_graph.mean(dim=(1, 2, 3))  # [B]

                        # Weighted by advantage (positive = reinforce, negative = avoid)
                        rwr_loss = (advantages * mse_per_graph).mean()

                        # KL anchor: MSE between policy and reference
                        with torch.no_grad():
                            v_ref = ref_model(x_t, t_tensor, traj_motifs)
                        kl_loss = ((v_new - v_ref) ** 2)
                        if traj_mask is not None:
                            kl_loss = kl_loss * mask_expanded
                            kl_loss = kl_loss.sum(dim=(1, 2, 3)) / n_valid
                        else:
                            kl_loss = kl_loss.mean(dim=(1, 2, 3))
                        kl_loss = kl_loss.mean()

                        # Combined loss
                        step_loss = (rwr_loss + args.beta_kl * kl_loss) / (len(accum_trajectories) * args.steps_per_update)
                        step_loss.backward()

                        total_rwr_loss += rwr_loss.item()
                        total_kl_loss += kl_loss.item()
                        num_grad_steps += 1

                # Optimizer step
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()

                # Metrics
                avg_rwr = total_rwr_loss / max(num_grad_steps, 1)
                avg_kl = total_kl_loss / max(num_grad_steps, 1)
                combined_reward = all_rewards.mean().item()
                combined_svr = sum(epoch_svr[-accum_count:]) / accum_count

                writer.add_scalar("Rollout/Reward_Mean", combined_reward, global_step)
                writer.add_scalar("Rollout/Reward_Std", all_rewards.std().item(), global_step)
                writer.add_scalar("Rollout/SVR_Approx", combined_svr, global_step)
                writer.add_scalar("Rollout/Baseline_EMA", baseline_ema, global_step)
                writer.add_scalar("Loss/RWR", avg_rwr, global_step)
                writer.add_scalar("Loss/KL", avg_kl, global_step)
                writer.add_scalar("Loss/Total", avg_rwr + args.beta_kl * avg_kl, global_step)
                writer.add_scalar("Hyperparams/Sigma", sigma, global_step)

                epoch_loss.append(avg_rwr + args.beta_kl * avg_kl)

                progress.set_postfix({
                    "reward": f"{combined_reward:.3f}",
                    "svr": f"{combined_svr:.1%}",
                    "kl": f"{avg_kl:.4f}",
                    "rwr": f"{avg_rwr:.4f}",
                })

                global_step += 1
                accum_trajectories = []
                accum_count = 0

                # Free VRAM
                torch.cuda.empty_cache()

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
                "global_step": global_step,
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
    parser = argparse.ArgumentParser(description="RLAIF-RWR Training for Neural Universal Machine")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/num_dit_epoch_340.pt",
                        help="Path to pre-trained DiT checkpoint")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Rollout batch size (graphs per rollout)")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of Euler ODE steps")
    parser.add_argument("--steps-per-update", type=int, default=5,
                        help="Random subset of ODE steps to backprop through per trajectory")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Initial exploration noise σ")
    parser.add_argument("--sigma-decay-rate", type=float, default=0.95,
                        help="σ multiplicative decay factor")
    parser.add_argument("--sigma-decay-every", type=int, default=5,
                        help="Decay σ every N epochs")
    parser.add_argument("--sigma-min", type=float, default=0.1,
                        help="Minimum σ floor")
    parser.add_argument("--beta-kl", type=float, default=0.01,
                        help="KL penalty coefficient β")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--accum-rollouts", type=int, default=4,
                        help="Accumulate N rollouts before update (effective batch = batch_size × N)")
    parser.add_argument("--max-epochs", type=int, default=200,
                        help="Maximum training epochs")
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    train_rlaif(args)
