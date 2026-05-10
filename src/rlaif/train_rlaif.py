"""
RLAIF Training Script — Differentiable Structural Loss.

Instead of REINFORCE/RWR (which can't solve credit assignment across 20 ODE steps
with 98K-dimensional noise), we use differentiable structural constraints that
backprop directly through the last ODE step.

Approach:
  1. Run ODE steps 0..N-2 with no grad (cheap rollout)
  2. Run the LAST step with grad → get x_final
  3. Compute differentiable structural loss on x_final (out-degree, sharpness, etc.)
  4. Compute KL anchor: MSE between policy velocity and reference velocity
  5. Backprop combined loss → optimizer step
  6. Periodically evaluate discrete SVR (no grad) to measure real progress

This gives dense, per-element gradient signal without any REINFORCE variance.

Usage:
    python -m src.rlaif.train_rlaif [--checkpoint PATH] [--lr FLOAT] ...
"""

import copy
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
from src.models.validation import GraphValidator
from src.rlaif.structural_loss import compute_structural_loss


def load_checkpoint(path: str, device: torch.device) -> NeuralUniversalMachineDiT:
    """Load a pre-trained DiT from checkpoint."""
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def train_rlaif(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"RLAIF Structural Loss Training on {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Resume: {args.resume or 'None (fresh start)'}")
    print(f"  β_kl (KL weight): {args.beta_kl}")
    print(f"  β_struct (structural weight): {args.beta_struct}")
    print(f"  β_recon (reconstruction weight): {args.beta_recon}")
    print(f"  KL clamp: {args.kl_clamp}")
    print(f"  LR: {args.lr}")
    print(f"  ODE steps: {args.num_steps}")
    print(f"  Grad steps: {args.grad_steps} (last N steps with grad)")

    # --- Load models ---
    print("\nLoading pre-trained checkpoint...")
    policy_model = load_checkpoint(args.checkpoint, device)
    ref_model = load_checkpoint(args.checkpoint, device)

    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # --- Resume from RLAIF checkpoint if provided ---
    start_epoch = 1
    global_step = 0
    if args.resume:
        print(f"  Resuming from: {args.resume}")
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        policy_model.load_state_dict(resume_ckpt["model_state_dict"])
        start_epoch = resume_ckpt["epoch"] + 1
        global_step = resume_ckpt.get("global_step", 0)
        print(f"  Resuming at epoch {start_epoch}, global_step {global_step}")

    print(f"  Policy params: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"  Reference frozen: {sum(p.numel() for p in ref_model.parameters()):,}")

    # --- EMA model ---
    ema_model = copy.deepcopy(policy_model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad = False
    ema_decay = args.ema_decay
    print(f"  EMA decay: {ema_decay}")

    # --- Optimizer ---
    optimizer = AdamW(policy_model.parameters(), lr=args.lr, weight_decay=1e-5)
    if args.resume and "optimizer_state_dict" in resume_ckpt:
        optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])

    # --- Dataset ---
    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"  Dataset: {len(dataset)} graphs")

    # --- TensorBoard ---
    run_name = f"rlaif_struct_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"  TensorBoard: runs/{run_name}")

    # --- Checkpointing ---
    ckpt_dir = "checkpoints/rlaif"
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print("Starting RLAIF Structural Loss Training")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.max_epochs + 1):
        epoch_start_time = time.time()
        epoch_struct_losses = []
        epoch_kl_losses = []
        epoch_svr = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(progress):
            motifs = batch["motifs"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            x_target = batch["adjacency"].to(device)  # Ground truth graph
            B, N = motifs.shape
            dt = 1.0 / args.num_steps
            no_grad_steps = args.num_steps - args.grad_steps

            optimizer.zero_grad()

            # === Phase 1: No-grad rollout (steps 0 to no_grad_steps-1) ===
            policy_model.eval()
            x = torch.randn((B, 6, N, N), device=device)
            with torch.no_grad():
                for step in range(no_grad_steps):
                    t_val = step * dt
                    t_tensor = torch.full((B,), t_val, device=device)
                    v = policy_model(x, t_tensor, motifs)
                    x_cont = x[:, :2] + v[:, :2] * dt
                    x_cat = v[:, 2:]
                    x = torch.cat([x_cont, x_cat], dim=1)

            # === Phase 2: Grad rollout (last grad_steps steps) ===
            policy_model.train()
            x = x.detach()  # Cut gradient tape at boundary
            kl_losses = []

            for step in range(no_grad_steps, args.num_steps):
                t_val = step * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v_policy = policy_model(x, t_tensor, motifs)

                # KL anchor: MSE to reference velocity
                with torch.no_grad():
                    v_ref = ref_model(x, t_tensor, motifs)

                kl_per_elem = (v_policy - v_ref) ** 2
                mask_exp = padding_mask.unsqueeze(1).expand_as(kl_per_elem)
                kl_per_elem = kl_per_elem * mask_exp
                kl = kl_per_elem.sum(dim=(1, 2, 3)) / mask_exp.sum(dim=(1, 2, 3)).clamp(min=1)
                kl_losses.append(kl.mean())

                # Euler step (with grad flowing through v_policy)
                x_cont = x[:, :2] + v_policy[:, :2] * dt
                x_cat = v_policy[:, 2:]
                x = torch.cat([x_cont, x_cat], dim=1)

            x_final = x  # [B, 6, N, N] — final output, has grad

            # === Structural loss on final output ===
            struct_losses = compute_structural_loss(x_final, motifs)
            struct_loss = struct_losses["total"].mean()

            # === Reconstruction loss (MSE to ground truth graph) ===
            # x_target is [B, 3, N, N]: ch0=presence, ch1=edge_type, ch2=input_index
            # x_final is [B, 6, N, N]: ch0=presence, ch1=data_type, ch2:6=input_index_logits
            # Expand target to 6 channels to match model output
            target_6ch = torch.zeros_like(x_final)
            target_6ch[:, 0] = x_target[:, 0]  # presence
            target_6ch[:, 1] = x_target[:, 1]  # edge/data type
            # One-hot encode input_index (0-3) into channels 2-5
            idx = x_target[:, 2].long().clamp(0, 3)  # [B, N, N]
            target_6ch[:, 2:].scatter_(1, idx.unsqueeze(1), 1.0)

            mask_2d = padding_mask  # [B, N, N] — already 2D mask from dataset
            mask_6d = mask_2d.unsqueeze(1).expand_as(x_final)  # [B, 6, N, N]
            recon_diff = (x_final - target_6ch) ** 2 * mask_6d
            recon_loss = recon_diff.sum() / mask_6d.sum().clamp(min=1)
            recon_loss = torch.clamp(recon_loss, max=100.0)  # prevent explosion

            # === KL loss (clamped to prevent explosion) ===
            kl_loss = sum(kl_losses) / len(kl_losses) if kl_losses else torch.tensor(0.0)
            kl_loss = torch.clamp(kl_loss, max=args.kl_clamp)

            # Clamp structural loss to prevent wild oscillations from sharpened NOTEARS
            struct_loss = torch.clamp(struct_loss, max=50.0)

            # === Combined loss ===
            total_loss = (
                args.beta_struct * struct_loss
                + args.beta_recon * recon_loss
                + args.beta_kl * kl_loss
            )

            # Skip batch if loss is NaN or Inf (prevents weight corruption)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"  ⚠ Batch {batch_idx}: NaN/Inf loss (struct={struct_loss.item():.2f}, recon={recon_loss.item():.2f}, kl={kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss:.2f}) — skipping")
                optimizer.zero_grad()
                global_step += 1
                continue

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            # === EMA update ===
            with torch.no_grad():
                for p_ema, p_policy in zip(ema_model.parameters(), policy_model.parameters()):
                    p_ema.mul_(ema_decay).add_(p_policy, alpha=1.0 - ema_decay)

            # === Discrete SVR evaluation (every eval_every batches) ===
            svr = 0.0
            if batch_idx % args.eval_every == 0:
                policy_model.eval()
                with torch.no_grad():
                    x_eval = torch.randn((B, 6, N, N), device=device)
                    for step in range(args.num_steps):
                        t_val = step * dt
                        t_tensor = torch.full((B,), t_val, device=device)
                        v = policy_model(x_eval, t_tensor, motifs)
                        x_cont = x_eval[:, :2] + v[:, :2] * dt
                        x_cat = v[:, 2:]
                        x_eval = torch.cat([x_cont, x_cat], dim=1)

                    discrete = naive_discretize(x_eval, motifs)
                    val_results = GraphValidator.evaluate_batch(discrete, motifs)
                    svr = val_results["perfect_graphs"] / 100.0

                writer.add_scalar("Eval/SVR", svr, global_step)
                for law_key in ["out_degree_pass", "in_degree_pass", "no_orphan_pass",
                                "acyclic_data_pass", "terminal_sink_pass"]:
                    writer.add_scalar(f"Eval/{law_key}", val_results[law_key], global_step)

            # === Logging ===
            sl = struct_loss.item()
            rl = recon_loss.item()
            kl = kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            epoch_struct_losses.append(sl)
            epoch_kl_losses.append(kl)
            if batch_idx % args.eval_every == 0:
                epoch_svr.append(svr)

            writer.add_scalar("Loss/Structural", sl, global_step)
            writer.add_scalar("Loss/Reconstruction", rl, global_step)
            writer.add_scalar("Loss/KL", kl, global_step)
            writer.add_scalar("Loss/Total", args.beta_struct * sl + args.beta_recon * rl + args.beta_kl * kl, global_step)

            # Log individual structural components
            for key in ["out_degree", "sharpness", "type_sharpness", "terminal", "orphan", "data_in", "acyclic", "density"]:
                if key in struct_losses:
                    writer.add_scalar(f"Structural/{key}", struct_losses[key].mean().item(), global_step)

            progress.set_postfix({
                "struct": f"{sl:.4f}",
                "recon": f"{rl:.4f}",
                "kl": f"{kl:.4f}",
                "svr": f"{svr:.1%}" if batch_idx % args.eval_every == 0 else "—",
            })

            global_step += 1
            torch.cuda.empty_cache()

        # === EPOCH SUMMARY ===
        epoch_duration = time.time() - epoch_start_time
        avg_struct = sum(epoch_struct_losses) / len(epoch_struct_losses) if epoch_struct_losses else float('nan')
        avg_kl = sum(epoch_kl_losses) / len(epoch_kl_losses) if epoch_kl_losses else float('nan')
        avg_svr = sum(epoch_svr) / len(epoch_svr) if epoch_svr else 0
        n_skipped = len(dataloader) - len(epoch_struct_losses)
        print(f"\nEpoch {epoch} | Struct: {avg_struct:.4f} | KL: {avg_kl:.4f} | SVR: {avg_svr:.1%} | Time: {epoch_duration/60:.1f}min | Skipped: {n_skipped}")

        writer.add_scalar("Epoch/Avg_Structural", avg_struct, epoch)
        writer.add_scalar("Epoch/Avg_KL", avg_kl, epoch)
        writer.add_scalar("Epoch/Avg_SVR", avg_svr, epoch)

        # === CHECKPOINT (every epoch) ===
        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"rlaif_struct_epoch_{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": policy_model.state_dict(),
                "ema_state_dict": ema_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

        torch.cuda.empty_cache()

    writer.close()
    print("\nRLAIF Training Complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLAIF Structural Loss Training")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/num_dit_epoch_340.pt",
                        help="Path to pre-trained DiT checkpoint (reference model)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to RLAIF checkpoint to resume from")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (graphs per step)")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of Euler ODE steps")
    parser.add_argument("--grad-steps", type=int, default=3,
                        help="Number of LAST ODE steps to backprop through")
    parser.add_argument("--beta-kl", type=float, default=1.0,
                        help="KL penalty coefficient (MSE to reference velocity)")
    parser.add_argument("--beta-struct", type=float, default=0.5,
                        help="Structural loss coefficient")
    parser.add_argument("--beta-recon", type=float, default=0.5,
                        help="Reconstruction loss coefficient (MSE to target graph)")
    parser.add_argument("--kl-clamp", type=float, default=10.0,
                        help="Maximum KL loss value (clamp to prevent explosions)")
    parser.add_argument("--ema-decay", type=float, default=0.999,
                        help="EMA decay rate for model weights")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=10,
                        help="Evaluate discrete SVR every N batches")
    parser.add_argument("--max-epochs", type=int, default=5,
                        help="Maximum training epochs")
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    args = parser.parse_args()

    train_rlaif(args)
