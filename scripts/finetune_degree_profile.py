"""Exp 1.9 — Training-time enrichment fine-tune (BoM Arm D).

Fine-tunes the 340-epoch pretrained DiT with per-node expected in-degree
scalars perturbing the motif embeddings in the conditioning path (in-channel,
rather than decode-time FiLM bias).

Two modes:
  signal  — degrees = EXPECTED_DATA_IN[motif] (real Arm-B signal)
  null    — degrees = uniform random in [0, 2.5] (same shape, meaningless)

Both fine-tune from checkpoints/num_dit_epoch_340.pt with identical settings.
Gate (pre-registered): PASS iff signal typed-F1 >= 0.20 AND >= 0.05 above
decode-best (0.109), reproducible across 2 seeds. Null must lose by >= 0.05.
"""

import argparse
import math
import random
import time

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW

from src.models.model import NeuralUniversalMachineDiT, build_expected_in_degrees
from src.models.dataset import ExecutionGraphDataset
from src.models.loss import compute_flow_matching_loss

CKPT_PATH = "checkpoints/num_dit_epoch_340.pt"
JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"


def make_degrees(motifs: torch.Tensor, mode: str, device: torch.device,
                 generator: torch.Generator | None = None) -> torch.Tensor:
    """Build per-node degree conditioning for the batch."""
    if mode == "signal":
        return build_expected_in_degrees(motifs).to(device)
    # null: uniform random in [0, 2.5) — plausible range, wrong values
    rng = generator or torch.Generator(device=device)
    return torch.rand(motifs.shape, generator=rng, device=device) * 2.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "null"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--physical_bs", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)   # effective bs 16
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_nodes", type=int, default=128)
    ap.add_argument("--out_dir", type=str, default="checkpoints/tte")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff_bs = args.physical_bs * args.grad_accum
    print(f"[tte:{args.mode}] seed={args.seed} device={device} "
          f"phys_bs={args.physical_bs} accum={args.grad_accum} eff_bs={eff_bs} "
          f"lr={args.lr} epochs={args.epochs} max_nodes={args.max_nodes}",
          flush=True)

    # Model with degree conditioning branch.
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12,
                                      use_degree=True, degree_dim=1).to(device)

    # Load pretrained backbone (degree_proj is fresh).
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not unexpected, f"Unexpected keys: {unexpected}"
    # Only the new degree branch may be missing.
    for k in missing:
        assert "degree_proj" in k, f"Unexpected missing key: {k}"
    print(f"Loaded {CKPT_PATH}; new keys initialized: {missing}", flush=True)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable / 1e6:.1f}M", flush=True)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Training set: permutation augmentation ON (as in pretraining). Degrees are
    # computed from the already-permuted motif row, so they stay aligned.
    dataset = ExecutionGraphDataset(
        jsonl_path=JSONL_PATH,
        max_nodes=args.max_nodes,
        augment_permutation=True,
    )
    dataloader = DataLoader(dataset, batch_size=args.physical_bs, shuffle=True,
                            drop_last=False, num_workers=0)

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"train_{args.mode}_s{args.seed}.log")
    logf = open(log_path, "a")
    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    gen = torch.Generator().manual_seed(args.seed)
    nan_guard = 0
    t0 = time.time()
    steps_per_epoch = max(1, len(dataset) // (args.physical_bs * args.grad_accum))
    log(f"Dataset: {len(dataset)} graphs, steps/epoch={steps_per_epoch}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        n_opt = 0
        for step, batch in enumerate(dataloader):
            for k in ["adjacency", "motifs", "padding_mask"]:
                batch[k] = batch[k].to(device)
            degrees = make_degrees(batch["motifs"], args.mode, device, generator=gen)

            loss = compute_flow_matching_loss(model, batch, degrees=degrees)

            # NaN guard (pretraining harness convention)
            if not torch.isfinite(loss):
                nan_guard += 1
                if nan_guard > 5:
                    log(f"STOP: non-finite loss ×{nan_guard} at epoch {epoch} step {step}")
                    return 1
                optimizer.zero_grad(set_to_none=True)
                continue

            (loss / args.grad_accum).backward()
            running += loss.item()
            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Reject NaN grads across all params.
                grad_ok = all(
                    torch.isfinite(p.grad).all() if p.grad is not None else True
                    for p in model.parameters()
                )
                if grad_ok:
                    optimizer.step()
                    n_opt += 1
                optimizer.zero_grad(set_to_none=True)

        avg = running / len(dataloader)
        el = time.time() - t0
        log(f"epoch {epoch}/{args.epochs} loss={avg:.4f} elapsed={el:.0f}s "
            f"({el / epoch:.0f}s/epoch) nan_guard={nan_guard}")

        # Save every epoch (small model).
        torch.save({
            "epoch": epoch,
            "mode": args.mode,
            "seed": args.seed,
            "model_state_dict": model.state_dict(),
            "loss": avg,
        }, os.path.join(args.out_dir, f"tte_{args.mode}_s{args.seed}_e{epoch}.pt"))

    logf.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())