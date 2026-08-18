"""Exp 3 — Trainer for the autoregressive edge-list generator.

Teacher-forced next-token CE over BODY tokens only (prefix motif tokens are
conditioning; their positions are masked from the loss). Fresh full training
run on the same 3,951-graph corpus minus the pre-registered held-out split.

Usage:
  python scripts/train_ar_edge_list.py --epochs 30 --seed 0 \
      --out checkpoints/ar --run_tag ar_s0
"""

import argparse
import os
import time
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from src.models.ar_edge_list import (ARGraphDataset, EdgeListDecoder, VOCAB_SIZE,
                                     PAD_ID, EOS_ID, MAX_SEQ, collate_ar)
from torch.optim import AdamW

JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"


def build_loss_mask(tokens: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """True = predict this position (body tokens, i.e. after the prefix).

    Prefix = sequence of MOTIF tokens (vocab >= MOTIF_BASE) at the head.
    A position p in [prefix_len.. ] whose *next* token is not PAD is a body
    position. We predict every body token (before EOS).
    """
    # body starts at the first non-motif token (the first edge src/dst/... token)
    B, S = tokens.shape
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    for b in range(B):
        i = 0
        while i < S and tokens[b, i] >= 134:   # motif tokens
            i += 1
        # from position i onward (body), predict everything up to (not incl.) PAD
        while i < S and tokens[b, i] != PAD_ID:
            mask[b, i] = valid[b, i]
            i += 1
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", type=str, default="checkpoints/ar")
    ap.add_argument("--run", type=str, default="ar_s0")
    ap.add_argument("--smoke", action="store_true", help="tiny subset for CPU test")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AR-train] run={args.run} device={device} epochs={args.epochs} "
          f"bs={args.batch_size} lr={args.lr} d={args.d_model} L={args.n_layers}", flush=True)

    dataset = ARGraphDataset(jsonl_path=JSONL_PATH, max_nodes=128,
                             augment_permutation=False,   # canonical order == eval
                             eval_only=False)
    if args.smoke:
        dataset.graphs = dataset.graphs[:64]
        dataset.augment_permutation = False
        print("[AR-train] SMOKE MODE: 64 graphs", flush=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_ar, num_workers=0)

    model = EdgeListDecoder(d_model=args.d_model, n_heads=args.n_heads,
                            n_layers=args.n_layers, vocab=VOCAB_SIZE,
                            max_len=MAX_SEQ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[AR-train] params={n_params/1e6:.2f}M", flush=True)

    # slower LR for the loss surface; cosine schedule
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    os.makedirs(args.out_dir, exist_ok=True)
    logf = open(os.path.join(args.out_dir, f"{args.run}.log"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    autocast_ctx = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda" else nullcontext())
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        n_tok = 0
        for step, (tokens, valid, num_nodes) in enumerate(loader):
            tokens, valid = tokens.to(device), valid.to(device)
            with autocast_ctx:
                logits = model(tokens, valid)                     # [B, S, Vocab]
            mask = build_loss_mask(tokens, valid).to(device)   # [B, S]
            targets = tokens.clone()
            # shift: predict token p+1 from position p
            logits = logits[:, :-1, :]
            targets = targets[:, 1:]
            mask = mask[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1),
                reduction="none")
            loss = (loss * mask.float().reshape(-1)).sum() / max(
                mask.sum().item(), 1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * max(mask.sum().item(), 1)
            n_tok += mask.sum().item()
        avg = total / max(n_tok, 1)
        el = time.time() - t0
        log(f"epoch {epoch:02d}/{args.epochs} loss={avg:.4f} elapsed={el:.0f}s "
            f"({el/epoch:.0f}s/epoch)")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "loss": avg,
            "run": args.run,
        }, os.path.join(args.out_dir, f"{args.run}_e{epoch:02d}.pt"))
        if args.smoke and epoch >= 1:
            break
    logf.close()
    print("done", flush=True)


if __name__ == "__main__":
    main()