"""Exp 1.9 — Routing-fidelity evaluation for training-time enrichment.

Identical protocol to evaluate_routing_fidelity.py (N=512, augment_permutation
must be OFF, random matched-density baseline) but loads a fine-tuned checkpoint
with the degree-conditioning branch and feeds degrees through the ODE loop.

Usage:
  python scripts/evaluate_tte.py --mode signal --ckpt checkpoints/tte/tte_signal_s0_e20.pt
  python scripts/evaluate_tte.py --mode null   --ckpt checkpoints/tte/tte_null_s0_e20.pt
"""

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.model import NeuralUniversalMachineDiT, build_expected_in_degrees
from src.models.dataset import ExecutionGraphDataset
from src.models.constraint_solver import ConstraintSolver
from src.models.fidelity import edge_fidelity

JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_GRAPHS_TARGET = 512
BATCH_SIZE = 8
NUM_STEPS = 20


def build_random_baseline(gt_adj: torch.Tensor, num_nodes: int, k: int,
                          generator: torch.Generator) -> torch.Tensor:
    """Random matched-density baseline (identical to evaluate_routing_fidelity)."""
    N = gt_adj.shape[-1]
    rand_adj = torch.zeros((3, N, N), dtype=gt_adj.dtype)

    if k <= 0 or num_nodes < 2:
        return rand_adj

    gt_block_presence = gt_adj[0, :num_nodes, :num_nodes]
    gt_block_type = gt_adj[1, :num_nodes, :num_nodes]
    present = gt_block_presence > 0.5
    num_present = int(present.sum().item())
    if num_present > 0:
        data_ratio = float(gt_block_type[present].mean().item())
    else:
        data_ratio = 0.5

    positions = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    num_positions = len(positions)
    k = min(k, num_positions)

    perm = torch.randperm(num_positions, generator=generator)[:k].tolist()
    type_draws = torch.rand(k, generator=generator)

    for idx, pos_idx in enumerate(perm):
        i, j = positions[pos_idx]
        rand_adj[0, i, j] = 1.0
        rand_adj[1, i, j] = 1.0 if type_draws[idx].item() < data_ratio else 0.0

    return rand_adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["signal", "null"], required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[tte-eval:{args.mode}] device={device} ckpt={args.ckpt}", flush=True)

    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12,
                                      use_degree=True, degree_dim=1).to(device)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch')}, mode {ckpt.get('mode')})", flush=True)

    dataset = ExecutionGraphDataset(
        jsonl_path=JSONL_PATH,
        max_nodes=128,
        augment_permutation=False,
    )
    # PRE-REGISTERED GUARDRAIL — do not remove.
    assert dataset.augment_permutation == False, \
        "Fidelity is meaningless under node permutation"

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    gen = torch.Generator().manual_seed(args.seed)

    gen_untyped_f1, gen_typed_f1 = [], []
    rand_untyped_f1, rand_typed_f1 = [], []
    gen_edges_all, gt_edges_all = [], []
    num_scored = 0
    num_errors = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_scored >= NUM_GRAPHS_TARGET:
                break

            motifs = batch["motifs"].to(device)
            B, N = motifs.shape

            # Degrees follow the arm's training distribution.
            if args.mode == "signal":
                degrees = build_expected_in_degrees(motifs).to(device)
            else:
                degrees = (torch.rand(B, N, generator=gen, device=device) * 2.5)

            try:
                x = torch.randn((B, 6, N, N), generator=gen, device=device)
                dt = 1.0 / NUM_STEPS
                for step in range(NUM_STEPS):
                    t_val = step * dt
                    t_tensor = torch.full((B,), t_val, device=device)
                    v = model(x, t_tensor, motifs, degrees=degrees)
                    x_cont = x[:, :2] + v[:, :2] * dt
                    x_cat = v[:, 2:]
                    x = torch.cat([x_cont, x_cat], dim=1)
                discrete = ConstraintSolver.discretize_and_repair(x, motifs)
            except Exception as e:
                print(f"  [batch error] generation failed ({e}); skipping {B} graphs", flush=True)
                num_errors += B
                continue

            for b in range(B):
                if num_scored >= NUM_GRAPHS_TARGET:
                    break
                try:
                    num_nodes = int((motifs[b] != 0).sum().item())
                    if num_nodes == 0:
                        num_errors += 1
                        continue
                    gt = batch["adjacency"][b]
                    gen = discrete[b].cpu()

                    m = edge_fidelity(gen, gt, num_nodes)

                    rng = torch.Generator().manual_seed(num_scored)
                    rand_adj = build_random_baseline(gt, num_nodes, m["gt_edges"], rng)
                    m_rand = edge_fidelity(rand_adj, gt, num_nodes)

                    gen_untyped_f1.append(m["untyped_f1"])
                    gen_typed_f1.append(m["typed_f1"])
                    rand_untyped_f1.append(m_rand["untyped_f1"])
                    rand_typed_f1.append(m_rand["typed_f1"])
                    gen_edges_all.append(m["gen_edges"])
                    gt_edges_all.append(m["gt_edges"])

                    num_scored += 1
                except Exception as e:
                    print(f"  [graph error] index {num_scored}: {e}", flush=True)
                    num_errors += 1
                    continue

    gen_typed = np.array(gen_typed_f1)
    quantiles = (np.min(gen_typed), np.percentile(gen_typed, 25),
                 np.median(gen_typed), np.percentile(gen_typed, 75),
                 np.max(gen_typed)) if len(gen_typed) else (0, 0, 0, 0, 0)

    print("\n" + "=" * 50)
    print(f"ROUTING FIDELITY — {args.mode} ARM ({args.ckpt.split('/')[-1]})")
    print("=" * 50)
    print(f"Total graphs scored : {num_scored} (errors skipped: {num_errors})")
    print(f"Mean untyped_f1     : {np.mean(gen_untyped_f1):.4f}")
    print(f"Mean typed_f1       : {np.mean(gen_typed_f1):.4f}")
    print(f"Mean random untyped : {np.mean(rand_untyped_f1):.4f}")
    print(f"Mean random typed   : {np.mean(rand_typed_f1):.4f}")
    print(f"Mean gen_edges      : {np.mean(gen_edges_all):.4f}")
    print(f"Mean gt_edges       : {np.mean(gt_edges_all):.4f}")
    print("-" * 50)
    print("typed_f1 quantiles (min / p25 / median / p75 / max):")
    print(f"  {quantiles[0]:.4f} / {quantiles[1]:.4f} / {quantiles[2]:.4f} "
          f"/ {quantiles[3]:.4f} / {quantiles[4]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()