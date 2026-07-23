"""Task 3: Routing-fidelity evaluation.

Over N>=512 real ground-truth graphs, condition the DiT on each graph's own
motif sequence, generate via the canonical 20-step Euler ODE + ConstraintSolver,
and score edge-set fidelity against the source graph. A random matched-density
baseline (same number of edges scattered uniformly over the node block) serves
as the null hypothesis.

PRE-REGISTERED GUARDRAIL: node-permutation augmentation must be OFF, otherwise
per-graph edge fidelity against the source graph is meaningless.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.models.constraint_solver import ConstraintSolver
from src.models.fidelity import edge_fidelity

CKPT_PATH = "checkpoints/num_dit_epoch_340.pt"
JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_GRAPHS_TARGET = 512
BATCH_SIZE = 8
NUM_STEPS = 20


def build_random_baseline(gt_adj: torch.Tensor, num_nodes: int, k: int,
                          generator: torch.Generator) -> torch.Tensor:
    """Random matched-density baseline.

    Scatter `k` edges uniformly at random over distinct (i, j) positions in the
    [num_nodes x num_nodes] block with i != j. Edge types are sampled 0/1 with
    probability matching the ground-truth block's exec/data ratio.
    """
    N = gt_adj.shape[-1]
    rand_adj = torch.zeros((3, N, N), dtype=gt_adj.dtype)

    if k <= 0 or num_nodes < 2:
        return rand_adj

    # Ground-truth exec/data ratio over present edges in the valid block.
    gt_block_presence = gt_adj[0, :num_nodes, :num_nodes]
    gt_block_type = gt_adj[1, :num_nodes, :num_nodes]
    present = gt_block_presence > 0.5
    num_present = int(present.sum().item())
    if num_present > 0:
        data_ratio = float(gt_block_type[present].mean().item())  # type 1 == data
    else:
        data_ratio = 0.5

    # All distinct off-diagonal (i, j) positions in the block.
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Routing-fidelity evaluation on device: {device}")

    # Load model
    print(f"Loading checkpoint: {CKPT_PATH}")
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Dataset: NO permutation augmentation — fidelity is per-graph vs source.
    dataset = ExecutionGraphDataset(
        jsonl_path=JSONL_PATH,
        max_nodes=128,
        augment_permutation=False,
    )
    # PRE-REGISTERED GUARDRAIL — do not remove.
    assert dataset.augment_permutation == False, \
        "Fidelity is meaningless under node permutation"

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    gen_untyped_f1, gen_typed_f1 = [], []
    rand_untyped_f1, rand_typed_f1 = [], []
    gen_edges_all, gt_edges_all = [], []
    num_scored = 0
    num_errors = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_scored >= NUM_GRAPHS_TARGET:
                break

            motifs = batch["motifs"].to(device)  # [B, N]
            B, N = motifs.shape

            try:
                # Canonical 20-step Euler ODE sampling loop.
                x = torch.randn((B, 6, N, N), device=device)
                dt = 1.0 / NUM_STEPS
                for step in range(NUM_STEPS):
                    t_val = step * dt
                    t_tensor = torch.full((B,), t_val, device=device)
                    v = model(x, t_tensor, motifs)
                    x_cont = x[:, :2] + v[:, :2] * dt
                    x_cat = v[:, 2:]
                    x = torch.cat([x_cont, x_cat], dim=1)
                discrete = ConstraintSolver.discretize_and_repair(x, motifs)
            except Exception as e:
                print(f"  [batch error] generation failed ({e}); skipping {B} graphs")
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
                    gt = batch["adjacency"][b]          # [3, N, N] cpu
                    gen = discrete[b].cpu()             # [3, N, N]

                    m = edge_fidelity(gen, gt, num_nodes)

                    # Random matched-density baseline, seeded per graph index.
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
                    if num_scored % 8 == 0:
                        print(f"Scored {num_scored}/{NUM_GRAPHS_TARGET} "
                              f"(errors: {num_errors}) ...")
                except Exception as e:
                    print(f"  [graph error] index {num_scored}: {e}")
                    num_errors += 1
                    continue

    gen_typed = np.array(gen_typed_f1)
    quantiles = (np.min(gen_typed), np.percentile(gen_typed, 25),
                 np.median(gen_typed), np.percentile(gen_typed, 75),
                 np.max(gen_typed)) if len(gen_typed) else (0, 0, 0, 0, 0)

    print("\n" + "=" * 50)
    print("ROUTING FIDELITY RESULTS")
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
