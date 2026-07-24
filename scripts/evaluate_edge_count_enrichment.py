"""Task 1.6: Evaluate edge-count enrichment.

Same harness as Exp 1.5 (512 graphs, augment_permutation=False, 20-step ODE),
but the ODE loop receives a FiLM-style density bias on the presence channel,
nudging the model toward the target edge count at each step.

Key addition: density_bias_scale sweep to find the sweet spot. Too low = no effect;
too high = model erases edges (like RLAIF mode collapse). The scale is the only
new hyperparameter to tune.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.model import NeuralUniversalMachineDiT, ode_generate_with_density_bias
from src.models.dataset import ExecutionGraphDataset
from src.models.fidelity import edge_fidelity

CKPT_PATH = "checkpoints/num_dit_epoch_340.pt"
JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_GRAPHS = 512
BATCH_SIZE = 8
NUM_STEPS = 20

# Density bias scales to sweep (0 = baseline, same as Exp 1.5)
DENSITY_SCALES = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]


def build_random_baseline(gt_adj, num_nodes, k, generator):
    N = gt_adj.shape[-1]
    rand_adj = torch.zeros((3, N, N), dtype=gt_adj.dtype)
    if k <= 0 or num_nodes < 2:
        return rand_adj
    gt_block = gt_adj[:2, :num_nodes, :num_nodes]
    present = gt_block[0] > 0.5
    num_present = int(present.sum())
    data_ratio = float(gt_block[1, present].mean()) if num_present > 0 else 0.5
    positions = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    k = min(k, len(positions))
    perm = torch.randperm(len(positions), generator=generator)[:k].tolist()
    draws = torch.rand(k, generator=generator)
    for idx, pos_idx in enumerate(perm):
        i, j = positions[pos_idx]
        rand_adj[0, i, j] = 1.0
        rand_adj[1, i, j] = 1.0 if draws[idx] < data_ratio else 0.0
    return rand_adj


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Edge-count enrichment eval on device: {device}")

    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = ExecutionGraphDataset(jsonl_path=JSONL_PATH, max_nodes=128,
                                     augment_permutation=False)
    assert dataset.augment_permutation == False
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Results per scale
    results = {s: {"typed_f1": [], "untyped_f1": [], "gen_edges": [], "gt_edges": []}
               for s in DENSITY_SCALES}
    results["random"] = {"typed_f1": [], "untyped_f1": []}
    num_scored = 0

    with torch.no_grad():
        for batch in dataloader:
            if num_scored >= NUM_GRAPHS:
                break
            motifs = batch["motifs"].to(device)
            B, N = motifs.shape

            for b in range(B):
                if num_scored >= NUM_GRAPHS:
                    break
                try:
                    num_nodes = int((motifs[b] != 0).sum())
                    if num_nodes == 0:
                        continue
                    gt = batch["adjacency"][b]
                    k = int(gt[0, :num_nodes, :num_nodes].sum())

                    # Random baseline (once per graph)
                    rng = torch.Generator().manual_seed(num_scored)
                    rand_adj = build_random_baseline(gt, num_nodes, k, rng)
                    r = edge_fidelity(rand_adj, gt, num_nodes)
                    results["random"]["typed_f1"].append(r["typed_f1"])
                    results["random"]["untyped_f1"].append(r["untyped_f1"])

                    # For each density scale, generate with biased ODE
                    mot_b = motifs[b].unsqueeze(0)
                    density_val = float(k) / (num_nodes * max(1, num_nodes - 1))

                    for scale in DENSITY_SCALES:
                        ed = torch.tensor([density_val], device=device)
                        if scale == 0.0:
                            # Vanilla ODE (no bias) — matches Exp 1.5
                            from src.models.constraint_solver import ConstraintSolver
                            x = torch.randn((1, 6, N, N), device=device)
                            dt_val = 1.0 / NUM_STEPS
                            for step in range(NUM_STEPS):
                                tv = torch.full((1,), step * dt_val, device=device)
                                v = model(x, tv, mot_b)
                                x = torch.cat([x[:, :2] + v[:, :2] * dt_val, v[:, 2:]], dim=1)
                            disc = ConstraintSolver.discretize_and_repair(x, mot_b)
                        else:
                            disc = ode_generate_with_density_bias(
                                model, mot_b, ed, device,
                                num_steps=NUM_STEPS, density_bias_scale=scale)

                        gen = disc[0].cpu()
                        m = edge_fidelity(gen, gt, num_nodes)
                        results[scale]["typed_f1"].append(m["typed_f1"])
                        results[scale]["untyped_f1"].append(m["untyped_f1"])
                        results[scale]["gen_edges"].append(m["gen_edges"])
                        results[scale]["gt_edges"].append(m["gt_edges"])

                    num_scored += 1
                    if num_scored % 64 == 0:
                        print(f"Scored {num_scored}/{NUM_GRAPHS} ...")
                except Exception as e:
                    print(f"  error at graph {num_scored}: {e}")
                    continue

    # Report
    print("\n" + "=" * 70)
    print("EDGE-COUNT ENRICHMENT RESULTS (per density_bias_scale)")
    print("=" * 70)
    print(f"Graphs scored: {num_scored}")
    print(f"{'Scale':>6}  {'typed_F1':>10}  {'untyped_F1':>12}  {'gen_edges':>10}  {'gt_edges':>9}")
    print("-" * 70)
    for scale in DENSITY_SCALES:
        tf = np.mean(results[scale]["typed_f1"])
        uf = np.mean(results[scale]["untyped_f1"])
        ge = np.mean(results[scale]["gen_edges"])
        gt_mean = np.mean(results[scale]["gt_edges"])
        print(f"{scale:>6.2f}  {tf:>10.4f}  {uf:>12.4f}  {ge:>10.1f}  {gt_mean:>9.1f}")

    # Baseline row
    rtf = np.mean(results["random"]["typed_f1"])
    ruf = np.mean(results["random"]["untyped_f1"])
    print("-" * 70)
    print(f"{'random':>6}  {rtf:>10.4f}  {ruf:>12.4f}  {'—':>10}  {'—':>9}")
    print("=" * 70)

    # Best scale details
    best_scale = max(DENSITY_SCALES,
                     key=lambda s: np.mean(results[s]["typed_f1"]))
    best_tf = np.array(results[best_scale]["typed_f1"])
    print(f"\nBest scale: {best_scale:.2f} (typed-F1 = {best_tf.mean():.4f})")
    print(f"  typed-F1 quantiles: {np.min(best_tf):.4f} / "
          f"{np.percentile(best_tf,25):.4f} / {np.median(best_tf):.4f} / "
          f"{np.percentile(best_tf,75):.4f} / {np.max(best_tf):.4f}")


if __name__ == "__main__":
    main()
