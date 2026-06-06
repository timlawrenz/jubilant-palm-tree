"""
Evaluate RLAIF checkpoints beyond training loss.

For each checkpoint, generates graphs from random noise and measures:
  1. SVR (Structural Validation Rate) — % of graphs passing all 5 laws
  2. Per-law pass rates — which laws are hardest
  3. Graph statistics — edge count, node utilization, motif diversity
  4. Consistency — SVR across multiple random seeds
  5. Comparison of active weights vs EMA weights

Usage:
    python -m scripts.evaluate_checkpoints [--checkpoints-dir DIR] [--num-samples N]
"""

import argparse
import os
import json
import torch
import numpy as np
from collections import defaultdict

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.models.validation import GraphValidator
from src.rlaif.naive_discretizer import naive_discretize
from src.rlaif.reward import compute_reward


def load_model(state_dict, device):
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def generate_graphs(model, motifs, device, num_steps=20):
    """Run the ODE from noise to produce continuous output."""
    B, N = motifs.shape
    dt = 1.0 / num_steps
    x = torch.randn((B, 6, N, N), device=device)
    with torch.no_grad():
        for step in range(num_steps):
            t_val = step * dt
            t_tensor = torch.full((B,), t_val, device=device)
            v = model(x, t_tensor, motifs)
            x_cont = x[:, :2] + v[:, :2] * dt
            x_cat = v[:, 2:]
            x = torch.cat([x_cont, x_cat], dim=1)
    return x


def evaluate_checkpoint(ckpt_path, dataset, device, num_samples=200, seeds=(42, 123, 456)):
    """Evaluate a single checkpoint comprehensively."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    results = {"path": ckpt_path, "epoch": ckpt.get("epoch", "?")}
    weight_variants = {"active": ckpt["model_state_dict"]}
    if "ema_state_dict" in ckpt:
        weight_variants["ema"] = ckpt["ema_state_dict"]

    for variant_name, state_dict in weight_variants.items():
        model = load_model(state_dict, device)
        variant_results = {"seeds": {}}

        all_law_results = defaultdict(list)
        all_edge_counts = []
        all_node_counts = []

        for seed in seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Sample motifs from dataset
            indices = torch.randperm(len(dataset))[:num_samples]
            seed_law_results = defaultdict(int)
            seed_perfect = 0
            seed_total = 0

            batch_size = 8
            for start in range(0, num_samples, batch_size):
                end = min(start + batch_size, num_samples)
                batch_indices = indices[start:end]

                batch = [dataset[i] for i in batch_indices]
                motifs = torch.stack([b["motifs"] for b in batch]).to(device)
                B, N = motifs.shape

                x_final = generate_graphs(model, motifs, device)
                discrete = naive_discretize(x_final, motifs)

                # Per-law evaluation
                law_results = GraphValidator.evaluate_batch(discrete, motifs)
                for k, v in law_results.items():
                    seed_law_results[k] += v * B / 100.0

                # Graph statistics
                for i in range(B):
                    presence = discrete[i, 0]
                    valid = (motifs[i] != 0).sum().item()
                    edges = presence.sum().item()
                    all_edge_counts.append(edges)
                    all_node_counts.append(valid)

                seed_total += B

            # Normalize
            seed_results = {}
            for k, v in seed_law_results.items():
                pct = (v / seed_total) * 100.0
                seed_results[k] = round(pct, 1)
                all_law_results[k].append(pct)

            variant_results["seeds"][seed] = seed_results

        # Aggregate across seeds
        variant_results["avg"] = {}
        variant_results["std"] = {}
        for k in all_law_results:
            vals = all_law_results[k]
            variant_results["avg"][k] = round(np.mean(vals), 1)
            variant_results["std"][k] = round(np.std(vals), 1)

        variant_results["edge_stats"] = {
            "mean": round(np.mean(all_edge_counts), 1),
            "std": round(np.std(all_edge_counts), 1),
            "min": int(np.min(all_edge_counts)),
            "max": int(np.max(all_edge_counts)),
        }
        variant_results["node_stats"] = {
            "mean": round(np.mean(all_node_counts), 1),
        }

        del model
        torch.cuda.empty_cache()

        results[variant_name] = variant_results

    return results


def print_results(all_results):
    """Print a comparison table."""
    print("\n" + "=" * 90)
    print("CHECKPOINT EVALUATION RESULTS")
    print("=" * 90)

    # Header
    print(f"\n{'Epoch':<8} {'Weights':<8} {'SVR%':<8} {'±':<6} "
          f"{'OutDeg':<8} {'InDeg':<8} {'Orphan':<8} {'Acyclic':<8} {'Term':<8} "
          f"{'Edges':<12}")
    print("-" * 90)

    for r in all_results:
        epoch = r["epoch"]
        for variant in ["active", "ema"]:
            if variant not in r:
                continue
            v = r[variant]
            avg = v["avg"]
            std = v["std"]
            es = v["edge_stats"]
            svr = avg.get("perfect_graphs", 0)
            svr_std = std.get("perfect_graphs", 0)
            print(f"{epoch:<8} {variant:<8} {svr:<8.1f} {svr_std:<6.1f} "
                  f"{avg.get('out_degree_pass', 0):<8.1f} "
                  f"{avg.get('in_degree_pass', 0):<8.1f} "
                  f"{avg.get('no_orphan_pass', 0):<8.1f} "
                  f"{avg.get('acyclic_data_pass', 0):<8.1f} "
                  f"{avg.get('terminal_sink_pass', 0):<8.1f} "
                  f"{es['mean']:.0f}±{es['std']:.0f}")

    # Recommendation
    print("\n" + "=" * 90)
    best = None
    best_svr = -1
    best_std = 999
    for r in all_results:
        for variant in ["ema", "active"]:
            if variant not in r:
                continue
            svr = r[variant]["avg"].get("perfect_graphs", 0)
            std = r[variant]["std"].get("perfect_graphs", 0)
            if svr > best_svr or (svr == best_svr and std < best_std):
                best_svr = svr
                best_std = std
                best = (r["epoch"], variant, r["path"])

    if best:
        print(f"RECOMMENDED: Epoch {best[0]} ({best[1]} weights) — "
              f"SVR {best_svr:.1f}% ± {best_std:.1f}%")
        print(f"  Path: {best[2]}")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RLAIF checkpoints")
    parser.add_argument("--checkpoints-dir", type=str,
                        default="checkpoints/rlaif/vastai_run",
                        help="Directory containing checkpoint .pt files")
    parser.add_argument("--baseline", type=str,
                        default="checkpoints/num_dit_epoch_340.pt",
                        help="Pre-RLAIF baseline checkpoint")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="Number of graphs to generate per seed")
    parser.add_argument("--dataset", type=str,
                        default="dataset/compressed/compressed_motifs.jsonl")
    parser.add_argument("--epochs", type=str, default=None,
                        help="Comma-separated epochs to evaluate (default: all)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    dataset = ExecutionGraphDataset(jsonl_path=args.dataset, max_nodes=64)
    print(f"Dataset: {len(dataset)} graphs\n")

    all_results = []

    # Baseline
    if os.path.exists(args.baseline):
        print(f"Evaluating baseline: {args.baseline}")
        r = evaluate_checkpoint(args.baseline, dataset, device, args.num_samples)
        r["epoch"] = "base"
        all_results.append(r)

    # RLAIF checkpoints
    if args.epochs:
        epochs = [int(e) for e in args.epochs.split(",")]
    else:
        epochs = sorted([
            int(f.split("_")[-1].replace(".pt", ""))
            for f in os.listdir(args.checkpoints_dir)
            if f.startswith("rlaif_struct_epoch_") and f.endswith(".pt")
        ])

    for epoch in epochs:
        ckpt_path = os.path.join(args.checkpoints_dir, f"rlaif_struct_epoch_{epoch}.pt")
        if os.path.exists(ckpt_path):
            print(f"Evaluating epoch {epoch}: {ckpt_path}")
            r = evaluate_checkpoint(ckpt_path, dataset, device, args.num_samples)
            all_results.append(r)

    print_results(all_results)


if __name__ == "__main__":
    main()
