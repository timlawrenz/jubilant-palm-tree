"""
Visualize the ODE denoising process step-by-step.

Produces 3 outputs showing how noise transforms into a valid execution graph:
  1. Adjacency heatmap filmstrip (PNG)
  2. Graph layout evolution (animated GIF)
  3. Sharpness curve (PNG)

Usage:
    python scripts/visualize_denoising.py [--checkpoint PATH] [--seed INT]
"""

import argparse
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import imageio.v2 as imageio
from collections import Counter
from io import BytesIO
from PIL import Image

from src.models.model import NeuralUniversalMachineDiT


# Motif type names and colors
MOTIF_NAMES = {0: "Pad", 1: "Boundary", 2: "Sequence", 3: "Condition",
               4: "Loop", 5: "State", 6: "Message"}
MOTIF_COLORS = {0: "#cccccc", 1: "#2ecc71", 2: "#3498db", 3: "#e74c3c",
                4: "#9b59b6", 5: "#f39c12", 6: "#1abc9c"}


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint (supports both base and RLAIF checkpoints)."""
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("ema_state_dict", ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state_dict)
    model.eval()
    return model


def find_complex_graph(jsonl_path: str, max_nodes: int = 128):
    """Find a graph with diverse motifs (Condition, Loop, Message) and 8-20 nodes."""
    with open(jsonl_path) as f:
        graphs = [json.loads(l)["compressed_graph"] for l in f if l.strip()]

    best, best_score = None, 0
    for g in graphs:
        n = len(g["nodes"])
        if not (8 <= n <= 20) or n > max_nodes:
            continue
        motifs = [node["motif"] for node in g["nodes"]]
        c = Counter(motifs)
        score = (len(c) * 10 + c.get("Condition", 0) * 5 +
                 c.get("Loop", 0) * 5 + c.get("Message", 0) * 3)
        if score > best_score:
            best_score = score
            best = g
    return best


def prepare_motifs_tensor(graph: dict, max_nodes: int, device: torch.device):
    """Convert graph to motifs tensor for model conditioning."""
    motif_map = {"Boundary": 1, "Sequence": 2, "Condition": 3,
                 "Loop": 4, "State": 5, "Message": 6}
    motifs = torch.zeros(max_nodes, dtype=torch.long, device=device)
    for i, node in enumerate(graph["nodes"]):
        motifs[i] = motif_map.get(node["motif"], 0)
    return motifs.unsqueeze(0)  # [1, N]


@torch.no_grad()
def run_ode_with_intermediates(model, motifs, num_steps=20, device="cuda", seed=42):
    """Run ODE and capture state at every step."""
    torch.manual_seed(seed)
    B = 1
    N = motifs.shape[1]
    dt = 1.0 / num_steps

    x = torch.randn((B, 6, N, N), device=device)
    intermediates = [x.cpu().clone()]

    for step in range(num_steps):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=device)
        v = model(x, t_tensor, motifs)
        x_cont = x[:, :2] + v[:, :2] * dt
        x_cat = v[:, 2:]
        x = torch.cat([x_cont, x_cat], dim=1)
        intermediates.append(x.cpu().clone())

    return intermediates  # List of [1, 6, N, N] tensors, length num_steps+1


def compute_sharpness(intermediates, num_valid_nodes: int):
    """Compute metrics at each step: sharpness, edge count, and structure score."""
    sharpness = []
    edge_counts = []
    for x in intermediates:
        presence = torch.sigmoid(x[0, 0, :num_valid_nodes, :num_valid_nodes])
        s = (presence * (1.0 - presence)).mean().item()
        sharpness.append(s)
        edge_counts.append((presence > 0.5).sum().item())
    return sharpness, edge_counts


def render_heatmap_filmstrip(intermediates, num_valid_nodes: int, output_path: str):
    """Render adjacency heatmap at selected steps + discretized final."""
    steps_to_show = [0, 5, 10, 15, 19, 20]  # 20 = discretized
    labels = ["Step 0\n(noise)", "Step 5", "Step 10", "Step 15",
              "Step 19", "Step 20\n(discretized)"]

    fig, axes = plt.subplots(1, len(steps_to_show), figsize=(18, 3.5))
    fig.suptitle("Denoising Progression: Presence Channel", fontsize=14, y=1.02)

    # Diverging colormap: blue (0) → white (0.5) → red (1)
    cmap = plt.cm.RdBu_r

    for i, (step_idx, label) in enumerate(zip(steps_to_show, labels)):
        ax = axes[i]

        if step_idx == 20:
            # Discretized version of final step
            x = intermediates[-1]
            presence = (torch.sigmoid(x[0, 0, :num_valid_nodes, :num_valid_nodes]) > 0.5).float()
            im = ax.imshow(presence.numpy(), cmap='binary', vmin=0, vmax=1,
                           interpolation='nearest', aspect='equal')
        else:
            x = intermediates[step_idx]
            # Show raw logits for better visual contrast
            logits = x[0, 0, :num_valid_nodes, :num_valid_nodes]
            vmax = max(abs(logits.min().item()), abs(logits.max().item()), 1.0)
            im = ax.imshow(logits.numpy(), cmap=cmap, vmin=-vmax, vmax=vmax,
                           interpolation='nearest', aspect='equal')
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(im, ax=axes[-1], shrink=0.8, label="Logit Magnitude / Edge Prob")
    fig.tight_layout(rect=[0, 0, 0.95, 0.95])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Heatmap filmstrip: {output_path}")


def render_graph_gif(intermediates, motifs_1d, num_valid_nodes: int, output_path: str):
    """Render animated GIF of graph layout evolution."""
    # Compute a fixed layout from the final graph
    final_x = intermediates[-1]
    final_presence = torch.sigmoid(final_x[0, 0, :num_valid_nodes, :num_valid_nodes])
    final_adj = (final_presence > 0.5).numpy()

    G_final = nx.DiGraph()
    G_final.add_nodes_from(range(num_valid_nodes))
    for i in range(num_valid_nodes):
        for j in range(num_valid_nodes):
            if final_adj[i, j]:
                G_final.add_edge(i, j)

    # Fixed layout for all frames
    pos = nx.spring_layout(G_final, seed=42, k=2.0/np.sqrt(num_valid_nodes))

    node_colors = [MOTIF_COLORS.get(motifs_1d[i].item(), "#cccccc")
                   for i in range(num_valid_nodes)]

    frames = []
    steps_to_render = list(range(0, 21))  # All 21 states

    for step_idx in steps_to_render:
        x = intermediates[step_idx]
        presence = torch.sigmoid(x[0, 0, :num_valid_nodes, :num_valid_nodes])
        edge_type = torch.sigmoid(x[0, 1, :num_valid_nodes, :num_valid_nodes])

        # Threshold at 0.5
        adj = (presence > 0.5).numpy()

        G = nx.DiGraph()
        G.add_nodes_from(range(num_valid_nodes))
        exec_edges = []
        data_edges = []

        for i in range(num_valid_nodes):
            for j in range(num_valid_nodes):
                if adj[i, j]:
                    if edge_type[i, j].item() < 0.5:
                        exec_edges.append((i, j))
                    else:
                        data_edges.append((i, j))

        G.add_edges_from(exec_edges + data_edges)

        # Render frame
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.set_title(f"Step {step_idx}/20" + (" (discretized)" if step_idx == 20 else ""),
                     fontsize=12, fontweight='bold')

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=200, edgecolors='black', linewidths=0.5)
        if exec_edges:
            nx.draw_networkx_edges(G, pos, edgelist=exec_edges, ax=ax,
                                   edge_color='#2980b9', width=1.5,
                                   arrows=True, arrowsize=10, alpha=0.8)
        if data_edges:
            nx.draw_networkx_edges(G, pos, edgelist=data_edges, ax=ax,
                                   edge_color='#e74c3c', width=1.0,
                                   arrows=True, arrowsize=8, alpha=0.6,
                                   style='dashed')

        # Legend
        ax.text(0.02, 0.02, f"Edges: {len(exec_edges)} exec (blue) + {len(data_edges)} data (red)",
                transform=ax.transAxes, fontsize=8, verticalalignment='bottom')
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')

        # Convert to image
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        frame = np.array(Image.open(buf))
        frames.append(frame)

    # Save GIF (0.5s per frame, hold last frame longer)
    durations = [0.5] * (len(frames) - 1) + [2.0]
    imageio.mimsave(output_path, frames, duration=durations, loop=0)
    print(f"  Graph evolution GIF: {output_path}")


def render_sharpness_curve(sharpness: list, edge_counts: list, num_valid_nodes: int,
                           output_path: str):
    """Plot crystallization metrics over ODE steps."""
    steps = list(range(len(sharpness)))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle("Denoising Dynamics", fontsize=13)

    # Top: Edge count evolution
    ax1.plot(steps, edge_counts, 'b-o', linewidth=2, markersize=5)
    ax1.axhline(y=num_valid_nodes, color='green', linestyle='--', alpha=0.5,
                label=f'Expected ~{num_valid_nodes} edges')
    ax1.set_ylabel("Edges (presence > 0.5)", fontsize=10)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Bottom: Sharpness
    ax2.plot(steps, sharpness, 'r-o', linewidth=2, markersize=5)
    ax2.axhline(y=0.25, color='grey', linestyle='--', alpha=0.5, label='Max uncertainty')
    ax2.set_xlabel("ODE Step", fontsize=11)
    ax2.set_ylabel("Avg Sharpness p·(1-p)", fontsize=10)
    ax2.legend(loc='upper right')
    ax2.set_xlim(-0.5, len(sharpness) - 0.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Metrics curve: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize ODE denoising process")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/rlaif/rlaif_struct_epoch_5.pt",
                        help="Model checkpoint to use")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible noise")
    parser.add_argument("--output-dir", type=str, default="output/visualizations",
                        help="Output directory for visualizations")
    parser.add_argument("--num-steps", type=int, default=20,
                        help="Number of ODE steps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Denoising Visualization ===")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Seed: {args.seed}")

    # Load model
    print("\nLoading model...")
    model = load_model(args.checkpoint, device)

    # Find a complex graph for conditioning
    print("Selecting complex graph for conditioning...")
    graph = find_complex_graph("dataset/compressed/compressed_motifs.jsonl")
    num_valid = len(graph["nodes"])
    motif_types = [n["motif"] for n in graph["nodes"]]
    print(f"  Selected: {num_valid} nodes — {Counter(motif_types)}")

    # Prepare conditioning
    motifs = prepare_motifs_tensor(graph, max_nodes=128, device=device)

    # Run ODE with intermediates
    print(f"\nRunning {args.num_steps}-step ODE...")
    intermediates = run_ode_with_intermediates(
        model, motifs, num_steps=args.num_steps, device=device, seed=args.seed)
    print(f"  Captured {len(intermediates)} states (step 0 = noise, step 20 = final)")

    # Compute metrics
    sharpness, edge_counts = compute_sharpness(intermediates, num_valid)
    print(f"  Sharpness: {sharpness[0]:.4f} (noise) → {sharpness[-1]:.4f} (final)")
    print(f"  Edge count: {edge_counts[0]} (noise) → {edge_counts[-1]} (final)")

    # Render outputs
    print("\nRendering visualizations...")
    render_heatmap_filmstrip(
        intermediates, num_valid,
        os.path.join(args.output_dir, "denoising_heatmap_filmstrip.png"))

    render_graph_gif(
        intermediates, motifs[0], num_valid,
        os.path.join(args.output_dir, "denoising_graph_evolution.gif"))

    render_sharpness_curve(
        sharpness, edge_counts, num_valid,
        os.path.join(args.output_dir, "denoising_metrics_curve.png"))

    print("\n✓ All visualizations saved to:", args.output_dir)


if __name__ == "__main__":
    main()
