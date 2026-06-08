import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.models.constraint_solver import ConstraintSolver
from src.models.validation import GraphValidator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Baseline SVR on {device}...")

    # Load Model
    ckpt_path = "checkpoints/num_dit_epoch_340.pt"
    print(f"Loading {ckpt_path}")
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load Dataset (for motif sequences and padding masks)
    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=False,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # We will evaluate a subset of batches
    num_batches_to_eval = 5
    
    metrics_accum = {
        "out_degree_pass": [], "in_degree_pass": [], "no_orphan_pass": [],
        "acyclic_data_pass": [], "terminal_sink_pass": [], "perfect_graphs": []
    }
    edge_counts = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_eval:
                break
                
            motifs = batch["motifs"].to(device)
            B, N = motifs.shape
            
            # Generate from noise
            x = torch.randn((B, 6, N, N), device=device)
            num_steps = 20
            dt = 1.0 / num_steps
            
            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v = model(x, t_tensor, motifs)
                x_cont = x[:, :2] + v[:, :2] * dt
                x_cat = v[:, 2:]
                x = torch.cat([x_cont, x_cat], dim=1)
                
            # Discretize (Constraint Solver)
            discrete = ConstraintSolver.discretize_and_repair(x, motifs)
            
            # Count edges
            presence = discrete[:, 0] # [B, N, N]
            edge_counts.extend(presence.sum(dim=(1,2)).cpu().tolist())
            
            # Evaluate
            val_results = GraphValidator.evaluate_batch(discrete, motifs)
            
            for k in metrics_accum.keys():
                metrics_accum[k].append(val_results[k])
                
            print(f"Batch {i+1}/{num_batches_to_eval} - SVR: {val_results['perfect_graphs']:.1f}%")

    # Aggregate and Print
    print("\n" + "="*50)
    print("REPAIRED EVALUATION RESULTS (Pre-trained DiT + Constraint Solver)")
    print("="*50)
    print(f"Total Samples: {num_batches_to_eval * 32}")
    for k, v in metrics_accum.items():
        avg = np.mean(v)
        print(f"{k}: {avg:.2f}%")
    print(f"Average Edge Count: {np.mean(edge_counts):.1f}")
    print("="*50)

if __name__ == "__main__":
    main()
