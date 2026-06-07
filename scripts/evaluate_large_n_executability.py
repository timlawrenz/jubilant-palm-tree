import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.models.constraint_solver import ConstraintSolver
from src.models.validation import GraphValidator
from src.execution_engine.dummy_custodian import DummySemanticCustodian
from src.execution_engine.interpreter import GraphInterpreter

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Large-N Executability on {device}...")

    # Load Model
    ckpt_path = "checkpoints/num_dit_epoch_340.pt"
    print(f"Loading {ckpt_path}")
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=False,
    )
    # 32 batches of 32 = 1024 samples
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    num_batches_to_eval = 32
    
    metrics = {"halted": 0, "infinite_loop": 0, "error_no_entry": 0, "error_unexpected_sink": 0, "crash": 0}
    perfect_graphs_found = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_eval:
                break
                
            motifs = batch["motifs"].to(device)
            B, N = motifs.shape
            total_samples += B
            
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
                
            discrete = ConstraintSolver.discretize_and_repair(x, motifs)
            
            # evaluate_batch returns percentages for the whole batch. 
            # We need to filter per graph to find the perfect ones.
            for b in range(B):
                # Isolate single graph
                adj_b = discrete[b].unsqueeze(0)
                motifs_b = motifs[b].unsqueeze(0)
                res = GraphValidator.evaluate_batch(adj_b, motifs_b)
                if res["perfect_graphs"] == 100.0:
                    perfect_graphs_found += 1
                    
                    # Convert to execution schema via dummy custodian
                    try:
                        exec_graph = DummySemanticCustodian.tensor_to_graph(adj_b[0], motifs_b[0])
                        interpreter = GraphInterpreter(exec_graph)
                        result = interpreter.run_with_limit(max_steps=1000)
                        status = result["status"]
                        if status in metrics:
                            metrics[status] += 1
                        else:
                            metrics["crash"] += 1
                    except Exception as e:
                        metrics["crash"] += 1
                        
            print(f"Batch {i+1}/{num_batches_to_eval} - Found {perfect_graphs_found} perfect 6-Law graphs so far...")

    print("\n" + "="*50)
    print("LARGE-N EXECUTABILITY AUDIT RESULTS")
    print("="*50)
    print(f"Total Graphs Generated: {total_samples}")
    print(f"Total Structurally Perfect (6-Law) Graphs: {perfect_graphs_found}")
    if total_samples > 0:
        print(f"Strict 6-Law SVR: {(perfect_graphs_found / total_samples) * 100:.2f}%")
        
    print("-" * 50)
    if perfect_graphs_found > 0:
        for k, v in metrics.items():
            pct = (v / perfect_graphs_found) * 100
            print(f"{k}: {v} ({pct:.1f}%)")
    else:
        print("No perfect graphs found to execute.")
    print("="*50)

if __name__ == "__main__":
    main()
