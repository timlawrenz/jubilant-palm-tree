"""AR Executive SVR check (closes the audit-flagged gap from the step-back review).

The AR model's ROUTING fidelity is proven (0.776/0.777 held-out). This measures
STRUCTURAL validity: of the AR-generated discrete graphs, what fraction pass
the 6 Laws (SVR) through the same ConstraintSolver pipeline used by all prior
arms? Protocol mirrors evaluate_baseline_solver.py: discrete adjacency ->
GraphValidator.evaluate_batch. Same held-out 512 split as the fidelity gate.

Usage: python -c "$(cat scripts/check_ar_svr.py)" <ckpt> <d_model> <n_layers>
"""
import sys

import torch
import numpy as np

from src.models.ar_edge_list import ARGraphDataset, EdgeListDecoder, logit_heatmap_from_edges
from src.models.constraint_solver import ConstraintSolver
from src.models.validation import GraphValidator
from scripts.evaluate_ar_edge_list import decode_one, edge_fidelity, adjacency_from_graph, motif_ids_from_graph

JSONL = "dataset/compressed/compressed_motifs.jsonl"


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ar/ar_s0_e150.pt"
    d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AR-SVR] ckpt={ckpt_path} device={device}", flush=True)

    model = EdgeListDecoder(d_model=d_model, n_layers=n_layers).to(device)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    ds = ARGraphDataset(jsonl_path=JSONL, augment_permutation=False, eval_only=True, max_graphs=512)
    metrics_accum = {k: [] for k in ["perfect_graphs", "out_degree_pass", "in_degree_pass",
                                     "no_orphan_pass", "acyclic_data_pass", "terminal_sink_pass"]}
    typed_f1s, n_done = [], 0
    with torch.no_grad():
        for i in range(len(ds)):
            g = ds.graphs[i]
            num_nodes = len(g["nodes"])
            edges = decode_one(model, g, num_nodes, device)
            heat = logit_heatmap_from_edges(edges, num_nodes).unsqueeze(0).to(device)
            motifs = torch.tensor([motif_ids_from_graph(g)], dtype=torch.long, device=device)
            discrete = ConstraintSolver.discretize_and_repair(heat, motifs)
            vr = GraphValidator.evaluate_batch(discrete, motifs)
            for k in metrics_accum:
                metrics_accum[k].append(vr[k])
            gt = adjacency_from_graph(g)
            m = edge_fidelity(discrete[0].cpu(), gt, num_nodes)
            typed_f1s.append(m["typed_f1"])
            n_done += 1
            if n_done % 64 == 0:
                print(f"  {n_done}/512 (SVR so far {np.mean(metrics_accum['perfect_graphs']):.1f}%)", flush=True)

    print("\n" + "=" * 56)
    print(f"AR EXECUTIVE SVR — {ckpt_path.split('/')[-1]} (held-out N={n_done})")
    print("=" * 56)
    for k, v in metrics_accum.items():
        print(f"{k}: {np.mean(v):.2f}%")
    print(f"mean typed-F1 (concurrent fidelity): {np.mean(typed_f1s):.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()