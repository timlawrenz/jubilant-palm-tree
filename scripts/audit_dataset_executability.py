import torch
from torch.utils.data import DataLoader

from src.models.dataset import ExecutionGraphDataset
from src.models.validation import GraphValidator
from src.execution_engine.dummy_custodian import DummySemanticCustodian
from src.execution_engine.interpreter import GraphInterpreter

def main():
    print("Auditing the GROUND TRUTH dataset against the 6 Laws + executability...")

    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=False,
    )
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    law_totals = {
        "out_degree_pass": 0.0, "in_degree_pass": 0.0, "no_orphan_pass": 0.0,
        "acyclic_data_pass": 0.0, "terminal_sink_pass": 0.0, "global_spine_pass": 0.0,
        "perfect_graphs": 0.0
    }
    exec_metrics = {"halted": 0, "infinite_loop": 0, "error_no_entry": 0, "error_unexpected_sink": 0, "crash": 0}
    n_batches = 0
    total = 0
    perfect = 0

    for batch in dataloader:
        motifs = batch["motifs"]
        adjacency = batch["adjacency"]  # ground truth [B,3,N,N]
        B = motifs.shape[0]
        total += B
        n_batches += 1

        res = GraphValidator.evaluate_batch(adjacency, motifs)
        for k in law_totals:
            law_totals[k] += res[k]

        # Per-graph: for the perfect ones, attempt execution
        for b in range(B):
            adj_b = adjacency[b].unsqueeze(0)
            motifs_b = motifs[b].unsqueeze(0)
            r = GraphValidator.evaluate_batch(adj_b, motifs_b)
            if r["perfect_graphs"] == 100.0:
                perfect += 1
                try:
                    g = DummySemanticCustodian.tensor_to_graph(adj_b[0], motifs_b[0])
                    interp = GraphInterpreter(g)
                    out = interp.run_with_limit(max_steps=1000)
                    s = out["status"]
                    exec_metrics[s] = exec_metrics.get(s, 0) + 1
                except Exception:
                    exec_metrics["crash"] += 1

    print("\n" + "="*60)
    print("GROUND TRUTH DATASET AUDIT")
    print("="*60)
    print(f"Total graphs: {total}")
    print("\n--- Per-Law Pass Rates (avg over batches) ---")
    for k, v in law_totals.items():
        print(f"{k}: {v / n_batches:.2f}%")

    print(f"\n--- Executability of 6-Law-Perfect Dataset Graphs ({perfect} found) ---")
    if perfect > 0:
        for k, v in exec_metrics.items():
            print(f"{k}: {v} ({(v/perfect)*100:.1f}%)")
    else:
        print("No 6-Law-perfect graphs in the dataset.")
    print("="*60)

if __name__ == "__main__":
    main()
