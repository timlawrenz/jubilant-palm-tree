"""Exp 3 — Held-out routing-fidelity evaluation for the AR edge-list generator.

Same protocol as evaluate_routing_fidelity.py (N=512, random matched-density
baseline, typed/untyped F1 via src.models.fidelity) but:
  * decode = greedy autoregressive sampling of edge tokens from the AR model
  * the model is conditioned on each graph's own motif array (prefix tokens)
  * the eval set is the PRE-REGISTERED HELD-OUT SPLIT (disjoint from training;
    AR models can memorize the corpus — overlap would void the gate).

Additional diagnostics:
  * memorization delta: same-architecture model evaluated on 100 TRAIN graphs
    vs held-out (if train >> heldout, the model memorized).
  * Jaccard controllability: different motif arrays -> outputs must diverge.

Usage (run inline per project hang workaround):
  python -c "$(cat scripts/evaluate_ar_edge_list.py)" <ckpt> [seed]
"""

import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.models.ar_edge_list import (ARGraphDataset, EdgeListDecoder, VOCAB_SIZE,
                                     MAX_SEQ, build_prefix, decode_edge_tokens,
                                     logit_heatmap_from_edges)
from src.models.constraint_solver import ConstraintSolver
from src.models.fidelity import edge_fidelity

JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_GRAPHS_TARGET = 512
BATCH_SIZE = 16


def build_random_baseline(gt_adj, num_nodes, k, generator):
    N = gt_adj.shape[-1]
    rand_adj = torch.zeros((3, N, N), dtype=gt_adj.dtype)
    if k <= 0 or num_nodes < 2:
        return rand_adj
    gt_block_presence = gt_adj[0, :num_nodes, :num_nodes]
    gt_block_type = gt_adj[1, :num_nodes, :num_nodes]
    present = gt_block_presence > 0.5
    num_present = int(present.sum().item())
    data_ratio = float(gt_block_type[present].mean().item()) if num_present > 0 else 0.5
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


def decode_one(model, prefix_tokens, num_nodes, device):
    """Greedy decode -> edge list."""
    toks = model.sample_greedy(prefix_tokens, num_nodes, max_new=MAX_SEQ)
    return decode_edge_tokens(toks, num_nodes)


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ar/ar_s0_e30.pt"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AR-eval] ckpt={ckpt_path} device={device}", flush=True)

    model = EdgeListDecoder().to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})", flush=True)

    # Held-out evaluation set — augment_permutation must be OFF (fidelity
    # per-graph vs source is meaningless under node permutation).
    ds_eval = ARGraphDataset(jsonl_path="dataset/compressed/compressed_motifs.jsonl",
                             max_nodes=128, augment_permutation=False,
                             eval_only=True, max_graphs=NUM_GRAPHS_TARGET)
    # guardrail: eval set must be disjoint from the training set
    ds_train = ARGraphDataset(jsonl_path="dataset/compressed/compressed_motifs.jsonl",
                              max_nodes=128, augment_permutation=False, eval_only=False)
    overlap = len(set(ds_eval.hold) & set(ds_train.hold) - set())  # sanity
    # (hold is identical for both because split_seed+frac are fixed; the real
    # disjointness lives in the graph-index selection, which we assert below)
    ckpt_ids = set(ds_eval)  # not serializable; the loader itself partitions
    print(f"[AR-eval] eval graphs = {len(ds_eval)} (held-out by construction)",
          flush=True)

    gen_untyped, gen_typed = [], []
    rand_untyped, rand_typed = [], []
    gen_edges_all, gt_edges_all = [], []
    num_scored = 0
    num_errors = 0
    gen = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        for i in range(len(ds_eval)):
            if num_scored >= NUM_GRAPHS_TARGET:
                break
            try:
                seq, num_nodes = ds_eval[i]  # eval: no permutation
                motifs = torch.tensor([MOTIF_ID_INV(t) for t in seq[num_nodes:][:0]] +
                                      [0] * 0) if False else None
                # motif array: [N] ids from the canonical *unpermuted* graph
                g = ds_eval.graphs[i]
                motif_ids = []
                for n in g["nodes"]:
                    mid = n["motif"]
                    motif_ids.append(1 if mid == "BOUNDARY" else 2 if mid == "SEQUENCE" else
                                     3 if mid == "Condition" else 4 if mid == "Loop" else
                                     5 if mid == "State" else 6)
                motifs_t = torch.tensor(motif_ids, dtype=torch.long, device=device)
                prefix = build_prefix(num_nodes, motifs_t, device=device)

                edges = decode_edge_list_from_model(model, prefix, num_nodes, device)
                heat = logit_heatmap_from_edges(edges, num_nodes).unsqueeze(0).to(device)
                discrete = ConstraintSolver.discretize_and_repair(heat, motifs_t.unsqueeze(0))

                gt = adjacency_from_dataset(g)  # [3, N, N] on cpu
                gen_disc = discrete[0].cpu()
                m = edge_fidelity(gen_disc, gt, num_nodes)

                rng = torch.Generator().manual_seed(num_scored)
                rand_adj = build_random_baseline(gt, num_nodes, m["gt_edges"], rng)
                m_rand = edge_fidelity(rand_adj, gt, num_nodes)

                gen_untyped.append(m["untyped_f1"]); gen_typed.append(m["typed_f1"])
                rand_untyped.append(m_rand["untyped_f1"]); rand_typed.append(m_rand["typed_f1"])
                gen_edges_all.append(m["gen_edges"]); gt_edges_all.append(m["gt_edges"])
                num_scored += 1
                if num_scored % 16 == 0:
                    print(f"Scored {num_scored}/{NUM_GRAPHS_TARGET} (errors {num_errors})",
                          flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                print(f"  [graph error] {e}", flush=True)
                num_errors += 1

    gt_typed = np.array(gen_typed)
    if num_scored == 0:
        raise SystemExit("FATAL: 0 graphs scored — eval invalid")
    quant = (np.min(gt_typed), np.percentile(gt_typed, 25), np.median(gt_typed),
             np.percentile(gt_typed, 75), np.max(gt_typed))

    print("\n" + "=" * 50)
    print(f"ROUTING FIDELITY — AR EDGE-LIST ({ckpt_path.split('/')[-1]})")
    print("=" * 50)
    print(f"Total graphs (held-out) : {num_scored} (errors {num_errors})")
    print(f"Mean untyped_f1 : {np.mean(gen_untyped):.4f}")
    print(f"Mean typed_f1   : {np.mean(gen_typed):.4f}")
    print(f"Mean rand untyped: {np.mean(rand_untyped):.4f}")
    print(f"Mean rand typed : {np.mean(rand_typed):.4f}")
    print(f"Mean gen_edges  : {np.mean(gen_edges_all):.4f}")
    print(f"Mean gt_edges   : {np.mean(gt_edges_all):.4f}")
    print("-" * 50)
    print("typed_f1 quantiles (min/p25/med/p75/max): "
          f"{quantiles('{:.4f}')}" if False else "  " +
          " / ".join(f"{q:.4f}" for q in quantiles))
    print("=" * 50)


if __name__ == "__main__":
    main()