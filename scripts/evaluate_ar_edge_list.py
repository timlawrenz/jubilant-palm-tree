"""Exp 3 — Held-out routing-fidelity evaluation for the AR edge-list generator.

Same protocol as evaluate_routing_fidelity.py (typed/untyped F1 via
src.models.fidelity, random matched-density baseline) but:
  * decode = greedy autoregressive edge-token sampling from the AR model
  * conditioning = the graph's own motif array (prefix tokens)
  * eval set   = the PRE-REGISTERED HELD-OUT SPLIT (disjoint from training —
    an AR model can memorize the corpus; overlap would void the gate)
  * diagnostics = random baseline, train-cache memorization probe, and
    output-length sanity.

Usage (inline per project hang workaround):
  python -c "$(cat scripts/evaluate_ar_edge_list.py)" <ckpt_path> [device_hint]
"""

import sys

import numpy as np
import torch

from src.models.ar_edge_list import (ARGraphDataset, EdgeListDecoder, MAX_SEQ,
                                     decode_edge_tokens, logit_heatmap_from_edges)
from src.models.constraint_solver import ConstraintSolver
from src.models.fidelity import edge_fidelity

JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_GRAPHS_TARGET = 512
MAX_NODES = 128

# dataset.py convention: {Boundary:1, Sequence:2, Condition:3, Loop:4, State:5, Message:6}
MOTIF_MAP = {"Boundary": 1, "Sequence": 2, "Condition": 3, "Loop": 4,
             "State": 5, "Message": 6}
MOTIF_TOKEN_BASE = 134  # motif id 1 -> token 134


def motif_ids_from_graph(graph, N: int = MAX_NODES) -> list[int]:
    ids = [MOTIF_MAP.get(n["motif"], 0) for n in graph["nodes"]]
    return ids + [0] * (N - len(ids))   # pad to the solver's [1, 128] contract


def adjacency_from_graph(graph, N: int = MAX_NODES) -> torch.Tensor:
    """[3, N, N] GT adjacency in canonical node order (permutation OFF)."""
    adj = torch.zeros((3, N, N), dtype=torch.float32)
    nodes = graph["nodes"]
    id_to_idx = {n["node_id"]: i for i, n in enumerate(nodes)}
    for e in graph["edges"]:
        s, t = id_to_idx[e["source_node"]], id_to_idx[e["target_node"]]
        adj[0, s, t] = 1.0
        adj[1, s, t] = float(e["edge_type"])
        adj[2, s, t] = float(min(int(e["input_index"]), 3))
    return adj


def build_random_baseline(gt_adj, num_nodes, k, generator):
    N = gt_adj.shape[-1]
    rand_adj = torch.zeros((3, N, N), dtype=gt_adj.dtype)
    if k <= 0 or num_nodes < 2:
        return rand_adj
    present = gt_adj[0, :num_nodes, :num_nodes] > 0.5
    num_present = int(present.sum().item())
    data_ratio = float(gt_adj[1, :num_nodes, :num_nodes][present].mean().item()) \
        if num_present > 0 else 0.5
    positions = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    k = min(k, len(positions))
    perm = torch.randperm(len(positions), generator=generator)[:k].tolist()
    type_draws = torch.rand(k, generator=generator)
    for idx, pos_idx in enumerate(perm):
        i, j = positions[pos_idx]
        rand_adj[0, i, j] = 1.0
        rand_adj[1, i, j] = 1.0 if type_draws[idx].item() < data_ratio else 0.0
    return rand_adj


@torch.no_grad()
def decode_one(model, graph, num_nodes, device):
    """Greedy AR decode -> edge list for one graph.

    Prefix = per-node motif tokens (conditioning); body = edge quadruplets.
    Stops at EOS token or MAX_SEQ.
    """
    prefix = torch.tensor([MOTIF_TOKEN_BASE + (m - 1)
                           for m in (MOTIF_MAP.get(n["motif"], 0) for n in graph["nodes"])
                           if m in (1, 2, 3, 4, 5, 6)],
                          dtype=torch.long, device=device)
    tokens = prefix.clone()
    for _ in range(MAX_SEQ):
        if tokens.shape[0] >= MAX_SEQ:
            break
        valid = torch.ones(1, tokens.shape[0], dtype=torch.bool, device=device)
        logits = model(tokens.unsqueeze(0), valid)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == 140:  # EOS
            tokens = torch.cat([tokens, torch.tensor([140], device=device)])
            break
        tokens = torch.cat([tokens, torch.tensor([nxt], device=device)])
    return decode_edge_tokens(tokens.tolist(), num_nodes)


def eval_split(model, dataset, split_name, device, limit=None):
    gen_untyped, gen_typed, gen_edges_all, gt_edges_all = [], [], [], []
    rand_untyped, rand_typed = [], []
    num_scored = 0
    num_errors = 0

    for i in range(len(dataset)):
        if limit is not None and num_scored >= limit:
            break
        try:
            graph = dataset.graphs[i]
            num_nodes = len(graph["nodes"])
            if num_nodes == 0:
                num_errors += 1
                continue
            gt = adjacency_from_graph(graph)
            edges = decode_one(model, graph, num_nodes, device)
            heat = logit_heatmap_from_edges(edges, num_nodes).unsqueeze(0).to(device)
            motifs = torch.tensor([motif_ids_from_graph(graph)],
                                  dtype=torch.long, device=device)
            discrete = ConstraintSolver.discretize_and_repair(heat, motifs)[0].cpu()

            m = edge_fidelity(discrete, gt, num_nodes)
            rng = torch.Generator().manual_seed(num_scored)
            rand_adj = build_random_baseline(gt, num_nodes, m["gt_edges"], rng)
            m_r = edge_fidelity(rand_adj, gt, num_nodes)

            gen_untyped.append(m["untyped_f1"]); gen_typed.append(m["typed_f1"])
            rand_untyped.append(m_r["untyped_f1"]); rand_typed.append(m_r["typed_f1"])
            gen_edges_all.append(m["gen_edges"]); gt_edges_all.append(m["gt_edges"])
            num_scored += 1
            if num_scored % 16 == 0:
                print(f"  [{split_name}] scored {num_scored}/{len(dataset)} "
                      f"(errors {num_errors})", flush=True)
        except Exception as e:
            num_errors += 1
            if num_errors <= 3:
                print(f"  [graph error {split_name}] {e}", flush=True)

    if num_scored == 0:
        raise SystemExit(f"FATAL: {split_name} scored 0 graphs — eval invalid")

    a = np.array(gen_typed)
    q = (float(a.min()), float(np.percentile(a, 25)), float(np.median(a)),
         float(np.percentile(a, 75)), float(a.max()))
    return {
        "split": split_name, "n": num_scored, "errors": num_errors,
        "untyped": float(np.mean(gen_untyped)), "typed": float(np.mean(gen_typed)),
        "rand_untyped": float(np.mean(rand_untyped)), "rand_typed": float(np.mean(rand_typed)),
        "gen_edges": float(np.mean(gen_edges_all)), "gt_edges": float(np.mean(gt_edges_all)),
        "quantiles": q,
    }


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ar/ar_s0_e150.pt"
    d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AR-eval] ckpt={ckpt_path} d_model={d_model} n_layers={n_layers} "
          f"device={device}", flush=True)

    model = EdgeListDecoder(d_model=d_model, n_layers=n_layers).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})", flush=True)

    # Held-out evaluation suite (the gate) — augment_permutation OFF.
    ds_eval = ARGraphDataset(jsonl_path=JSONL_PATH, augment_permutation=False,
                             eval_only=True, max_graphs=NUM_GRAPHS_TARGET)
    # Train-cache contrast suite (memorization probe): 100 train graphs.
    ds_train = ARGraphDataset(jsonl_path=JSONL_PATH, augment_permutation=False,
                            eval_only=False, max_graphs=100)

    print("=== HELD-OUT EVAL (the gate) ===", flush=True)
    ev = eval_split(model, ds_eval, "holdout", device, limit=NUM_GRAPHS_TARGET)
    print("=== TRAIN-CACHE PROBE (memorization) ===", flush=True)
    tc = eval_split(model, ds_train, "train-cache", device, limit=100)

    print("\n" + "=" * 60)
    print(f"AR EDGE-LIST ROUTING FIDELITY — {ckpt_path.split('/')[-1]}")
    print("=" * 60)
    for r in (ev, tc):
        print(f"[{r['split']}] N={r['n']} (errors {r['errors']})")
        print(f"   untyped-F1 {r['untyped']:.4f} | typed-F1 {r['typed']:.4f} "
              f"| rand-typed {r['rand_typed']:.4f}")
        print(f"   gen_edges {r['gen_edges']:.1f} vs gt {r['gt_edges']:.1f}")
        print(f"   typed quantiles min/p25/med/p75/max: "
              f"{' / '.join(f'{x:.4f}' for x in r['quantiles'])}")
    print("-" * 54)
    delta = tc["typed"] - ev["typed"]
    print(f"memorization delta (train-cache − holdout typed-F1): {delta:+.4f} "
          f"  (pre-registered guardrail: ≤ +0.02)")
    if delta > 0.02:
        print("WARNING: train-cache >> holdout — probable edge-list memorization; "
              "gate is under suspicion.")


if __name__ == "__main__":
    main()