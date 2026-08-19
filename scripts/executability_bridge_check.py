#!/usr/bin/env python3
"""MQ3 precondition — executability bridge check.

Does the Graph-Walk Interpreter execute (halt) ≥90% of solver-valid AR outputs?
Protocol: AR greedy decode with GT-BoM (ceiling arm) -> ConstraintSolver ->
GraphValidator (6-Laws) -> for solver-valid graphs only, build ExecutionGraph
with candidates from GT, run GraphInterpreter.run_with_limit, report halt rate.

Usage: python scripts/executability_bridge_check.py [ckpt] [d_model] [n_layers]
"""
import json
import sys

import torch
import numpy as np

from src.models.ar_edge_list import (ARGraphDataset, EdgeListDecoder, logit_heatmap_from_edges)
from src.models.constraint_solver import ConstraintSolver
from src.models.validation import GraphValidator
from scripts.evaluate_ar_edge_list import decode_one, motif_ids_from_graph
from src.execution_engine.schema import ExecutionGraph, Node, Edge, MotifType, EdgeType
from src.execution_engine.interpreter import GraphInterpreter

JSONL = "dataset/compressed/compressed_motifs.jsonl"
MOTIF_STR = {"Boundary": MotifType.BOUNDARY, "Sequence": MotifType.SEQUENCE,
             "Condition": MotifType.CONDITION, "Loop": MotifType.LOOP,
             "State": MotifType.STATE, "Message": MotifType.MESSAGE}


def discrete_to_execution_graph(discrete, motifs, lit_pool):
    """discrete [1,3,128,128], motifs [1,128] (1-6, 0 pad), lit_pool dict/int-keys.
    Returns ExecutionGraph or None (if no entry)."""
    d = discrete[0]
    n = int((motifs[0] != 0).sum().item())
    nodes, edges = [], []
    id_map = {i: i for i in range(n)}  # node ids already canonical

    for i in range(n):
        mid = int(motifs[0, i].item())
        nodes.append(Node(node_id=i, motif=MOTIF_STR[motif_str(mid)],
                          literal_pointer=i))  # pointer = node id (LT-tolerant)

    for i in range(n):
        for j in range(n):
            if d[0, i, j] > 0.5:
                et = EdgeType.DATA if d[1, i, j] > 0.5 else EdgeType.EXECUTION
                idx = int(d[2, i, j].item())
                edges.append(Edge(source_node=i, target_node=j, edge_type=et,
                                  input_index=idx))
    try:
        pool = {int(k): (v if not isinstance(v, str) else str(v)) for k, v in lit_pool.items()}
    except Exception:
        pool = {i: "lit" for i in range(max(len(lit_pool), 1))}
    return ExecutionGraph(nodes=nodes, edges=edges, literal_pool=pool)


def motif_str(mid):
    return {1: "Boundary", 2: "Sequence", 3: "Condition", 4: "Loop",
            5: "State", 6: "Message"}.get(mid, "Boundary")


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ar/ar_s0_e150.pt"
    d_model = int(sys.argv[2]) if len(sys.argv) > 2 else 384
    n_layers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = EdgeListDecoder(d_model=d_model, n_layers=n_layers).to(device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    ds = ARGraphDataset(jsonl_path=JSONL, augment_permutation=False, eval_only=True,
                        max_graphs=512)
    graphs = ds.graphs

    # GT rows (for literal pools) via the same split reconstruction
    import random as _r
    rows_all = []
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            if len(d["compressed_graph"]["nodes"]) <= 128:
                rows_all.append(d)
    n = len(rows_all)
    rng = _r.Random(42); order = list(range(n)); rng.shuffle(order)
    hold = sorted(order[:int(round(n * 0.16))])
    rows = [rows_all[i] for i in hold[:512]]

    n_solver_valid = 0
    n_halt = 0
    n_no_entry = 0
    n_inf = 0
    n_ctor_err = 0

    with torch.no_grad():
        for idx, (g, row) in enumerate(zip(graphs, rows)):
            num_nodes = len(g["nodes"])
            gt_motifs = [x["motif"] for x in g["nodes"]]
            edges_l = decode_one(model, g, num_nodes, device)
            heat = logit_heatmap_from_edges(edges_l, num_nodes).unsqueeze(0).to(device)
            mot_tensor = torch.tensor([motif_ids_from_graph(g)], dtype=torch.long, device=device)
            discrete = ConstraintSolver.discretize_and_repair(heat, mot_tensor)
            vr = GraphValidator.evaluate_batch(discrete, mot_tensor)
            if vr["perfect_graphs"] < 1.0:
                continue   # NOT solver-valid -> out of precondition denominator
            n_solver_valid += 1
            try:
                eg = discrete_to_execution_graph(discrete, mot_tensor,
                                                 row["compressed_graph"].get("literal_pool", {}))
            except Exception:
                n_ctor_err += 1
                continue
            try:
                result = GraphInterpreter(eg).run_with_limit(max_steps=2000)
            except Exception as e:
                n_ctor_err += 1
                continue
            st = result.get("status")
            if st == "halted":
                n_halt += 1
            elif st == "error_no_entry":
                n_no_entry += 1
            elif st == "infinite_loop":
                n_inf += 1
            else:
                n_ctor_err += 1
            if (n_solver_valid) % 32 == 0:
                print(f"  solver-valid {n_solver_valid} halt {n_halt} "
                      f"(rate {n_halt/max(n_solver_valid,1):.3f})", flush=True)

    rate = n_halt / max(n_solver_valid, 1)
    print("\n" + "=" * 60)
    print("EXECUTABILITY BRIDGE CHECK")
    print("=" * 60)
    print(f"solver-valid (denominator): {n_solver_valid}")
    print(f"halted (execute correctly):  {n_halt}")
    print(f"error_no_entry:             {n_no_entry}")
    print(f"infinite_loop:              {n_inf}")
    print(f"constructor errors:         {n_ctor_err}")
    print(f"HALT RATE among solver-valid: {rate:.3f}  (precondition: ≥ 0.90)")
    print("=" * 60)


if __name__ == "__main__":
    main()
