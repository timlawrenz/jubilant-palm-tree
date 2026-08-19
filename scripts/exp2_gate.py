#!/usr/bin/env python3
"""Exp 2 FULL GATE RUNNER — N=512 held-out, three arms.

Same protocol as the pilot (validated ceiling), scaled to the full held-out
set. Arms: CEILING (GT BoM), FLOOR (random BoM), LLM (qwen3-coder:30b).
Writes raw per-graph rows to a CSV + summary. Expected runtime ~25 min.
"""
import csv
import json
import random
import statistics
import sys
import time
import urllib.request

import torch

from src.models.ar_edge_list import (ARGraphDataset, EdgeListDecoder, MAX_NODES,
                                     MOTIF_BASE, decode_edge_tokens,
                                     logit_heatmap_from_edges)
from src.models.constraint_solver import ConstraintSolver
from src.models.fidelity import edge_fidelity
from scripts.evaluate_ar_edge_list import adjacency_from_graph

JSONL = "dataset/compressed/compressed_motifs.jsonl"
MODEL = "qwen3-coder:30b"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else "docs/assets/exp/legislative-branch/gate_rows.csv"
MOTIFS = ["Boundary", "Sequence", "Condition", "Loop", "State", "Message"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYS = f"""You transform Ruby method metadata into a structural motif list for a code-generator.
The six motif types are: {MOTIFS}.
A method's structure is a sequence of these motifs, ONE PER PROGRAM NODE, in execution order.

Example 1:
Method: compute_total
Literal pool: 0:'total', 1:'+', 2:'each'
Output: Boundary, Sequence, Condition, Loop, State

Example 2:
Method: find_by_id
Literal pool: 0:'id', 1:'where', 2:'first'
Output: Boundary, Sequence, Message, State

Rules:
- Emit ONLY comma-separated motif names, one per program node, in execution order.
- No explanations, no numbering, no markdown.
- Use exactly these spellings: {MOTIFS}."""


def build_prompt(d, g, max_lits=20):
    lp = g["literal_pool"]
    lit_text = "\n".join(f"  {k}: {v!r}" for k, v in list(lp.items())[:max_lits])
    return f"""Method: {d['method_name']}
Repo: {d['repo_name']}
File: {d['file_path']}
Literal pool (index: value):
{lit_text}
Output:"""


def llm_call(user):
    body = json.dumps({"model": MODEL, "system": SYS, "prompt": user,
                       "stream": False,
                       "options": {"temperature": 0.0, "num_predict": 600},
                       "keep_alive": "30m"}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=590).read())
    return time.time() - t0, r.get("response", "").strip()


def parse_motifs(out):
    txt = out.strip().split("\n")[0]
    toks = [t.strip().strip("*`") for t in txt.split(",")]
    return [t for t in toks if t in MOTIFS]


@torch.no_grad()
def decode_prefix(model, motif_list, device):
    toks = [MOTIF_BASE + MOTIFS.index(m) for m in motif_list]
    prefix = torch.tensor(toks, dtype=torch.long, device=device)
    seq = prefix.clone()
    for _ in range(MAX_NODES * 8):
        if seq.shape[0] >= MAX_NODES * 4:
            break
        valid = torch.ones(1, seq.shape[0], dtype=torch.bool, device=device)
        logits = model(seq.unsqueeze(0), valid)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == 140:
            seq = torch.cat([seq, torch.tensor([140], device=device)])
            break
        seq = torch.cat([seq, torch.tensor([nxt], device=device)])
    edges = decode_edge_tokens(seq.tolist(), len(motif_list))
    return edges


def score_graph(model, motif_list, g, device):
    motif_list = motif_list[:MAX_NODES]
    n = len(motif_list)
    gt = adjacency_from_graph(g)
    edges = decode_prefix(model, motif_list, device)
    heat = logit_heatmap_from_edges(edges, n).unsqueeze(0).to(device)
    motifs = torch.zeros(1, MAX_NODES, dtype=torch.long, device=device)
    for i, m in enumerate(motif_list):
        if i < MAX_NODES:
            motifs[0, i] = MOTIFS.index(m) + 1
    disc = ConstraintSolver.discretize_and_repair(heat, motifs)[0].cpu()
    block = min(n, len(g["nodes"]))
    m = edge_fidelity(disc, gt, block)
    return m["typed_f1"], m["gen_edges"], m["gt_edges"]


def main():
    ckpt_path = "checkpoints/ar/ar_s0_e150.pt"
    model = EdgeListDecoder(d_model=384, n_layers=8).to(DEVICE)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    ds = ARGraphDataset(jsonl_path=JSONL, augment_permutation=False, eval_only=True,
                        max_graphs=512)
    graphs = ds.graphs[:N]
    import random as _r
    rows_all = []
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            g = d["compressed_graph"]
            if len(g["nodes"]) <= 128:
                rows_all.append(d)
    ntot = len(rows_all)
    rng = _r.Random(42)
    order = list(range(ntot))
    rng.shuffle(order)
    n_hold = max(1, int(round(ntot * 0.16)))
    hold = sorted(order[:n_hold])
    picked = [rows_all[i] for i in hold[:N]]
    for g, d in zip(graphs, picked):
        assert len(g["nodes"]) == len(d["compressed_graph"]["nodes"]), "pairing mismatch"
    print(f"paired {len(picked)} rows with {len(graphs)} held-out graphs", flush=True)

    # FLOOR first (cheap, no LLM)
    print("=== FLOOR (random BoM) ===", flush=True)
    rngf = random.Random(42)
    f_typed = []
    for g in graphs:
        rand_motifs = [rngf.choice(MOTIFS) for _ in g["nodes"]]
        f1, _, _ = score_graph(model, rand_motifs, g, DEVICE)
        f_typed.append(f1)
    print(f"floor mean: {statistics.mean(f_typed):.4f}", flush=True)

    # CEILING (GT BoM)
    print("=== CEILING (GT BoM) ===", flush=True)
    c_typed = []
    for g in graphs:
        gt_motifs = [x["motif"] for x in g["nodes"]]
        f1, _, _ = score_graph(model, gt_motifs, g, DEVICE)
        c_typed.append(f1)
    print(f"ceiling mean: {statistics.mean(c_typed):.4f}", flush=True)

    # LLM arm
    print(f"=== LLM ({MODEL}) ===", flush=True)
    rows = []
    l_fid, l_typed, l_times = [], [], []
    for idx, (d, g) in enumerate(zip(picked, graphs)):
        user = build_prompt(d, g)
        dt, out = llm_call(user)
        pred = parse_motifs(out)
        gt_motifs = [x["motif"] for x in g["nodes"]]
        n = min(len(pred), len(gt_motifs))
        fid = sum(1 for i in range(n) if pred[i] == gt_motifs[i]) / max(n, 1) if n else 0.0
        f1, ge, gte = score_graph(model, pred if pred else ["Boundary"], g, DEVICE)
        l_fid.append(fid); l_typed.append(f1); l_times.append(dt)
        rows.append({"method": d["method_name"], "n_gt": len(gt_motifs), "n_pred": len(pred),
                     "fid": round(fid, 3), "e2e": round(f1, 3), "t": round(dt, 2)})
        if (idx + 1) % 32 == 0:
            print(f"  {idx+1}/{N} mean_e2e_so_far={statistics.mean(l_typed):.4f} "
                  f"mean_t={statistics.mean(l_times):.1f}s", flush=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n_gt", "n_pred", "fid", "e2e", "t"])
        w.writeheader(); w.writerows(rows)
    print("\n" + "=" * 60, flush=True)
    print(f"EXP 2 GATE (N={len(graphs)} held-out)", flush=True)
    print(f"  CEILING (GT BoM): {statistics.mean(c_typed):.4f}", flush=True)
    print(f"  FLOOR (random):   {statistics.mean(f_typed):.4f}", flush=True)
    print(f"  LLM mean e2e:     {statistics.mean(l_typed):.4f}", flush=True)
    print(f"  LLM mean fid:     {statistics.mean(l_fid):.4f}", flush=True)
    print(f"  mean t/call:      {statistics.mean(l_times):.1f}s", flush=True)
    print(f"  rows: {OUT_CSV}", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
