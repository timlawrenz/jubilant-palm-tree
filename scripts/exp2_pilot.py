#!/usr/bin/env python3
"""Exp 2 pilot (N=20): Legislative LLM → AR Executive, three arms.

Arms (all held-out, same 20 graphs):
  llm    : qwen3-coder:30b few-shot → motif list → prefix → AR decode
  ceiling: GT motif array → prefix → AR decode  (must ≈ 0.776 to validate chain)
  floor  : random motif array (uniform) → AR decode

Metrics per graph: BoM fidelity (motif match over min(L,N)), end-to-end
typed-F1 (edge_fidelity within min(L,N) block), parse success, wall time.
"""
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
N_PILOT = int(sys.argv[1]) if len(sys.argv) > 1 else 20
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
    """Build prefix from motif names, greedy-decode edges.

    Convention: Boundary=1 -> token MOTIF_BASE+0 = 134, Sequence -> 135, etc.
    (module's motif_token(mid) = MOTIF_BASE + (mid-1))."""
    toks = [MOTIF_BASE + MOTIFS.index(m) for m in motif_list]
    prefix = torch.tensor(toks, dtype=torch.long, device=device)
    seq = prefix.clone()
    for _ in range(MAX_NODES * 8):
        if seq.shape[0] >= MAX_NODES * 4:
            break
        valid = torch.ones(1, seq.shape[0], dtype=torch.bool, device=device)
        logits = model(seq.unsqueeze(0), valid)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == 140:  # EOS
            seq = torch.cat([seq, torch.tensor([140], device=device)])
            break
        seq = torch.cat([seq, torch.tensor([nxt], device=device)])
    edges = decode_edge_tokens(seq.tolist(), len(motif_list))
    return edges


def score_graph(model, motif_list, g, device):
    motif_list = motif_list[:MAX_NODES]  # hard cap: 128-node model budget
    n = len(motif_list)
    gt = adjacency_from_graph(g)
    edges = decode_prefix(model, motif_list, device)
    heat = logit_heatmap_from_edges(edges, n).unsqueeze(0).to(device)
    motifs = torch.zeros(1, MAX_NODES, dtype=torch.long, device=device)
    for i, m in enumerate(motif_list):
        if i < MAX_NODES:
            motifs[0, i] = MOTIFS.index(m) + 1
    disc = ConstraintSolver.discretize_and_repair(heat, motifs)[0].cpu()
    # score within the min(L,N) block against GT
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
    graphs = ds.graphs[:N_PILOT]
    # EXACT pairing: reconstruct the same deterministic holdout split (seed 42,
    # frac 0.16) and take the first N_PILOT held-out rows in the same order.
    import random as _r
    rows_all = []
    with open(JSONL) as f:
        for line in f:
            d = json.loads(line)
            g = d["compressed_graph"]
            if len(g["nodes"]) <= 128:
                rows_all.append(d)
    n = len(rows_all)
    rng = _r.Random(42)
    order = list(range(n))
    rng.shuffle(order)
    n_hold = max(1, int(round(n * 0.16)))
    hold = sorted(order[:n_hold])
    picked = [rows_all[i] for i in hold[:N_PILOT]]
    # sanity: graph identity must match between dataset and row
    for g, d in zip(graphs, picked):
        assert len(g["nodes"]) == len(d["compressed_graph"]["nodes"]), "pairing mismatch"
    print(f"paired {len(picked)} rows with {len(graphs)} held-out graphs", flush=True)

    # --- ceiling arm: GT motifs ---
    print("=== CEILING (GT BoM) ===", flush=True)
    c_typed = []
    for g in graphs:
        gt_motifs = [n["motif"] for n in g["nodes"]]
        f1, _, _ = score_graph(model, gt_motifs, g, DEVICE)
        c_typed.append(f1)
    print(f"ceiling mean typed-F1: {statistics.mean(c_typed):.4f} (expect ~0.776)", flush=True)

    # --- floor arm: random BoM ---
    print("=== FLOOR (random BoM) ===", flush=True)
    rng = random.Random(42)
    f_typed = []
    for g in graphs:
        rand_motifs = [rng.choice(MOTIFS) for _ in g["nodes"]]
        f1, _, _ = score_graph(model, rand_motifs, g, DEVICE)
        f_typed.append(f1)
    print(f"floor mean typed-F1: {statistics.mean(f_typed):.4f}", flush=True)

    # --- llm arm ---
    print(f"=== LLM ({MODEL}) ===", flush=True)
    l_fid, l_typed, l_times, l_parse = [], [], [], 0
    for d, g in zip(picked, graphs):
        user = build_prompt(d, g)
        dt, out = llm_call(user)
        pred = parse_motifs(out)
        gt_motifs = [n["motif"] for n in g["nodes"]]
        n = min(len(pred), len(gt_motifs))
        fid = sum(1 for i in range(n) if pred[i] == gt_motifs[i]) / max(n, 1) if n else 0.0
        f1, ge, gte = score_graph(model, pred if pred else ["Boundary"], g, DEVICE)
        l_fid.append(fid); l_typed.append(f1); l_times.append(dt)
        l_parse += (len(pred) > 0)
        print(f"  {d['method_name']!r} n_gt={len(gt_motifs)} n_pred={len(pred)} "
              f"fid={fid:.2f} e2e={f1:.3f} t={dt:.1f}s", flush=True)
    print(f"\nLLM: parse={l_parse}/{len(graphs)} mean_fid={statistics.mean(l_fid):.3f} "
          f"mean_e2e={statistics.mean(l_typed):.4f} mean_t={statistics.mean(l_times):.1f}s "
          f"(512 extrap: {statistics.mean(l_times)*512/60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
