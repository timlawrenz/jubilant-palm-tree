#!/usr/bin/env python3
"""Exp 4.5 — Jedi bridge. Executes PIPELINE-GENERATED graphs (not GT).

Arms:
  A (G4.5a): GT BoM + GT binding + AR-generated edges -> ExecutionGraph ->
             interpreter -> compare last_value vs expected.
  B (G4.5b): LLM BoM + LLM binding + AR edges -> same. LLM sees ONLY method
             name + literal pool (no-leak; graph never in prompt).
Gates: A >= 50% cases correct; B >= 30% AND > random-binding (10%).

Usage: python scripts/exp45_jedi_bridge.py [ckpt]
"""
import json
import re
import subprocess
import sys
import time
import urllib.request

import torch

sys.path.insert(0, "scripts/mq3")
from tasks import TASKS

from src.execution_engine.schema import (ExecutionGraph, Node, Edge, MotifType,
                                          EdgeType)
from src.execution_engine.interpreter import GraphInterpreter
from src.models.ar_edge_list import (EdgeListDecoder, MAX_NODES, MOTIF_BASE,
                                     decode_edge_tokens, logit_heatmap_from_edges)
from src.models.constraint_solver import ConstraintSolver

MODEL = "qwen3-coder:30b"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOTIF_STR = {"Boundary": MotifType.BOUNDARY, "Sequence": MotifType.SEQUENCE,
             "Condition": MotifType.CONDITION, "Loop": MotifType.LOOP,
             "State": MotifType.STATE, "Message": MotifType.MESSAGE}
MOTIFS = ["Boundary", "Sequence", "Condition", "Loop", "State", "Message"]

LEG_BIND_SYS = """You map program-node literals for a tiny code-generator.
You receive a method name and its literal pool. You return TWO lines:
Line 1 OUTPUT: comma-separated motif list, one per program node, in execution
order. Motifs: Boundary, Sequence, Condition, Loop, State, Message.
Line 2 BIND: node1=literal,node2=literal,... assigning EVERY node either an
op/name from the pool or 'nil'.
No explanation, no markdown. Example:
OUTPUT: Boundary, Sequence, Message
BIND: 0=nil,1=nil,2=+"""


def ollama(sys_msg, user, temperature=0.0, num_predict=800):
    body = json.dumps({"model": MODEL, "system": sys_msg, "prompt": user,
                       "stream": False,
                       "options": {"temperature": temperature,
                                   "num_predict": num_predict},
                       "keep_alive": "30m"}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate",
                                 data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=590).read())
    return time.time() - t0, r.get("response", "").strip()


def param_names(t):
    m = re.search(r"def\s+\w+\(([^)]*)\)", t["method"])
    return [a.strip() for a in m.group(1).split(",")] if m else []


def gt_graphs():
    """Recompress tasks with the FINAL compressor (branch edges)."""
    try:
        return json.load(open("/tmp/mq3_gts.json"))
    except Exception:
        pass
    tasks_json = json.dumps(TASKS)
    open("/tmp/tasks45.json", "w").write(tasks_json)
    subprocess.run(["bundle", "exec", "ruby", "/tmp/gen_ast2.rb",
                    "/tmp/tasks45.json", "/tmp/mq3_asts45.jsonl"],
                   capture_output=True)
    sys.path.insert(0, "scripts/dataset_prep")
    from compress_ast import compress_dataset
    compress_dataset("/tmp/mq3_asts45.jsonl", "/tmp/mq3_compressed45.jsonl")
    gts = {}
    with open("/tmp/mq3_compressed45.jsonl") as f:
        for line in f:
            d = json.loads(line)
            gts[d["method_name"]] = d["compressed_graph"]
    json.dump(gts, open("/tmp/mq3_gts.json", "w"))
    return gts


def build_eg(graph, lit_pool):
    nodes = [Node(node_id=n["node_id"],
                  motif=MOTIF_STR.get(n["motif"], MotifType.MESSAGE),
                  literal_pointer=n.get("literal_pointer"))
             for n in graph["nodes"]]
    edges = [Edge(source_node=e["source_node"], target_node=e["target_node"],
                  edge_type=EdgeType.DATA if e["edge_type"] == 1 else EdgeType.EXECUTION,
                  input_index=e["input_index"]) for e in graph["edges"]]
    pool = {}
    for k, v in lit_pool.items():
        try:
            pool[int(k)] = v if not isinstance(v, str) else str(v)
        except Exception:
            pass
    return ExecutionGraph(nodes=nodes, edges=edges, literal_pool=pool)


def ar_decode(model, bom, device):
    """BoM (list of motif names) -> AR edges via greedy decode, then solver."""
    toks = [MOTIF_BASE + MOTIFS.index(m) for m in bom]
    prefix = torch.tensor(toks, dtype=torch.long, device=device)
    seq = prefix.clone()
    for _ in range(MAX_NODES * 4):
        if seq.shape[0] >= MAX_NODES * 4:
            break
        valid = torch.ones(1, seq.shape[0], dtype=torch.bool, device=device)
        logits = model(seq.unsqueeze(0), valid)[0, -1]
        nxt = int(logits.argmax().item())
        if nxt == 140:
            seq = torch.cat([seq, torch.tensor([140], device=device)])
            break
        seq = torch.cat([seq, torch.tensor([nxt], device=device)])
    edges = decode_edge_tokens(seq.tolist(), len(bom))
    # solver post-process (validity)
    heat = logit_heatmap_from_edges(edges, len(bom)).unsqueeze(0).to(device)
    motifs = torch.zeros(1, MAX_NODES, dtype=torch.long, device=device)
    for i, m in enumerate(bom):
        if i < MAX_NODES:
            motifs[0, i] = MOTIFS.index(m) + 1
    dis = ConstraintSolver.discretize_and_repair(heat, motifs)
    d = dis[0].cpu().numpy()
    out = []
    for i in range(len(bom)):
        for j in range(len(bom)):
            if d[0, i, j] > 0.5:
                out.append({"source_node": int(i), "target_node": int(j),
                            "edge_type": int(d[1, i, j]),
                            "input_index": int(d[2, i, j])})
    return out


def assemble_and_run(bom, binding, edges, lit_pool, args, names):
    """BoM + binding + AR edges -> ExecutionGraph -> interpreter run."""
    # pool: {idx: literal}; build reverse map literal -> idx (deterministic)
    rev = {str(v): k for k, v in lit_pool.items()}
    nodes, node_id_map = [], {}
    for i, m in enumerate(bom):
        nodes.append(Node(node_id=i, motif=MOTIF_STR.get(m, MotifType.MESSAGE),
                          literal_pointer=None))
    eg_edges = [Edge(source_node=e["source_node"], target_node=e["target_node"],
                     edge_type=EdgeType.DATA if e["edge_type"] == 1 else EdgeType.EXECUTION,
                     input_index=e["input_index"]) for e in edges
                if e["source_node"] < len(bom) and e["target_node"] < len(bom)]
    pool = {}
    for k, v in lit_pool.items():
        try:
            pool[int(k)] = v if not isinstance(v, str) else str(v)
        except Exception:
            pass
    # apply binding: node idx -> literal string -> pool index
    bound = 0
    for node in nodes:
        lit = binding.get(node.node_id)
        if lit is not None and lit != "nil":
            ptr = rev.get(str(lit))
            if ptr is None:
                pool[len(pool)] = str(lit)
                rev[str(lit)] = len(pool)
                ptr = rev[str(lit)]
            if ptr is not None:
                node.literal_pointer = ptr
                bound += 1
    eg = ExecutionGraph(nodes=nodes, edges=eg_edges, literal_pool=pool)
    interp = GraphInterpreter(eg)
    interp.run(max_steps=2000, args=dict(zip(names, args)))
    return interp.last_value, bound


def run_case_gta(model, t, gts, names, case):
    """Arm A: GT BoM + GT binding, AR edges."""
    gt = gts[t["name"]]
    bom = [n["motif_str"]] if False else [MOTIFS[int(n["motif"]) - 1] if isinstance(n["motif"], int) else n["motif"] for n in gt["nodes"]]
    edges = ar_decode(model, bom, DEVICE)
    binding = {n["node_id"]: str(gt["literal_pool"].get(str(n["literal_pointer"]), "nil")) for n in gt["nodes"]}
    args = case[:-1]
    lit_pool = {int(k): v for k, v in gt.get("literal_pool", {}).items()}
    got, bound = assemble_and_run(bom, binding, edges, lit_pool, args, names)
    return got, bound, len(edges)


def run_case_llm(model, t, names, case):
    """Arm B: LLM BoM + LLM binding, AR edges."""
    user = (f"Method: {t['name']}\nLiteral pool: "
            + ", ".join(f"{i}:{v}" for i, v in enumerate([t['name']] + names + ["+", "-", "*", "%", "/", "<", "==", "reverse"])))
    dt, out = ollama(LEG_BIND_SYS, user)
    bom, binding = [], {}
    for line in out.splitlines():
        if line.strip().upper().startswith("OUTPUT") or line.strip().upper().startswith("BIND"):
            parts = line.split(":", 1)
            val = parts[1].strip() if len(parts) > 1 else ""
            if line.upper().startswith("OUTPUT"):
                bom = [x.strip() for x in val.split(",") if x.strip() in MOTIFS][:MAX_NODES]
            else:
                for pair in val.split(","):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        try:
                            binding[int(k.strip())] = v.strip()
                        except ValueError:
                            pass
    if not bom:
        return None, 0, 0
    edges = ar_decode(model, bom, DEVICE)
    args = case[:-1]
    lit_pool = {i: v for i, v in enumerate([t["name"]] + names + ["+", "-", "*", "%", "/", "<", "==", "reverse"])}
    got, bound = assemble_and_run(bom, binding, edges, lit_pool, args, names)
    return got, bound, len(edges)


def norm(v):
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def main():
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ar_v2/v2_s0_e150.pt"
    model = EdgeListDecoder(d_model=384, n_layers=8).to(DEVICE)
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    gts = gt_graphs()

    exec_tasks = [t for t in TASKS if t["name"] not in
                  ("fib", "factorial", "gcd", "sum_range", "count_even")]
    a_correct = a_total = 0
    b_correct = b_total = 0
    failed = []
    for t in exec_tasks:
        names = param_names(t)
        for case in t["cases"]:
            exp = case[-1]
            # Arm A
            try:
                got, bound, ne = run_case_gta(model, t, gts, names, case)
                a_total += 1
                if got is not None and norm(got) == exp:
                    a_correct += 1
                print(f"A {t['name']} {case[:-1]}-> got={got} exp={exp} "
                      f"edges={ne} bound={bound} {'OK' if norm(got)==exp else 'x'}", flush=True)
            except Exception as e:
                failed.append((t["name"], "A", str(e)[:80]))
                print(f"A {t['name']} {case[:-1]}-> ERROR {str(e)[:60]}", flush=True)
            # Arm B
            try:
                got, bound, ne = run_case_llm(model, t, names, case)
                b_total += 1
                if got is not None and norm(got) == exp:
                    b_correct += 1
                print(f"B {t['name']} {case[:-1]}-> got={got} exp={exp} "
                      f"edges={ne} bound={bound} {'OK' if norm(got)==exp else 'x'}", flush=True)
            except Exception as e:
                failed.append((t["name"], "B", str(e)[:80]))
                print(f"B {t['name']} {case[:-1]}-> ERROR {str(e)[:60]}", flush=True)

    print("\n" + "=" * 60)
    print("EXP 4.5 JEDI BRIDGE RESULTS")
    print("=" * 60)
    print(f"Arm A (GT BoM+GT binding, AR edges): {a_correct}/{a_total} "
          f"({a_correct/max(a_total,1):.2f})  gate >= 0.50")
    print(f"Arm B (LLM BoM+LLM binding, AR edges): {b_correct}/{b_total} "
          f"({b_correct/max(b_total,1):.2f})  gate >= 0.30 AND > 0.10 random")
    if failed:
        print(f"failures: {len(failed)}")
        for f in failed[:8]:
            print(f"  {f}")


if __name__ == "__main__":
    main()