"""Edge-set fidelity metric: compare a generated graph's adjacency to ground truth.

Scores only the [num_nodes, num_nodes] top-left block (real, non-padding nodes).
Channels: ch0 = presence, ch1 = edge_type (0=exec, 1=data), ch2 = input_index.
"""
import torch

def edge_fidelity(gen_adj: torch.Tensor, gt_adj: torch.Tensor, num_nodes: int) -> dict:
    g = gen_adj[:, :num_nodes, :num_nodes]
    t = gt_adj[:, :num_nodes, :num_nodes]
    gen_pres = (g[0] > 0.5)
    gt_pres = (t[0] > 0.5)
    def prf(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1
    tp = int((gen_pres & gt_pres).sum()); fp = int((gen_pres & ~gt_pres).sum()); fn = int((~gen_pres & gt_pres).sum())
    up, ur, uf1 = prf(tp, fp, fn)
    type_match = (g[1].round() == t[1].round())
    typed_tp = int((gen_pres & gt_pres & type_match).sum())
    typed_fp = int((gen_pres & ~(gt_pres & type_match)).sum())
    typed_fn = int((gt_pres & ~(gen_pres & type_match)).sum())
    tp_, tr_, tf1 = prf(typed_tp, typed_fp, typed_fn)
    return {"untyped_precision": up, "untyped_recall": ur, "untyped_f1": uf1,
            "typed_precision": tp_, "typed_recall": tr_, "typed_f1": tf1,
            "gen_edges": int(gen_pres.sum()), "gt_edges": int(gt_pres.sum())}
