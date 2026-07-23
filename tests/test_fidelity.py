import torch
from src.models.fidelity import edge_fidelity

def _adj(edges, n=4):
    a = torch.zeros(3, n, n)
    for (u, v, et) in edges:
        a[0, u, v] = 1.0
        a[1, u, v] = float(et)
    return a

def test_identical_is_perfect():
    a = _adj([(0,1,0),(1,2,0)])
    r = edge_fidelity(a, a, 4)
    assert r["untyped_f1"] == 1.0
    assert r["typed_f1"] == 1.0

def test_disjoint_is_zero():
    a = _adj([(0,1,0)])
    b = _adj([(2,3,0)])
    r = edge_fidelity(a, b, 4)
    assert r["untyped_f1"] == 0.0

def test_wrong_type_breaks_typed_only():
    gt = _adj([(0,1,0)])
    gen = _adj([(0,1,1)])
    r = edge_fidelity(gen, gt, 4)
    assert r["untyped_f1"] == 1.0
    assert r["typed_f1"] == 0.0
