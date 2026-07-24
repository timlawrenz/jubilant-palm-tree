"""Count-ablation: same-noise-seed, same-motif-sequence, different edge counts.

Proves the density bias signal is actually being READ by the model.
If Jaccard >= 0.85 with different counts, the model ignores the signal (FAIL).
"""
import torch

from src.models.model import NeuralUniversalMachineDiT, ode_generate_with_density_bias
from src.models.dataset import ExecutionGraphDataset

CKPT_PATH = "checkpoints/num_dit_epoch_340.pt"
JSONL_PATH = "dataset/compressed/compressed_motifs.jsonl"
NUM_STEPS = 20
SEEDS = (1234, 7, 99)
BEST_SCALE = 0.2  # override from sweep results after eval


def presence_edges(discrete):
    adj = discrete[0, 0]
    idx = (adj > 0.5).nonzero(as_tuple=False)
    return {(int(i), int(j)) for i, j in idx}


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Count-ablation on device: {device}")

    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12)
    ckpt = torch.load(CKPT_PATH, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    dataset = ExecutionGraphDataset(jsonl_path=JSONL_PATH, max_nodes=128,
                                     augment_permutation=False)
    # Pick one motif sequence (same graph for all seeds)
    item = dataset[0]
    mot_a = item["motifs"].unsqueeze(0).to(device)
    adj = item["adjacency"]
    num_nodes = int((mot_a[0] != 0).sum())
    k = int(adj[0, :num_nodes, :num_nodes].sum())
    density_true = float(k) / (num_nodes * max(1, num_nodes - 1))
    density_wrong = density_true * 2.0  # 2× count — should produce different output

    n = 128
    print(f"\n===== EDGE-COUNT ABLATION =====")
    print(f"num_nodes={num_nodes}  true_edges={k}  density={density_true:.4f}  wrong={density_wrong:.4f}")
    print(f"bias_scale={BEST_SCALE}  seeds={SEEDS}")

    jaccards = []
    with torch.no_grad():
        for seed in SEEDS:
            g = torch.Generator(device=device).manual_seed(seed)
            x0 = torch.randn((1, 6, n, n), device=device, generator=g)  # unused directly; ODE func makes its own noise

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) if device.type == "cuda" else None
            out_true = ode_generate_with_density_bias(
                model, mot_a, torch.tensor([density_true], device=device),
                device, num_steps=NUM_STEPS, density_bias_scale=BEST_SCALE)

            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) if device.type == "cuda" else None
            out_wrong = ode_generate_with_density_bias(
                model, mot_a, torch.tensor([density_wrong], device=device),
                device, num_steps=NUM_STEPS, density_bias_scale=BEST_SCALE)

            e_true = presence_edges(out_true)
            e_wrong = presence_edges(out_wrong)
            j = jaccard(e_true, e_wrong)
            jaccards.append(j)
            print(f"  seed={seed}: |true|={len(e_true)}  |wrong|={len(e_wrong)}  "
                  f"|∩|={len(e_true&e_wrong)}  |∪|={len(e_true|e_wrong)}  Jaccard={j:.4f}")

    mean_j = sum(jaccards) / len(jaccards)
    print(f"\nmean Jaccard: {mean_j:.4f}")
    print(f"threshold: Jaccard >= 0.85 → count signal IGNORED; < 0.85 → count signal READ")
    print(f"verdict: {'IGNORED (FAIL)' if mean_j >= 0.85 else 'READ (count signal active)'}")


if __name__ == "__main__":
    main()
