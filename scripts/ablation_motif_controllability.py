"""Motif controllability ablation (Task 4, routing-fidelity plan).

Headline disambiguator: hold the diffusion noise seed FIXED and vary ONLY the
motif (bill-of-materials) conditioning sequence. Because the 20-step ODE loop
is deterministic given (x0, motifs), any change in the decoded output topology
is attributable to the motif signal alone.

    Low  Jaccard overlap  -> motifs steer routing (controllable)
    High Jaccard overlap  -> the model ignores motif conditioning (red flag)

Inference-only probe. Do not run training here.
"""

import torch

from src.models.constraint_solver import ConstraintSolver
from src.models.dataset import ExecutionGraphDataset
from src.models.model import NeuralUniversalMachineDiT

CHECKPOINT_PATH = "checkpoints/num_dit_epoch_340.pt"
DATASET_PATH = "dataset/compressed/compressed_motifs.jsonl"
MAX_NODES = 128
NUM_STEPS = 20
SEEDS = (1234, 7, 99)

# Interpretation thresholds for mean Jaccard overlap.
CONTROLLABLE_THRESHOLD = 0.70
IGNORED_THRESHOLD = 0.90


def load_model(device: torch.device) -> NeuralUniversalMachineDiT:
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def generate(model, x0: torch.Tensor, motifs: torch.Tensor) -> torch.Tensor:
    """Canonical 20-step ODE loop starting from a CLONE of x0."""
    x = x0.clone()
    dt = 1.0 / NUM_STEPS
    B = x.shape[0]
    for step in range(NUM_STEPS):
        t_val = step * dt
        t_tensor = torch.full((B,), t_val, device=x.device)
        v = model(x, t_tensor, motifs)
        x = torch.cat([x[:, :2] + v[:, :2] * dt, v[:, 2:]], dim=1)
    discrete = ConstraintSolver.discretize_and_repair(x, motifs)
    return discrete


def presence_edges(discrete: torch.Tensor) -> set:
    """Untyped presence edge set {(i, j)} from discrete[0, 0] > 0.5."""
    adj = discrete[0, 0]
    idx = (adj > 0.5).nonzero(as_tuple=False)
    return {(int(i), int(j)) for i, j in idx}


def jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def interpretation(mean_j: float) -> str:
    if mean_j < CONTROLLABLE_THRESHOLD:
        return "motifs steer routing (controllable)"
    if mean_j >= IGNORED_THRESHOLD:
        return "WARNING: motifs ignored (conditioning bug)"
    return "ambiguous"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    dataset = ExecutionGraphDataset(
        jsonl_path=DATASET_PATH, max_nodes=MAX_NODES, augment_permutation=False
    )
    model = load_model(device)

    # Two distinct real ground-truth motif sequences at batch size 1.
    item_a = dataset[0]
    item_b = dataset[1]
    mot_a = item_a["motifs"].unsqueeze(0).to(device)
    mot_b = item_b["motifs"].unsqueeze(0).to(device)
    assert not torch.equal(mot_a, mot_b), "motif sequences A and B must differ"

    n = MAX_NODES  # union node block; keep it simple and comparable

    print("===== MOTIF CONTROLLABILITY ABLATION =====")
    print(f"checkpoint: {CHECKPOINT_PATH}")
    print(f"seeds: {SEEDS} | steps: {NUM_STEPS} | node block: {n}")

    jaccards = []
    with torch.no_grad():
        for seed in SEEDS:
            g = torch.Generator(device=device).manual_seed(seed)
            x0 = torch.randn((1, 6, n, n), device=device, generator=g)

            out_a = generate(model, x0, mot_a)
            out_b = generate(model, x0, mot_b)

            edges_a = presence_edges(out_a)
            edges_b = presence_edges(out_b)
            inter = edges_a & edges_b
            union = edges_a | edges_b
            j = jaccard(edges_a, edges_b)
            jaccards.append(j)

            print(
                f"seed={seed:>5}  |A|={len(edges_a):>4}  |B|={len(edges_b):>4}  "
                f"|inter|={len(inter):>4}  |union|={len(union):>4}  "
                f"Jaccard={j:.4f}"
            )

    mean_j = sum(jaccards) / len(jaccards)
    print(f"mean Jaccard over {len(jaccards)} seeds: {mean_j:.4f}")
    print(
        f"thresholds: Jaccard < {CONTROLLABLE_THRESHOLD:.2f} -> controllable; "
        f">= {IGNORED_THRESHOLD:.2f} -> motifs ignored; else ambiguous"
    )
    print(f"interpretation: {interpretation(mean_j)}")


if __name__ == "__main__":
    main()
