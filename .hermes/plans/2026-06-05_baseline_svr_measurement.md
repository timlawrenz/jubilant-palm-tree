# Baseline SVR Measurement Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Establish the definitive baseline Syntactic Validity Rate (SVR) by running the pre-trained Flow-Matching DiT (Epoch 340) through our existing continuous-to-discrete solver, measured by the *corrected* 5 Laws of Physics `GraphValidator`. This provides the starting point for all decode-time solver improvements.

**Architecture:** 
1. Create a dedicated evaluation script (`scripts/evaluate_baseline_solver.py`) that loads `checkpoints/num_dit_epoch_340.pt`.
2. Generate a statistically significant batch of graphs from noise (e.g., 200 samples).
3. Pass the continuous output through the existing `naive_discretizer.py` (which serves as our baseline "solver").
4. Evaluate the discrete graphs using `GraphValidator.evaluate_batch`.
5. Document the precise failure modes (which laws fail most frequently) to target the next decode-time algorithmic repairs.

**Tech Stack:** Python, PyTorch

---

### Task 1: Create the Baseline Evaluation Script

**Objective:** Write a standalone script to generate and evaluate baseline graphs without the overhead of the RLAIF training loop or TensorBoard.

**Files:**
- Create: `scripts/evaluate_baseline_solver.py`

**Step 1: Write the script**

```python
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.models.model import NeuralUniversalMachineDiT
from src.models.dataset import ExecutionGraphDataset
from src.rlaif.naive_discretizer import naive_discretize
from src.models.validation import GraphValidator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating Baseline SVR on {device}...")

    # Load Model
    ckpt_path = "checkpoints/num_dit_epoch_340.pt"
    print(f"Loading {ckpt_path}")
    model = NeuralUniversalMachineDiT(hidden_dim=256, num_heads=8, depth=12).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load Dataset (for motif sequences and padding masks)
    dataset = ExecutionGraphDataset(
        jsonl_path="dataset/compressed/compressed_motifs.jsonl",
        max_nodes=128,
        augment_permutation=False,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # We will evaluate a subset of batches
    num_batches_to_eval = 5
    
    metrics_accum = {
        "out_degree_pass": [], "in_degree_pass": [], "no_orphan_pass": [],
        "acyclic_data_pass": [], "terminal_sink_pass": [], "perfect_graphs": []
    }
    edge_counts = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches_to_eval:
                break
                
            motifs = batch["motifs"].to(device)
            B, N = motifs.shape
            
            # Generate from noise
            x = torch.randn((B, 6, N, N), device=device)
            num_steps = 20
            dt = 1.0 / num_steps
            
            for step in range(num_steps):
                t_val = step * dt
                t_tensor = torch.full((B,), t_val, device=device)
                v = model(x, t_tensor, motifs)
                x_cont = x[:, :2] + v[:, :2] * dt
                x_cat = v[:, 2:]
                x = torch.cat([x_cont, x_cat], dim=1)
                
            # Discretize (Baseline Solver)
            discrete = naive_discretize(x, motifs)
            
            # Count edges
            presence = discrete[:, 0] # [B, N, N]
            edge_counts.extend(presence.sum(dim=(1,2)).cpu().tolist())
            
            # Evaluate
            val_results = GraphValidator.evaluate_batch(discrete, motifs)
            
            for k in metrics_accum.keys():
                metrics_accum[k].append(val_results[k])
                
            print(f"Batch {i+1}/{num_batches_to_eval} - SVR: {val_results['perfect_graphs']:.1f}%")

    # Aggregate and Print
    print("\n" + "="*50)
    print("BASELINE EVALUATION RESULTS (Pre-trained DiT + Naive Solver)")
    print("="*50)
    print(f"Total Samples: {num_batches_to_eval * 32}")
    for k, v in metrics_accum.items():
        avg = np.mean(v)
        print(f"{k}: {avg:.2f}%")
    print(f"Average Edge Count: {np.mean(edge_counts):.1f}")
    print("="*50)

if __name__ == "__main__":
    main()
```

**Step 2: Run the script to verify failure/pass**
Run: `PYTHONPATH=. python scripts/evaluate_baseline_solver.py`
Expected: Output showing the true baseline metrics with the corrected validator.

**Step 3: Commit**
```bash
git add scripts/evaluate_baseline_solver.py
git commit -m "eval: add baseline measurement script for pretrained DiT + solver"
```

---

### Task 2: Document the Baseline Results

**Objective:** Record the output from Task 1 into the `02_EXPERIMENTS_AND_RESULTS.md` ledger to form the starting line for the `exp/decode-time-solver` branch.

**Files:**
- Modify: `docs/02_EXPERIMENTS_AND_RESULTS.md`

**Step 1: Update the ledger**
Run the script, copy the results, and add a new subsection: "Baseline SVR Re-Measurement (Pre-Trained + Discretizer)" near the end of the pre-training section or start of the solver section. Explicitly state the average edge count, overall SVR, and individual law pass rates.

**Step 2: Commit**
```bash
git add docs/02_EXPERIMENTS_AND_RESULTS.md
git commit -m "docs: record true baseline SVR with corrected validator"
```

---

### Execution Handoff
Plan complete. Run this to execute the evaluation and record the precise gap the new deterministic solver needs to bridge.
