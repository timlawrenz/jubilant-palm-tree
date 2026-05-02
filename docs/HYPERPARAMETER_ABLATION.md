# Ablation Study: Hyperparameter Optimization for Neural Universal Machine

To optimize the training of the Diffusion Transformer (DiT) across the hybrid continuous/discrete loss objectives, we executed a rigorous ablation study focusing on the "Physics Engine" phase (graphs $\le$ 10 nodes).

The primary goal was to identify which parameters stabilized the discrete `input_index` categorical predictions—measured via the strict `In_Degree_Pass` and `Out_Degree_Pass` topological rules—while maintaining the continuous optimal transport vector field.

## Experimental Design
We isolated three core hyperparameters across an 8-run grid search:
1.  **Effective Batch Size** (`eff_bs`): 4, 16, 32, 64
2.  **Learning Rate** (`lr`): 1e-5, 1e-4, 5e-4
3.  **Axial Attention Depth** (`depth`): 6, 12, 24

Each configuration was trained for 50 epochs on the 1,890-graph subset. Performance was evaluated by averaging the Syntactic Validity pass rates over the final 10 epochs of each run.

---

## Results & Analysis

### 1. Batch Size Scaling
*Baseline: LR=1e-4, Depth=12*

| Effective BS | In-Degree Pass | Out-Degree Pass |
| :--- | :--- | :--- |
| 4 | 9.09% | 20.45% |
| **16** | **32.38%** | **46.02%** |
| 32 | 9.65% | 14.77% |
| 64 | 17.04% | 15.90% |

**Analysis:**
`BS=16` vastly outperformed the alternatives. 
*   At `BS=4`, gradient variance is too high; the model thrashes and struggles to resolve stable cross-entropy logits.
*   At `BS=32` and `BS=64`, the gradients become *too smooth*. The categorical dimension requires sharp, localized gradients to cleanly snap an edge's `input_index` into a discrete bucket (0 vs 1 vs 2). High batch sizes wash out these categorical gradients, causing the discrete accuracy to plummet.

### 2. Network Depth
*Baseline: BS=16, LR=1e-4*

| Depth | In-Degree Pass | Out-Degree Pass |
| :--- | :--- | :--- |
| 6 | 22.15% | 25.00% |
| **12** | **32.38%** | 46.02% |
| 24 | 20.45% | **50.56%** |

**Analysis:**
6 layers of Axial Attention are insufficient to execute the global "message passing" required to route graph logic. While `Depth=24` provided a minor (+4%) boost to `Out_Degree` reasoning, it severely degraded the `In_Degree` accuracy (-12%). `Depth=12` represents the optimal "Goldilocks" parameter for evaluating bidirectional topological constraints.

### 3. Learning Rate
*Baseline: BS=16, Depth=12*

| Learning Rate | In-Degree Pass | Out-Degree Pass |
| :--- | :--- | :--- |
| 1e-5 | 20.45% | 1.13% |
| **1e-4** | **32.38%** | **46.02%** |
| 5e-4 | 12.50% | 27.27% |

**Analysis:**
*   `1e-5` is far too cautious; after 50 epochs, the network had barely begun to understand `Out_Degree` routing laws.
*   `5e-4` is dangerously aggressive. Training logs revealed initial catastrophic gradient explosions (Loss spiked to >26.0 in Epoch 1) that scrambled the continuous vector field, before AdamW dragged the weights back to stability. `1e-4` remains perfectly stable.

---

## Conclusion
The study mathematically validates the original baseline hypothesis. The optimal configuration for the Permuted Dense DiT architecture on this dataset is:
*   **Effective Batch Size:** 16
*   **Network Depth:** 12 Axial Attention Blocks
*   **Learning Rate:** 1e-4

*Note: Because Axial Attention scales quadratically ($O(N^2)$), running a batch size of 16 on 128-node graphs exceeds 8GB VRAM limits. The curriculum orchestrator in `train.py` utilizes **Dynamic Gradient Accumulation** to artificially lock the effective batch size at 16 while stepping down physical batch sizes to prevent OOM errors during scale-up.*