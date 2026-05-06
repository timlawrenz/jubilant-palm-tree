# Denoising Visualizations

These visualizations show the Neural Universal Machine's ODE denoising process — how pure noise transforms into a valid execution graph over 20 steps.

## Source Graph

- **Repository**: [Discourse](https://github.com/discourse/discourse)
- **File**: `app/serializers/post_serializer.rb`
- **Method**: `version`
- **Purpose**: Determines which version of a post to display (handles edits, wiki status, revision visibility)

### Graph Structure
- **19 nodes**: 1 Boundary, 1 Sequence, 3 Conditions, 1 State, 13 Messages
- **18 edges**: mix of execution flow and data dependencies
- Selected for motif diversity — contains branching, method calls, and state mutation

## Files

| File | Description |
|------|-------------|
| `denoising_heatmap_filmstrip.png` | Adjacency matrix (presence logits) at steps 0, 5, 10, 15, 19, and discretized. Shows structure emerging from noise. |
| `denoising_graph_evolution.gif` | 21-frame animation of the graph topology crystallizing. Blue edges = execution flow, red dashed = data dependencies. |
| `denoising_metrics_curve.png` | Edge count convergence (183 → 21) and sharpness over ODE steps. |

## Reproduction

```bash
python -m scripts.visualize_denoising \
  --checkpoint checkpoints/rlaif/rlaif_struct_epoch_5.pt \
  --seed 42
```

## Model

- **Checkpoint**: `checkpoints/rlaif/rlaif_struct_epoch_5.pt` (RLAIF structural loss, 5 epochs)
- **Architecture**: NeuralUniversalMachineDiT (14.3M params, hidden=256, heads=8, depth=12)
- **ODE**: 20-step Euler solver, flow matching
