---
title: "The Literal Value Bottleneck: Why GNN Autoencoders Fail at Code Generation"
thumbnail: /blog/assets/gnn-ruby-code-study/thumbnail.png
authors:
- user: timlawrenz
tags:
- graph-neural-networks
- code-generation
- datasets
- research
---

# The Literal Value Bottleneck: Why GNN Autoencoders Fail at Code Generation

We trained GNN autoencoders on 22,000 Ruby ASTs. The models achieved **81% node type accuracy** and **99.5% type diversity**, yet generated exactly **0% syntactically valid code**. Here is why.

Graph Neural Networks seem like a natural fit for code — after all, Abstract Syntax Trees *are* graphs. If GNNs can learn molecular structures well enough to generate valid drug candidates, surely they can learn code structure well enough to generate valid programs?

We ran 51 GPU experiments across five GNN architectures, three decoder strategies, four hidden dimensions, and three loss functions to find out. The answer is a definitive no — but the *reason* is not what we expected.

## The Setup: 22,452 Ruby ASTs with 74-Dimensional Features

We parsed 22,452 Ruby methods from open-source repositories into AST graphs. Each node gets a **74-dimensional one-hot feature vector** encoding its AST type — one of 73 known types (`def`, `send`, `args`, `lvar`, `str`, ...) plus a single `UNKNOWN` token. Literal values — method names, variable names, string contents, numeric values — are stripped of their content and mapped to `UNKNOWN`.

The dataset is split 85/15 into training (19,084) and validation (3,368) sets.

<Tip>
📦 **Dataset**: [timlawrenz/gnn-ruby-code-study](https://huggingface.co/datasets/timlawrenz/gnn-ruby-code-study) on the Hub
</Tip>

## GNNs *Do* Learn Code Structure

Before diving into the failure, let's establish that GNNs genuinely learn meaningful representations of code.

For **cyclomatic complexity prediction** (a graph-level regression task), we compared five architectures: GCN, GraphSAGE, GAT, GIN, and GraphConv. The results are clear:

| Architecture | Layers | Val MAE ↓ | R² |
|---|---|---|---|
| **SAGE** | **5** | **4.018** | **0.709** |
| GIN | 3 | 4.589 | 0.629 |
| GAT (wide) | 3 | 4.662 | 0.612 |
| SAGE (baseline) | 3 | 4.782 | 0.635 |
| GCN | 3 | 5.321 | 0.563 |

A 5-layer GraphSAGE achieves **R² = 0.71**, explaining 71% of the variance in cyclomatic complexity. That's a **16% improvement** over the 3-layer baseline — and it's **9.9σ significant** based on 18 replicate runs (σ = 0.073).

Two patterns jump out:

1. **Depth dominates width.** Going from 3 to 5 layers improves MAE by 16%. Doubling the hidden dimension from 64 to 128? Zero improvement. For ASTs with depths of 10–30, deeper networks capture cross-branch dependencies that directly relate to complexity.

2. **GIN punches above its weight.** GIN's injective sum aggregation — which preserves the full multiset of neighbor features — gives it a 4% edge over SAGE at equal depth. This is the Weisfeiler-Leman advantage in practice.

So the models clearly learn the graph structure. The question is: can they *reconstruct* it?

## The Generation Catastrophe: 0% Across the Board

We trained graph autoencoders to encode an AST into a latent vector and decode it back. We tried everything:

- **5 architectures**: GCN, SAGE, GAT, GIN, GraphConv
- **3 loss functions**: simple (node type CE), improved (+ parent prediction), comprehensive
- **3 decoder edge modes**: chain (sequential), teacher-forced (ground-truth edges), iterative (predicted edges)
- **4 hidden dimensions**: 128, 256, 512, and deep 5-layer variants

<Tip warning={true}>
**Every single configuration produces 0% syntactically valid Ruby.**
</Tip>

| Decoder Conv | Hidden Dim | Loss | Syntax Validity |
|---|---|---|---|
| GAT | 256 | improved | 0% |
| SAGE | 256 | improved | 0% |
| GIN | 256 | improved | 0% |
| GCN | 256 | improved | 0% |
| GIN (teacher-forced, 5-layer) | 256 | improved | 0% |

Validation loss converges (to ~3.8 with teacher forcing), so the models *are* learning something. But what?

## The Core Discovery: The Literal Value Bottleneck

Here's where it gets interesting. When we gave our best model — a 5-layer teacher-forced GIN decoder — the ground-truth tree structure and only asked it to predict node types, it achieved:

- **81% node type accuracy**
- **99.5% type diversity** (8.6 unique types per sample)
- **0% syntax validity**

How can a model be 81% accurate and produce 0% valid code?

Look at this Ruby method:

```ruby
def call(storage)
  new(storage).call
end
```

Its AST contains 12 elements:

| Node | Ground Truth | Predicted | Match |
|---|---|---|---|
| 0 | `def` | `def` | ✓ |
| 1 | `"call"` → UNKNOWN | UNKNOWN | ✓ |
| 2 | `args` | `args` | ✓ |
| 3 | `arg` | `arg` | ✓ |
| 4 | `"storage"` → UNKNOWN | UNKNOWN | ✓ |
| 5 | `send` | `send` | ✓ |
| 6 | `send` | `send` | ✓ |
| 7 | `nil` → UNKNOWN | UNKNOWN | ✓ |
| 8 | `"new"` → UNKNOWN | UNKNOWN | ✓ |
| 9 | `lvar` | `lvar` | ✓ |
| 10 | `"storage"` → UNKNOWN | UNKNOWN | ✓ |
| 11 | `"call"` → UNKNOWN | UNKNOWN | ✓ |

**12/12 correct. 100% accuracy.** The model perfectly reconstructs the AST skeleton. But 6 of those 12 nodes are `UNKNOWN` — they are literal values (method names `call` and `new`, variable name `storage`, and a nil sentinel) that were encoded as the undifferentiated `UNKNOWN` token. The model correctly predicts `UNKNOWN` for all of them, which is technically right but utterly useless — the actual string content that makes the code meaningful is irrecoverable.

When we analyzed 500 validation samples, the numbers are stark:

| Category | Percentage |
|---|---|
| Typed AST nodes (`def`, `send`, `args`, ...) | 53.2% |
| Literal values (identifiers, strings, numbers) | **46.8%** |

**Nearly half of every AST is literal values.** Method names, variable names, string contents, numeric literals — all collapsed into a single `UNKNOWN` token. No amount of architectural tweaking, loss function engineering, or hidden dimension scaling can recover information that was never encoded.

This is the **literal value bottleneck**: the failure isn't in model capacity or architecture. It's in the input representation itself.

## Without Structure, Everything Collapses

The bottleneck becomes even more apparent when we remove teacher forcing. Without ground-truth edges, the chain decoder (which connects nodes sequentially, destroying the tree topology) exhibits **catastrophic mode collapse**:

- **92.7%** of all predicted tokens are `UNKNOWN`
- Only `def` (3.6%) and `send` (3.0%) appear as alternatives
- Average unique types per sample drops from 8.6 to **1.6**
- Type accuracy plummets from 81% to **48%**

The model learns a degenerate strategy: predict the most common token (which happens to be `UNKNOWN`, since 47% of the ground truth *is* `UNKNOWN`) and call it a day.

Teacher forcing fixes the *structural* component (restoring type accuracy to 81%), but the *lexical* component — the literal value bottleneck — remains.

## Dimension Doesn't Matter, Depth Does (Slightly)

One striking result: hidden dimensions of 128, 256, and 512 produce nearly **identical** outcomes:

| Config | Hidden Dim | Type Accuracy | Heuristic Validity |
|---|---|---|---|
| tf-gin-128 | 128 | 81.4% | 97.0% |
| tf-gin-256 | 256 | 81.3% | 97.0% |
| tf-gin-512 | 512 | 81.8% | 96.5% |
| tf-gin-256-deep | 256 (5 layers) | 81.1% | **99.5%** |

More capacity doesn't help. The bottleneck is **information-theoretic**, not computational. Going deeper (5 layers) nudges heuristic validity from 97% to 99.5%, consistent with the depth-over-width finding from complexity prediction — but the syntax validity needle doesn't move from 0%.

## The Infrastructure: 51 Experiments for $4.32

All of this — 51 GPU experiments across two hardware platforms — was orchestrated by [Ratiocinator](https://github.com/timlawrenz/ratiocinator), an autonomous LLM-driven research pipeline.

Ratiocinator handles the full lifecycle: it provisions Vast.ai RTX 4090 instances, deploys code via Git, installs dependencies, runs training, collects metrics, and tears down instances. The 18 baseline replicates that gave us our variance estimate? Those came from Ratiocinator accidentally running the same configuration three times due to an environment variable bug — which turned into a useful statistical gift.

| Experiment Set | Arms | Hardware | Cost |
|---|---|---|---|
| Autonomous research (3 iterations) | 24 | RTX 4090 (Vast.ai) | $1.50 |
| Architecture comparison | 8 | RTX 4090 (Vast.ai) | $1.20 |
| Generation analysis | 7 | RTX 4090 (Vast.ai) | $0.80 |
| Decoder topology | 6 | RTX 4090 (Vast.ai) | $0.70 |
| GIN deep dive | 5 | RTX 2070 SUPER (local) | ~$0.10 |
| **Total** | **51** | | **~$4.32** |

A 40-GPU ablation study for the price of a latte. By treating the research process itself as a distributed systems problem, Ratiocinator proves that high-velocity architectural ablation doesn't require a massive compute budget — just ruthless pipeline optimization.

## What Would It Take to Fix This?

Our findings point to three concrete directions:

1. **Literal value prediction heads.** Add separate output heads for identifier names (via a vocabulary or copy mechanism), string contents, and numeric values. The structural decoder already works — it's the lexical reconstruction that's missing.

2. **Hybrid architectures.** Use GNN encoders for structural understanding, but pair them with autoregressive or grammar-constrained decoders for sequential output. The GNN captures the *shape* of the code; a sequential decoder fills in the *content*.

3. **Pointer networks / copy mechanisms.** Let the decoder point back to nodes in the input graph to copy identifier names, rather than generating them from scratch. This is analogous to copy mechanisms in summarization models.

The fact that GNNs achieve R² = 0.71 for complexity prediction proves they learn meaningful code representations. The challenge is building decoders that can reconstruct the **full richness** of code — not just its structural skeleton — from those representations.

## Try It Yourself

The full dataset, pre-trained models, and experiment configurations are available:

- **📊 Dataset**: [timlawrenz/gnn-ruby-code-study](https://huggingface.co/datasets/timlawrenz/gnn-ruby-code-study) (22,452 Ruby methods with ASTs)
- **📄 Paper**: [Full research paper](https://huggingface.co/datasets/timlawrenz/gnn-ruby-code-study/blob/main/paper.md) with all tables and appendices
- **💻 Code**: [jubilant-palm-tree](https://github.com/timlawrenz/jubilant-palm-tree) (branch: `experiment/ratiocinator-gnn-study`)
- **🤖 Orchestrator**: [Ratiocinator](https://github.com/timlawrenz/ratiocinator) — the autonomous experiment runner

```bash
# Reproduce the key result
git clone -b experiment/ratiocinator-gnn-study https://github.com/timlawrenz/jubilant-palm-tree
cd jubilant-palm-tree
pip install torch torchvision torch_geometric

# Complexity prediction (the success story)
python train.py --conv_type SAGE --num_layers 5 --epochs 50

# Autoencoder with teacher-forced GIN (the 81%-accurate failure)
python train_autoencoder.py --decoder_conv_type GIN --decoder_edge_mode teacher_forced --epochs 30
```

If you use this dataset or findings, please cite:

```bibtex
@misc{lawrenz2025gnnruby,
  title={Graph Neural Networks for Ruby Code Complexity Prediction and Generation:
         A Systematic Architecture Study},
  author={Tim Lawrenz},
  year={2026},
  howpublished={\url{https://huggingface.co/datasets/timlawrenz/gnn-ruby-code-study}}
}
```
