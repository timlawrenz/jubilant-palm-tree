---
title: "The Plateau: Five Ways to Enrich a Heatmap, Zero Breakthroughs"
thumbnail: /assets/img/posts/jubilant-palm-tree-svr-comparison.png
authors:
- user: timlawrenz
tags:
- research
- jubilant-palm-tree
- graph-neural-networks
- code-generation
- diffusion-transformers
---

The last post in this project ended with a promise. [Eradicating Syntax](/2026/05/05/eradicating-syntax-the-neural-universal-machine.html) walked through a Diffusion Transformer that generates executable graphs instead of code, and closed with "What's next: RLAIF." This post is what actually happened next. Not the reinforcement learning success story the ending implied. Five attempts to make the model route specific programs, five failures, and one correction to a number I published that turned out to be wrong.

I owe you that correction first, because it reframes everything that follows.

## A correction before anything else

The May 5 post claimed a **100% Syntactic Validity Rate** on 128-node graphs. That number did not survive. Two things were wrong with it. The measurement ran on curriculum-generated graphs, the synthetic ones the model trains on first, not on real methods from the corpus {% cite lawrenz2025gnnruby %}. And the validator checking them had an inverted acyclicity test, so it passed graphs containing cycles. When we re-measured with a corrected validator on real corpus methods, the DiT's syntactic validity was **0%**. With the Constraint Solver repairing the output, it reached **10%**.

The repo caught this on May 9, four days after the post went up. The blog did not. That gap is exactly the kind of thing my research process is supposed to prevent, and it didn't, which is its own lesson. The May 5 post now carries the correction inline.

Here is the reframe, and it matters more than the embarrassment. The 100% figure, even taken at face value, was a statement about *structure*: the model could produce graphs that pass static legality checks. It said nothing about *routing*: whether you can point the model at one specific program and get that program back. Structure was solved. Routing was not. And routing is the whole game. A code generator that emits legal-looking programs at random is a random number generator with a legal output filter.

The question this post actually answers: can the dense-heatmap DiT be steered to a specific program? I ran five enrichment attempts at that question. All five are documented below with their gates, their numbers, and their verdicts.

* TOC
{:toc}

## The setup: routing fidelity, stated plainly

The system from May 5 has three branches. A Legislative branch (an LLM) turns intent into a Bill of Materials: a list of the six node motifs a program needs {% cite bohm1966flow %}. An Executive branch (the DiT, in the diffusion-transformer lineage of {% cite peebles2023scalable %}) turns that Bill of Materials into an adjacency matrix. A Judicial branch (the Constraint Solver) snaps the continuous output into legal graph shape.

The Executive's conditioning signal is the motif list. So the routing question is concrete: given the motif sequence `[Boundary, Sequence, Condition, State, ...]` from a real Ruby method, does the DiT produce *that method's graph*, or just any dense graph containing those ingredients?

The metric is typed-F1 over edges: generated edges must match ground-truth edges in both endpoints and type. The gate I pre-registered before running anything: **typed-F1 ≥ 0.20** to continue, with a random matched-density baseline reported alongside so I could tell signal from noise.

The baseline measurement (512 held-out graphs, permutation augmentation off, same ODE solver and Constraint Solver as every later run):

| Metric | DiT | Random baseline |
|---|---|---|
| Mean typed-F1 | **0.085** | 0.046 |
| Mean generated edges | 181.5 | 28 (ground truth) |
| Best single graph | 0.600 | |

Three facts jump out of that table. The DiT beats random by 1.85×, so the motif signal genuinely steers it. A controllability probe confirmed the steering: swap in a different Bill of Materials under the same noise seed and the output changes substantially (Jaccard 0.50 between the two outputs). And the model over-generates edges by 6.5×, producing dense graphs that have the right *kinds* of nodes but not the right *connections*. One graph in the sample hit 0.600. The median was 0.053. The 0.600 was an outlier, not a capability.

Verdict on the baseline: AMBIGUOUS. Steerable, yes. Specific, no. The natural reading was that the Bill of Materials was too thin, so the obvious move was to enrich the conditioning signal. That is the story of the next five attempts.

## Attempt 1: RLAIF, the promise from May 5

The plan in the May 5 post was reinforcement learning with the five graph laws as reward. That formulation died first. PPO with a discrete validity reward proved intractable on this setup, and the replacement, a differentiable structural loss (a NOTEARS acyclicity term plus degree and density constraints against the pre-trained weights), did something worse than fail: it collapsed. The model discovered that an empty graph trivially satisfies every structural constraint, and edge count fell from 486 to 3 to 0 within five epochs. Peak syntactic validity during RLAIF training was 7.1%, and it fell to zero with the collapse. The ablation grid showed no weighting of reconstruction versus structure that avoided one failure or the other.

The finding that survived: the model's easiest path to satisfy any edge-related pressure is to erase edges entirely. Mode collapse is the default attractor. You will see that pattern again below, twice.

## Attempts 2 through 5: four arms at the heatmap

With RLAIF closed, the enrichment program moved to the Executive itself. Four arms, each pre-registered with the same 0.20 gate and the same evaluation harness, each attacking the conditioning signal a different way.

### Arm 2: tell it how many edges (edge-count bias)

The cheapest hypothesis: the model over-generates 6.5×, so tell it the target edge count. At each ODE step, a bias nudges the presence channel toward the target density. No retraining; the frozen checkpoint just gets a hint.

The mechanism worked. Edge count dropped from 199 to 26 when pushed, nearly matching ground truth (28). Fidelity fell at every setting: the best arm scored 0.082 against a 0.088 baseline. The density channel carries two signals at once, *whether an edge exists* and *which edge it is*. A uniform bias on the channel cannot push one without flattening the other. Verdict: FAIL.

### Arm 3: tell each node its degree (degree-profile bias)

Sharper hint: instead of one global number, give each node its expected in-degree, computed from corpus statistics. This is the arm that worked, in the limited sense that it worked.

| Setting | typed-F1 | vs baseline |
|---|---|---|
| Baseline | 0.075 | |
| In-degree bias (best) | **0.109** | +46% |
| Out-degree bias | 0.057 | −23% |

A 46% lift, reproduced at two sample sizes (0.110 at N=64, 0.109 at N=128), and distinguishable from noise. Also not close to the gate: 0.109 against 0.20. And the lift came with the edge count inflating from 193 to 307, so some of the gain was correct edges and a lot was spurious ones. Telling a node "expect about 2.3 inputs" helps it accept edges. It does not help it pick the right donors. Out-degree bias actively hurt, the same flattening pathology as Arm 2. Verdict: AMBIGUOUS, best result of the program, insufficient.

### Arm 4: draw part of the graph for it (partial-adjacency seeding)

The most direct test of the heatmap's capacity: clamp 10 to 20% of the ground-truth edges into the ODE trajectory as fixed, and let the model complete the rest. Fill-in-the-blank instead of generate-from-noise.

The seeded edges survived: fidelity scaled linearly with the seeding fraction, reaching 0.30 at 20% seeded and 0.65 at 50%. And the completion? **Zero.** Across every seeding fraction, the model generated no edges beyond the ones clamped in. The unseeded-edge F1 was 0.000 at every setting. Clamping positions in the ODE trajectory breaks the denoising dynamics for everything around them. The model treats the scaffold as fixed and emits nothing else. The RLAIF attractor again: under constraint, erase edges. Verdict: NEGATIVE, and the most informative failure of the four. A model that cannot complete a partially drawn graph has no representation of "edges around a fixed scaffold."

### Arm 5: retrain with the signal (training-time enrichment)

Every arm so far pushed hints at a frozen model. The last arm baked the hint in: fine-tune the pre-trained checkpoint for 20 epochs with each node's expected in-degree concatenated onto its motif embedding, so the gradients themselves see the signal. Two arms from the same base, identical settings: one with real degrees, one with a null control (random degrees, same shape, meaningless) to catch placebo effects.

| Model | typed-F1 | vs frozen base |
|---|---|---|
| Frozen base (same harness) | 0.0895 | |
| Signal arm | 0.0894 | −0.0001 |
| Null arm | 0.0769 | −0.0126 |

The signal arm is statistically indistinguishable from the frozen base. The separation between signal and null is +0.0125, below the pre-registered 0.05 that would have let me attribute anything to the signal. The training loss tells the same story from the inside: 0.0976 to 0.0985 over 20 epochs. Flat. The fine-tune learned nothing, which is why I call it *inert* rather than failed. It did not hurt. It did nothing at all.

![Training loss of the Exp 1.9 signal arm (top curve) overlaid with the null arm, flat across 20 epochs.](docs/blog/assets/jubilant-palm-tree-flat-loss-exp19.png){: style="width: 100%;"}

Verdict: FAIL. The pre-registered outcome map fired: with the enrichment program closed, the next move was a paradigm change, not another enrichment.

## The turn: what five failures have in common

Line the attempts up and the pattern is hard to miss:

| Attempt | Mechanism | Best typed-F1 | Verdict |
|---|---|---|---|
| RLAIF | Structural reward on weights | 7.1% SVR, then collapse | Negative |
| Edge-count bias | Global density hint at decode | 0.082 | FAIL |
| Degree-profile bias | Per-node in-degree hint at decode | **0.109** | AMBIGUOUS |
| Partial seeding | Clamp real edges, complete the rest | 0.30* (all seeded) | NEGATIVE |
| Training-time | Degree signal in fine-tune | 0.089 | FAIL |

*Every single number below 0.20. The asterisk on 0.30 is doing a lot of work: that score is almost entirely the seeded edges echoing back, with zero completion.

Five different delivery mechanisms: reward shaping on the weights, global bias at decode, per-node bias at decode, clamping at decode, and gradients at train time. One consistent result. When the delivery mechanism varies and the result does not, the problem is not the delivery.

The problem is the heatmap. A dense adjacency matrix trained under a flow-matching objective {% cite lipman2023flow %} carries a presence value at every (i,j) coordinate: *how much edge* information everywhere and *which edge* information nowhere in particular. Every enrichment I tried operates on edge density, in aggregate. None of them can point at one specific coordinate and say "this edge, not that one," because nothing in the representation singles out a coordinate to point at. The model can produce dense motif-consistent structure (it does that reliably). It cannot select among the many legal graphs that share a Bill of Materials, and the Bill of Materials is all it ever sees.

![Syntactic validity rate across paradigms: DiT alone 0%, DiT with Constraint Solver 10%, autoregressive edge-list with solver 63% and 72% for two seeds.](docs/blog/assets/jubilant-palm-tree-svr-comparison.png){: style="width: 100%;"}

That figure carries the whole pivot in one image, by the way. The two left bars are this post's DiT measured honestly. The two right bars are the next post's model under the same solver. Same corpus, same validator. The difference is not the solver, the data, or the laws. It is the generator.

## The meaning: structure was solved, routing was the question

I want to be precise about what five failures bought, because it is not nothing.

The May 5 result, corrected, says the DiT can produce structurally legal graphs with solver assistance 10% of the time. This post's five attempts say no enrichment of the dense-heatmap conditioning, decode-time or training-time, pushes routing fidelity past 0.109 against a 0.20 gate. Together they isolate the bottleneck with convergent negative evidence: the dense-matrix flow-matching Executive does not have the routing capacity, and you cannot add it by enriching the input.

Every experiment here was gated before it ran, the null controls ran alongside, and each arm closed with a written verdict in the experiment ledger. That is why I can say "closed" and mean it. A killed idea gets a tombstone, not a comeback tour. The enrichment program is closed.

**Structure is not routing.** The May 5 post thought it had a generator of programs. It had a generator of legal graphs. Finding that out, cleanly, with gates and nulls and a correction on the record, is the most valuable output this project has produced so far. Negative results that identify a root cause are not waste; they are the narrowing that tells you where the actual problem lives. In this case the problem lives in the representation, and the next post changes it.

## What's next

The pivot: if a dense heatmap cannot carry *which edge*, stop denoising a matrix. A program graph is a sequence of decisions, so generate it the way language models generate anything: one edge at a time. Same motifs, same corpus, same solver, same held-out harness. That experiment exists now, it passed its gate at 0.776 where the DiT plateaued at 0.109, and it is the subject of the next post in this series.

If the meta-story sounds familiar, it is: this series is the concrete case study for [Found ≠ Fixed](/2026/08/19/found-not-fixed-arft.html), where I wrote about pre-registered gates and honest tombstones. This is what those rules look like when they earn their keep.

## Numbers at a glance

| Number | Value | What it measures |
|---|---|---|
| Routing gate (pre-registered) | 0.20 typed-F1 | Continue-enrichment threshold |
| DiT baseline | 0.085 | Routing fidelity, 512 held-out graphs |
| Random baseline | 0.046 | Matched-density null |
| Best enrichment (degree bias) | 0.109 (+46%) | Best of five attempts |
| Training-time arm | 0.0894 vs base 0.0895 | Inert fine-tune |
| Unseeded-edge F1 (partial seeding) | 0.000 | No completion capacity |
| DiT syntactic validity (corrected) | 0% (10% with solver) | On real corpus methods |
| May 5 claim (retracted) | "100% SVR" | Curriculum graphs, buggy validator |

## References

{% bibliography --cited %}
