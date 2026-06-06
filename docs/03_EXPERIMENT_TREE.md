# Research & Experiment Tree

A central, living map of ideas, concepts, and experiments. This is the project's
strategic backlog — it shows what we are doing, what we have ruled out, and what
remains open, with links to the branch where each lives.

**Status tags:** `[ACTIVE]` · `[NEXT]` · `[TBD]` · `[PAUSED]` · `[CONCLUDED]` · `[BLOCKED]`

---

## The 10,000-Foot View

The project must answer three macro-questions, in dependency order. Everything
below hangs off one of them.

1. **Structural Validity** — Can we reliably emit graphs that pass all 5 Laws of Physics?
2. **Semantic Correctness** — Do valid graphs actually *compute the intended behavior*?
3. **End-to-End Usefulness** — Does the full pipeline beat simply asking an LLM for code?

We have a strong pre-trained generator and a trustworthy validator. We have ruled
out learning hard constraints via soft gradient penalties (RLAIF). The frontier is
now: enforce validity deterministically, then prove the valid graphs mean something.

---

## Macro-Question 1: Structural Validity

> Goal: SVR approaching the dataset ceiling (~83%) on generated graphs.

- **[ACTIVE] Decode-Time Constraint Solving** — `exp/decode-time-solver`
  - *Concept:* Let the DiT emit a continuous probability heatmap (its strength), then
    project onto the feasible set deterministically. Extend the Judicial Constraint
    Solver to guarantee acyclicity (topological edge-ordering / cycle-breaking) and
    degree laws by construction.
  - *Why now:* Hard combinatorial constraints are cheap to enforce by projection,
    brutal to learn by penalty (proven by the RLAIF negative result).
  - *Success metric:* `pretrained DiT + full solver` SVR measured with the corrected validator.
  - *First experiment:* The clean baseline — run the current pre-trained checkpoint
    through the existing solver and measure true SVR. We have never done this with the
    fixed validator. This single number reframes the entire project.

- **[NEXT] Acyclicity Repair Algorithm**
  - *Concept:* Within the solver, implement deterministic cycle-breaking on the data
    plane (e.g. DFS back-edge removal ranked by lowest edge probability).
  - *Depends on:* the baseline measurement above isolating acyclicity as the dominant failure.

- **[TBD] Solver-in-the-Loop Sampling**
  - *Concept:* Apply lightweight projection *between* ODE steps, not just at the end,
    so the model's later steps denoise toward an already-feasible region.
  - *Open question:* Does intermediate projection help or destabilize the flow trajectory?

- **[CONCLUDED] RLAIF Structural Penalty** — `feat/rlaif-ablation` (frozen)
  - *Concept:* Differentiable versions of the 5 Laws as loss terms on continuous output.
  - *Result:* Mode collapse / reward hacking. A stiff exponential penalty (NOTEARS
    `tr(e^A)`) cannot be statically balanced against quadratic reconstruction MSE; the
    model erases edges to trivially satisfy constraints. See `02_EXPERIMENTS_AND_RESULTS.md`
    and `docs/assets/rlaif_ablation/`.
  - *Salvaged finding:* "Sharpened NOTEARS" makes the acyclic penalty effective on soft
    adjacency — a reusable, publishable result even though the training arm failed.

---

## Macro-Question 2: Semantic Correctness

> Goal: A structurally-valid generated graph, when executed, produces correct output.
> **This is the project's largest untested assumption.** Validity ≠ meaning.

- **[NEXT] Executability Audit of Generated Graphs**
  - *Concept:* Take structurally-valid generated graphs and actually run them through
    the Graph-Walk Interpreter. Measure: do they halt? do they produce *any* output?
    This is distinct from SVR and currently completely unmeasured.
  - *Why it matters:* The Fibonacci demo was hand-built. We have zero evidence that a
    *generated* graph computes anything coherent. This experiment de-risks the thesis.

- **[TBD] Conditional Generation (Intent → Topology)**
  - *Concept:* Currently the DiT generates from noise + a motif sequence. To be useful
    it must be *conditioned on a task spec*. Explore conditioning the DiT on an
    intent embedding so generation is goal-directed, not unconditional.
  - *Open question:* What is the conditioning signal, and where does the motif sequence
    come from at inference time (the LLM custodian)?

- **[TBD] Semantic Integration (The LLM Custodian)**
  - *Concept:* LLM maps human intent into the motif "bill of materials" and a Constant
    Pool; the structural matrix references literals via integer pointers.
  - *Depends on:* a reliable structural generator (MQ1) and a conditioning path.

---

## Macro-Question 3: End-to-End Usefulness

> Goal: Demonstrate the full pipeline solves a real task — and ideally beats a
> plain LLM-writes-code baseline on some axis (verifiability, validity-by-construction).

- **[TBD] Execution-Trace Reward (Functional RL)**
  - *Concept:* If RL is ever revisited, reward *functional correctness* (run the graph,
    check it sorts/sums/etc.), never static topology. This sidesteps the reward-hacking
    failure entirely because the reward is grounded in execution, not structure.
  - *Depends on:* MQ2 — there must be executable, conditioned generation to reward first.

- **[TBD] Benchmark vs. LLM Code Generation**
  - *Concept:* Define a small task suite. Compare NUM pipeline (valid-by-construction)
    against an LLM emitting Ruby/Python on validity, correctness, and verifiability.
  - *Open question:* What is the honest, falsifiable claim we can make? Frame this before
    investing — it determines whether the whole research program "succeeds."

---

## Cross-Cutting / Infrastructure

- **[TBD] Structural Loss Unit Tests** — extend `tests/` to cover `compute_structural_loss`
  (e.g. assert empty graphs incur high density+orphan loss). The validator is tested; the
  loss is not. Cheap insurance even though RLAIF is paused.
- **[TBD] Solver Unit Tests** — as the decode-time solver grows, test its repair guarantees
  with hand-built cyclic/degree-violating fixtures, the same way we hardened the validator.
- **[PAUSED] Compute / Training Harness** — bfloat16 AMP + NaN-gradient guards are committed
  on `main`. Revisit only if a future arm needs large-scale training again.
