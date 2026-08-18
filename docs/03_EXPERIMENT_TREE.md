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

- **[CONCLUDED] The 6th Law (Global Spine Validator)**
  - *Concept:* Update the `GraphValidator` to explicitly require exactly one Entry boundary and a fully connected macroscopic execution path, patching the specification gap discovered in the executability audit.
  - *Result:* Re-evaluation of the 10% solver dropped SVR to **3.12%**, isolating the truly executable scaffolds and filtering out locally-perfect but disconnected loops.
- **[PAUSED] Decode-Time Spine Repair**
  - *Concept:* Extend the Constraint Solver to deterministically assign an Entry node based on probability heatmaps and prune unreachable islands, lifting the global 6th Law pass rate.
  - *Depends on:* Verification that 6-Law graphs actually execute. Paused to prevent the solver from over-writing the model's fundamental failure to learn global control flow.
- **[CONCLUDED] Decode-Time Constraint Solving** — `exp/decode-time-solver` (merged to `main`)
  - *Concept:* Move hard-validity enforcement (acyclicity, exact degree) to a deterministic decode-time projector, applied *after* the DiT generates the continuous heatmap.
  - *Result:* End-to-end SVR 0% → **10%** with the corrected validator, no mode collapse. Acyclicity and Out-Degree mathematically guaranteed at 100%. Comprises the three sub-arms below.
- **[CONCLUDED] Index Collision Resolution**
  - *Concept:* Within the solver, resolve `input_index` collisions. If a Condition node has 2 exec edges claiming `index=0`, use the DiT's categorical logits to reassign one to `index=1`. 
  - *Result:* Resolved index collisions successfully. Execution Out-Degree hit 100%, and Data In-Degree climbed to ~57%. Final SVR reached 10.0%, doubling the best RLAIF performance without mode collapse.
- **[CONCLUDED] Degree Arithmetic Repair (Arity Snapping)**
  - *Concept:* Within the solver, implement deterministic top-K arity snapping for Execution and Data edges based on the specific Motif laws.
  - *Result:* Pushed Execution Out-Degree pass from 3% to 73%. Exposed index collisions as the final mathematical bottleneck.
- **[CONCLUDED] Acyclicity Repair Algorithm**
  - *Concept:* Within the solver, implement deterministic cycle-breaking on the data plane (e.g., DFS back-edge removal ranked by lowest edge probability).
  - *Result:* Successfully boosted Acyclicity Pass to 100% while maintaining edge density. SVR lifted to ~1.88%.

- **[TBD] Solver-in-the-Loop Sampling**
  - *Concept:* Apply lightweight projection *between* ODE steps, not just at the end,
    so the model's later steps denoise toward an already-feasible region.
  - *Open question:* Does intermediate projection help or destabilize the flow trajectory?

- **[CONCLUDED] RLAIF Structural Penalty** — `feat/rlaif-ablation` (frozen)
  - *Concept:* Differentiable versions of the 5 Laws as loss terms on continuous output.
  - *Result:* Mode collapse / reward hacking. A stiff exponential penalty (NOTEARS
    `tr(e^A)`) cannot be statically balanced against quadratic reconstruction MSE; the
    model erases edges to trivially satisfy constraints. See `02_EXPERIMENTS_AND_RESULTS.md`
    and `docs/assets/feat/rlaif-ablation/`.
  - *Salvaged finding:* "Sharpened NOTEARS" makes the acyclic penalty effective on soft
    adjacency — a reusable, publishable result even though the training arm failed.

---

## 2. Semantics & Execution

> **Pivotal finding (2026-06-05): the data is rich, the generator is the bottleneck — and our audit harness has a bug.**
> A ground-truth dataset audit showed **74.3%** of real compressed graphs pass all 6 Laws (vs. the generator's 1.95%), proving the executable structure IS learnable — the DiT is simply failing to reproduce it, not chasing an impossible target. HOWEVER, 100% of those known-good ground-truth graphs reported `error_no_entry` in the interpreter, which is impossible for real programs. This means `GraphInterpreter.run_with_limit`'s entry-detection heuristic is too strict and the "0% halting" results below are partly a harness artifact, not purely model failure. Raw log: `docs/assets/exp/validator-6th-law/dataset_audit.txt`.

### Can-we-save-this-arm? experiments (ordered)

- **[CONCLUDED] Exp 1 — Fix Interpreter Entry-Detection & Re-audit Ground Truth**
  - *Concept:* `run_with_limit` currently defines "entry" as a Boundary with zero incoming edges; real compressed graphs have entry Boundaries with incoming data edges, so it wrongly rejects them. Fix the heuristic (entry = Boundary with zero incoming *execution* edges, tie-broken sanely), then re-run the ground-truth audit.
  - *Result:* **Massive Success.** The Ground Truth dataset reached 100% halting. The generated Large-N 6-Law graphs ALSO reached **100% halting (20/20)**. The 6-Law specification is a perfect proxy for executability.

- **[CONCLUDED] Exp 1.5 — Routing Fidelity of the Executive Branch** — `AMBIGUOUS — Bill-of-Materials indictment (PIVOT)`
  - *Concept:* Feed real ground-truth motif sequences to the DiT, generate+solve, score edge-set fidelity (untyped + typed P/R/F1) against the source graph, contrast with a random-edge baseline. Controllability ablation: same noise seed, different motifs → measure Jaccard overlap.
  - *Result:* **Typed-F1 0.0852** (vs random baseline 0.0461, 1.85× above chance but far below the 0.50 threshold). **Jaccard 0.5045** (motif signal genuinely steers generation — outputs change with different Bills of Materials). **6.5× edge over-generation** (181 vs 28 ground truth). Best graph typed-F1 = 0.6000 (proves the DiT *can* route faithfully under favorable conditions, but doesn't most of the time).
  - *Implication:* The Executive Branch is steerable but the 1D motif array underspecifies programs. This is a **Legislative-Branch finding**: enrich the Bill of Materials (edge-count hints, degree profiles, partial adjacency seeding) before attempting the full intent→topology pathway (Exp 2).
  - *Artifacts:* `docs/assets/exp/routing-fidelity/fidelity_eval.txt`, `controllability_ablation.txt`; `src/models/fidelity.py`, `tests/test_fidelity.py`, `scripts/evaluate_routing_fidelity.py`, `scripts/ablation_motif_controllability.py` (branch `exp/routing-fidelity`)

- **[CONCLUDED] Exp 1.6 — Edge-Count Enrichment (BoM Arm A)** — `FAIL — density controls edges, fidelity never improves`
  - *Concept:* FiLM-style density bias on the presence channel at each ODE step, nudging toward target edge count. No model changes — pre-trained checkpoint as-is.
  - *Result:* **N=128 definitive sweep.** typed-F1=0.0875 baseline, all biased scales below (best 0.082 at scale 0.005, −5.8%). Edge count drops from 199 → 26 (mechanism works) but fidelity never beats vanilla. The presence channel carries routing signal that uniform bias strips indiscriminately.
  - *Implication:* Enriching the Bill of Materials with edge count is insufficient. Arm B (degree profiles) or C (partial-adjacency seeding) next.

- **[CONCLUDED] Exp 1.7 — Degree-Profile Enrichment (BoM Arm B)** — `AMBIGUOUS (+46% lift, insufficient)`
  - *Concept:* Per-node in-degree and out-degree expectations from dataset motif statistics, applied as FiLM bias on presence rows/columns at each ODE step.
  - *Result:* **N=128 definitive.** in_scale=0.05 lifts typed-F1 from 0.075 → 0.109 (+46%), best enrichment result yet. Out-degree bias hurts fidelity (−23%). Gen_edges increases 193 → 307 — the in-degree signal induces more edges, some correct.
  - *Implication:* The DiT IS receptive to non-trivial enrichment (strongest evidence yet). But presence-channel hints conflate edge count with edge quality. Arm C (partial-adjacency seeding) is the remaining option that directly targets *which* edges.

- **[CONCLUDED] Exp 1.8 — Partial-Adjacency Seeding (BoM Arm C)** — `NEGATIVE — scaffold works, completion fails`
  - *Concept:* Clamp a fraction p of ground-truth edges throughout the ODE trajectory; test whether the DiT can complete the rest.
  - *Result:* **N=64 definitive.** Typed-F1 scales with p (0.30 at p=0.20) but unseeded F1 = 0 everywhere. The model generates NO edges beyond the seed set — clamping breaks the denoising trajectory for all other edges.
  - *Implication:* The enrichment program is complete. The pre-trained DiT cannot be steered to specific programs via decode-time hints alone. Training-time enrichment or architectural change (Exp 3) needed.

- **[ACTIVE] Exp 1.9 — Training-Time Enrichment (BoM Arm D)** — `gate pre-registered, not yet executed`
  - *Concept:* Fine-tune the frozen DiT with the degree-profile signal (Arm B's tables) baked into the conditioning input channel, rather than applied as a decode-time bias. Tests whether the signal is informationally useful when delivered via gradients, not presence-channel nudges.
  - *Gate:* PASS if typed-F1 ≥ 0.20 AND ≥ +0.05 above decode-best (0.109), reproducible across 2 seeds. Null control: dummy (random) degree profile must lose by ≥ 0.05.
  - *Implication:* PASS un-gates Exp 2 (Legislative Branch). FAIL provides convergent negative evidence across 4 arms → justifies Exp 3 pivot.

- **[NEXT] Exp 2 — Build the A→C Conditioning Path (the Legislative Branch)**
  - *Concept:* Per `01_VISION_AND_ARCHITECTURE.md §8`, intent enters at the **Legislative Branch (LLM)**, which compiles human intent → the Bill of Materials (motif array) + Literal Pool. The DiT already consumes that Bill of Materials. This arm builds the LLM-shaped component that emits it from natural-language/source intent, completing the A→C path.
  - *Why it matters:* Tests the founding premise (intent → executable structure) end to end. Sits *on top* of a proven Executive Branch.
  - *Depends on:* **BoM enrichment** — at least one enrichment arm must lift typed-F1 > 0.20 before this arm is viable. Exp 1.5 proved the DiT is steerable but the 1D motif array underspecifies programs.
- **[TBD] Exp 3 — Invert the Architecture: Autoregressive Edge-List Generation**
  - *Concept:* If diffusion-over-dense-adjacency keeps failing at global control flow, treat the executable graph as the output "language": an autoregressive / seq2seq model that emits the edge-list directly, conditioned on intent. Control flow is inherently sequential/causal, which a dense-matrix diffusion prior may simply have the wrong inductive bias for.
  - *Why it matters:* Tests whether the failure is the *paradigm* (diffusion on matrices) rather than the *thesis* (A→C). Closer to how code LLMs already succeed, but the target is C (executable topology), not B (text).
  - *Depends on:* Exp 1 & 2 results — only pursue if conditioning a diffusion model proves insufficient.

### Concluded execution experiments

- **[CONCLUDED] Large-N Executability Audit on 6-Law Graphs**
  - *Concept:* Generate 1,000+ samples, run them through the full Constraint Solver + 6th Law Validator. For the truly perfect ones, run them in the Graph-Walk Interpreter to see if they halt.
  - *Result:* **0% Halting Rate on 20 perfect graphs (SVR 1.95%).** Stabilized the true generator SVR. NOTE: partly confounded by the interpreter entry-detection bug (see Exp 1) — re-confirm after the harness fix.
- **[CONCLUDED] Executability Audit of Generated Graphs**
  - *Concept:* Take structurally-valid generated graphs and actually run them through the Graph-Walk Interpreter using dummy operations.
  - *Result:* **0% Halting Rate.** 80.6% of structurally perfect graphs lacked an entry node; 19.4% died in unexpected sinks. Proved that local SVR constraints do not guarantee a globally executable macroscopic path (caveat: entry-detection harness bug, see Exp 1).

- **[TBD] Semantic Integration (The LLM Custodian)**
  - *Concept:* LLM maps human intent into the motif "bill of materials" and a Constant
    Pool; the structural matrix references literals via integer pointers.
  - *Depends on:* a reliable structural generator (MQ1) and a conditioning path (Exp 2).

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
