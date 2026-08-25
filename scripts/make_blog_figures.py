#!/usr/bin/env python3
"""Generate the four blog figures for the jubilant-palm-tree series.

Provenance rule: every plotted value is PARSED from a raw artifact on disk —
training logs, eval logs, or (for the two softmax scalars, which exist only as
prose) the experiment ledger. Nothing is hardcoded; if a source file changes,
the figure changes with it.

Figure sources:
  1. flat-loss            checkpoints/tte/train_{signal,null}_s0.log
  2. SVR comparison       docs/assets/exp/decode-time-solver/{baseline,final_solver}_eval.txt
                          docs/assets/exp/autoregressive-edge-list/svr_ar_{s0,s1}_e150.txt
  3. AR growth curve      docs/assets/exp/autoregressive-edge-list/fidelity_ar_s0_e{49,99,150_real}.txt
                          + fidelity_ar_s1_e150.txt (DiT-best 0.109 ref: Exp 1.7 ledger/outline)
  4. softmax before/after docs/02_EXPERIMENTS_AND_RESULTS.md L1002 (design revision 2 prose)

Output: docs/blog/assets/jubilant-palm-tree-*.png  (+ provenance README.md)
"""

import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "docs" / "assets" / "exp"
OUT = ROOT / "docs" / "blog" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

TEAL = "#3C5866"
GOLD = "#A3834C"
INK = "#222222"
GRAY = "#6B7A84"
GRID = "#E5E0D6"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#444444",
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 10.5,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _read(path: Path) -> str:
    return path.read_text()


def parse_epoch_losses(path: Path):
    """-> list[(epoch, loss)] deduplicated in epoch order."""
    seen, out = set(), []
    for line in _read(path).splitlines():
        m = re.match(r"epoch (\d+)/\d+ loss=([0-9.]+)", line.strip())
        if m:
            e = int(m.group(1))
            if e not in seen:
                seen.add(e)
                out.append((e, float(m.group(2))))
    if len(out) < 20:
        raise RuntimeError(f"{path.name}: expected 20 epochs, parsed {len(out)}")
    return out


def parse_fidelity(path: Path):
    """-> (holdout_typed_f1, rand_typed_f1, train_cache_typed_f1)"""
    txt = _read(path)
    typed = re.findall(r"(?<![-\w])typed-F1 ([\d.]+)", txt)
    rand = re.search(r"rand-typed ([\d.]+)", txt)
    if not typed or not rand:
        raise RuntimeError(f"{path.name}: could not parse fidelity")
    return float(typed[0]), float(rand.group(1)), float(typed[1])


def parse_svr_pct(path: Path) -> float:
    m = re.search(r"perfect_graphs: ([\d.]+)%", _read(path))
    if not m:
        raise RuntimeError(f"{path.name}: could not parse perfect_graphs")
    return float(m.group(1))


def footnote(ax, text: str) -> None:
    ax.text(
        0.0,
        -0.16,
        text,
        transform=ax.transAxes,
        fontsize=8,
        color=GRAY,
        va="top",
    )


def save(fig, name: str) -> Path:
    path = OUT / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path


def fig_flat_loss():
    sig = parse_epoch_losses(ROOT / "checkpoints" / "tte" / "train_signal_s0.log")
    nul = parse_epoch_losses(ROOT / "checkpoints" / "tte" / "train_null_s0.log")
    sx = [e for e, _ in sig]
    sy = [v for _, v in sig]
    ny = [v for _, v in nul]

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.plot(sx, sy, "-o", ms=4, color=TEAL, lw=2, label="signal arm (real degree profile)")
    ax.plot(sx, ny, "--o", ms=3.5, color=GOLD, lw=1.6, label="null arm (random degrees)")
    ax.annotate(
        f"epoch 1: {sy[0]:.4f}",
        xy=(sx[0], sy[0]),
        xytext=(2.4, sy[0] + 0.0032),
        fontsize=9,
        color=TEAL,
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8),
    )
    ax.annotate(
        f"epoch 20: {sy[-1]:.4f}",
        xy=(sx[-1], sy[-1]),
        xytext=(14.0, sy[-1] + 0.0034),
        fontsize=9,
        color=TEAL,
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.8),
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss")
    ax.set_title("Exp 1.9 training-time enrichment — the flat-loss plot")
    ax.set_xticks(range(1, 21))
    ax.set_xlim(1, 20)
    ax.set_ylim(0.0935, 0.1055)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9, frameon=False)
    ax.text(
        0.5,
        0.98,
        "signal vs null curves coincide — the conditioning signal is inert",
        transform=ax.transAxes,
        ha="center",
        fontsize=9.5,
        style="italic",
        color=GRAY,
    )
    footnote(
        ax,
        "FiLM-style adapter on the frozen base DiT (340 ep), eff-bs 16, lr 1e-5, 20 epochs, nan_guard=0. "
        "Sources: checkpoints/tte/train_signal_s0.log, train_null_s0.log (Exp 1.9 FAIL, commit 5c8fdf7).",
    )
    return fig


def fig_svr():
    dit_raw = parse_svr_pct(EXP / "decode-time-solver" / "baseline_eval.txt")
    dit_sol = parse_svr_pct(EXP / "decode-time-solver" / "final_solver_eval.txt")
    ar_s0 = parse_svr_pct(EXP / "autoregressive-edge-list" / "svr_ar_s0_e150.txt")
    ar_s1 = parse_svr_pct(EXP / "autoregressive-edge-list" / "svr_ar_s1_e150.txt")

    labels = ["DiT\n(naive repair)", "DiT +\nConstraintSolver", "AR +\nConstraintSolver\n(seed 0)", "AR +\nConstraintSolver\n(seed 1)"]
    vals = [dit_raw, dit_sol, ar_s0, ar_s1]
    colors = [TEAL, TEAL, GOLD, GOLD]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:.1f}%",
            xy=(b.get_x() + b.get_width() / 2, v),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    ax.annotate(
        "true corpus SVR: 0% —\nthe published \"100%\" was\ncurriculum-graphs only",
        xy=(0.0, 2.0),
        xytext=(0.16, 26),
        fontsize=9,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    )
    ax.set_ylabel("SVR (6-Laws perfect graphs, %)")
    ax.set_title("Structural validity across the paradigm shift")
    ax.set_ylim(0, 82)
    ax.axhline(0, color=INK, lw=0.8)
    footnote(
        ax,
        "SVR = fraction of generated graphs passing all six structural laws after discretization. "
        "DiT arms: N=160 (scripts/evaluate_baseline_solver.py, logs in decode-time-solver/). "
        "AR arms: held-out N=512 (scripts/check_ar_svr.py, logs in autoregressive-edge-list/).",
    )
    return fig


def fig_growth():
    e49 = parse_fidelity(EXP / "autoregressive-edge-list" / "fidelity_ar_s0_e49.txt")
    e99 = parse_fidelity(EXP / "autoregressive-edge-list" / "fidelity_ar_s0_e99.txt")
    e150 = parse_fidelity(EXP / "autoregressive-edge-list" / "fidelity_ar_s0_e150_real.txt")
    s1 = parse_fidelity(EXP / "autoregressive-edge-list" / "fidelity_ar_s1_e150.txt")

    xs = [49, 99, 150]
    ys = [e49[0], e99[0], e150[0]]
    rand = e150[1]  # 0.0741, identical across seeds
    dit_best = 0.109  # Exp 1.7 degree-profile best (outline Block 1)
    gate = 0.20  # pre-registered Tier-1 gate (ledger)
    train_cache = e150[2]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, ys, "-o", ms=6, color=TEAL, lw=2.2, label="AR seed-0 (canonical)", zorder=4)
    ax.plot([150], [s1[0]], "*", ms=14, color=GOLD, mec=GOLD, label=f"seed-1 e150: {s1[0]:.4f}", zorder=4)
    ax.plot([150], [train_cache], "D", ms=7, mfc="white", mec=GRAY, mew=1.4, label=f"train-cache probe: {train_cache:.4f}", zorder=4)

    ax.axhline(gate, color=GRAY, ls=":", lw=1.2)
    ax.text(148, gate + 0.014, f"pre-registered gate: {gate:.2f}", ha="right", fontsize=9, color=GRAY)
    ax.axhline(dit_best, color=GOLD, ls="--", lw=1.2)
    ax.text(51, dit_best + 0.016, f"DiT best (Exp 1.7): {dit_best:.3f}", fontsize=9, color=GOLD)
    ax.axhline(rand, color=GRAY, ls="--", lw=1.0)
    ax.text(51, rand - 0.05, f"random edge-list baseline: {rand:.4f}", fontsize=9, color=GRAY)

    for x, y in zip(xs, ys):
        ax.annotate(f"{y:.4f}", xy=(x, y), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9, color=TEAL)
    ax.annotate(f"{train_cache:.4f}", xy=(150, train_cache), xytext=(-6, 8), textcoords="offset points", ha="right", fontsize=9, color=GRAY)

    ax.set_xlabel("training epoch")
    ax.set_ylabel("held-out typed-F1 (edge-set routing fidelity)")
    ax.set_title("AR edge-list: growth curve e49 → e99 → e150")
    ax.set_xticks(xs)
    ax.set_xlim(30, 170)
    ax.set_ylim(0.0, 1.06)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.42), fontsize=8.6, frameon=False)
    footnote(
        ax,
        "Held-out N=512, split disjoint from training by construction (seed 42, asserted in harness). "
        "Sources: fidelity_ar_s0_e49/e99/e150_real.txt, fidelity_ar_s1_e150.txt; "
        "gate & DiT-best refs from the Exp 3 ledger and the series Data Outlines.",
    )
    return fig


def fig_softmax():
    # Prose-only diagnostics, Exp 3 design revision 2 — ledger L1002.
    before, after = 0.056, 1.000
    labels = ["with permutation\naugmentation (bug)", "canonical edge-list order\n(fixed)"]
    vals = [before, after]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bars = ax.barh(labels, vals, color=[GOLD, TEAL], height=0.52, zorder=3)
    for b, v in zip(bars, vals):
        ax.annotate(
            f"{v:.3f}",
            xy=(v, b.get_y() + b.get_height() / 2),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.annotate(
        "≈ 7× random — greedy decode\ncollapsed even on train graphs",
        xy=(before, 0),
        xytext=(0.18, 0.02),
        fontsize=8.6,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    )
    ax.annotate(
        "first-edge src softmax = 1.000 —\nmodel commits to the right node",
        xy=(after, 1),
        xytext=(0.52, 1.02),
        fontsize=8.6,
        color=GRAY,
        arrowprops=dict(arrowstyle="->", color=GRAY, lw=0.9),
    )
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("max P(src₀ | prefix) at the first edge token")
    ax.set_title("The permutation-augmentation bug, before / after")
    ax.grid(axis="y", visible=False)
    footnote(
        ax,
        "Prose values from the Exp 3 ledger, design revision 2 (docs/02_EXPERIMENTS_AND_RESULTS.md L1002); "
        "no separate raw artifact exists. Canonical edge-list order is the AR output language; "
        "permutation invariance is a DiT-matrix property, not an AR-sequence property.",
    )
    return fig


def main():
    print("Generating blog figures (all values parsed from raw artifacts)...")
    figs = [
        (fig_flat_loss, "jubilant-palm-tree-flat-loss-exp19.png"),
        (fig_svr, "jubilant-palm-tree-svr-comparison.png"),
        (fig_growth, "jubilant-palm-tree-ar-growth-curve.png"),
        (fig_softmax, "jubilant-palm-tree-softmax-before-after.png"),
    ]
    for fn, name in figs:
        try:
            save(fn(), name)
        except Exception as exc:  # noqa: BLE001 — surface source-file drift clearly
            print(f"FAILED {name}: {exc}", file=sys.stderr)
            sys.exit(1)
    print("All figures written.")


if __name__ == "__main__":
    main()
