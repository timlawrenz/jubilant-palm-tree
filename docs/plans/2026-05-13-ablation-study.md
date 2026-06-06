# NUM Ablation Study: Effective Batch Size Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Conduct an ablation study to investigate the effect of `effective_batch_size` (and other parameters like `lr` and `depth`) on `In_Degree_Pass` and `Out_Degree_Pass` metrics, restricted to `max_nodes <= 10`, and generate blog-ready visuals.

**Architecture:** Use the local TensorBoard ablation pipeline. The study will run sequentially on local hardware, using `--force_phase_1` to constrain `max_nodes`. We will leverage the existing drafted scripts in `scripts/ablation/`, refining them to match the exact requirements and applying the pi216.ai / lawrenz.com blog color palette to the matplotlib outputs.

**Tech Stack:** Python, PyTorch, TensorBoard, tbparse, pandas, seaborn, matplotlib

---

### Task 1: Refine the Orchestrator Script

**Objective:** Ensure the ablation grid accurately scales gradient accumulation to match the requested effective batch sizes while forcing `max_nodes <= 10`.

**Files:**
- Modify: `scripts/ablation/run_ablation_grid.py`

**Step 1: Write minimal implementation**
The existing script has a good structure. We just need to confirm it runs cleanly and restricts to `max_nodes <= 10` using the `--force_phase_1` flag. Ensure the configurations cover `eff_bs` of `[4, 16, 32, 64]`.

```python
import subprocess
import os

configs = [
    # Baseline
    {"eff_bs": 16, "lr": 1e-4, "depth": 12},
    # Batch Size scaling
    {"eff_bs": 4, "lr": 1e-4, "depth": 12},
    {"eff_bs": 32, "lr": 1e-4, "depth": 12},
    {"eff_bs": 64, "lr": 1e-4, "depth": 12},
    # Learning Rate scaling
    {"eff_bs": 16, "lr": 5e-4, "depth": 12},
    {"eff_bs": 16, "lr": 1e-5, "depth": 12},
    # Network Depth
    {"eff_bs": 16, "lr": 1e-4, "depth": 6},
    {"eff_bs": 16, "lr": 1e-4, "depth": 24},
]

EPOCHS = 50
PHYSICAL_BS = 16

for idx, config in enumerate(configs):
    eff_bs = config["eff_bs"]
    lr = config["lr"]
    depth = config["depth"]
    
    if eff_bs < PHYSICAL_BS:
        phys_bs = eff_bs
        accum = 1
    else:
        phys_bs = PHYSICAL_BS
        accum = eff_bs // PHYSICAL_BS
        
    run_prefix = f"ablation_bs{eff_bs}_lr{lr}_depth{depth}"
    
    print(f"\n========================================================")
    print(f"=== Starting Ablation Run {idx+1}/{len(configs)}: {run_prefix} ===")
    print(f"========================================================")
    
    cmd = [
        "python", "src/train.py",
        "--epochs", str(EPOCHS),
        "--physical_batch_size", str(phys_bs),
        "--grad_accum_steps", str(accum),
        "--lr", str(lr),
        "--depth", str(depth),
        "--run_prefix", run_prefix,
        "--force_phase_1"  # Forces max_nodes <= 10
    ]
    
    subprocess.run(cmd, check=True)

print("\n\nAll ablation grid searches completed successfully!")
```

**Step 2: Commit**
```bash
git add scripts/ablation/run_ablation_grid.py
git commit -m "chore(ablation): refine orchestrator script for max_nodes <= 10"
```

---

### Task 2: Implement Metric Extraction

**Objective:** Extract `Validation/In_Degree_Pass` and `Validation/Out_Degree_Pass` metrics from TensorBoard logs into a CSV.

**Files:**
- Modify: `scripts/ablation/extract_metrics.py`

**Step 1: Write minimal implementation**
The existing script is mostly correct but needs to ensure it isolates the `step` and validation metrics accurately.

```python
import os
import pandas as pd
from tbparse import SummaryReader

def extract_all_runs(runs_dir="runs", output_csv="ablation_results.csv"):
    all_data = []
    
    for d in os.listdir(runs_dir):
        if d.startswith("ablation_"):
            parts = d.split('_')
            bs_str = parts[1].replace("bs", "")
            lr_str = parts[2].replace("lr", "")
            depth_str = parts[3].replace("depth", "")
            
            run_path = os.path.join(runs_dir, d)
            print(f"Reading {run_path}...")
            reader = SummaryReader(run_path)
            df = reader.scalars
            
            if df.empty:
                continue
                
            df_pivot = df.pivot(index='step', columns='tag', values='value').reset_index()
            
            df_pivot['eff_bs'] = int(bs_str)
            df_pivot['lr'] = float(lr_str)
            df_pivot['depth'] = int(depth_str)
            df_pivot['run_id'] = d
            
            all_data.append(df_pivot)
            
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv(output_csv, index=False)
        print(f"\nSuccessfully extracted metrics for {len(all_data)} runs to {output_csv}")
    else:
        print("\nNo ablation runs found or parsed.")

if __name__ == "__main__":
    extract_all_runs()
```

**Step 2: Commit**
```bash
git add scripts/ablation/extract_metrics.py
git commit -m "chore(ablation): implement tbparse metric extraction"
```

---

### Task 3: Implement Blog-Palette Visualization

**Objective:** Plot the ablation results using the specific pi216.ai blog palette for native integration into posts.

**Files:**
- Modify: `scripts/ablation/plot_ablation.py`

**Step 1: Write minimal implementation**
Apply the `BG`, `TEAL_DARK`, `GOLD`, `TEAL_MID`, and `BODY_TXT` colors directly to matplotlib.

```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# Blog Palette
BG        = "#FAF7F2"
TEAL_DARK = "#3C5866"
GOLD      = "#A3834C"
TEAL_MID  = "#6B8B9B"
SAGE      = "#8BA88E"
BODY_TXT  = "#2c3e50"

def apply_blog_style(fig, ax, title, xlabel, ylabel):
    ax.set_facecolor(BG)
    ax.set_xlabel(xlabel, color=BODY_TXT)
    ax.set_ylabel(ylabel, color=BODY_TXT)
    ax.set_title(title, color=TEAL_DARK, fontweight="bold")
    ax.tick_params(colors=BODY_TXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(TEAL_MID)
        spine.set_linewidth(0.8)
    ax.grid(True, color=TEAL_MID, alpha=0.25, linestyle="--", linewidth=0.7)
    
    # Style the legend
    leg = ax.get_legend()
    if leg:
        leg.get_frame().set_facecolor(BG)
        leg.get_frame().set_edgecolor(TEAL_MID)
        for text in leg.get_texts():
            text.set_color(BODY_TXT)

    # Gold accent bar
    fig.patches.append(mpatches.FancyBboxPatch(
        (0, 0.97), 1, 0.03, transform=fig.transFigure,
        boxstyle="square,pad=0", facecolor=GOLD, edgecolor="none", zorder=10))

def plot_ablation(csv_path="ablation_results.csv", output_dir="ablation_plots"):
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # 1. Batch Size Ablation (In Degree)
    subset_bs = df[(df['lr'] == 0.0001) & (df['depth'] == 12)]
    
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    sns.lineplot(data=subset_bs, x='step', y='Validation/In_Degree_Pass', hue='eff_bs', palette='viridis', ax=ax)
    apply_blog_style(fig, ax, "Effect of Batch Size on In_Degree_Pass (LR=1e-4, Depth=12)", "Step", "In_Degree_Pass %")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, 'ablation_batch_size_in_degree.png'), dpi=150, facecolor=BG)
    plt.close(fig)

    # 1b. Batch Size Ablation (Out Degree)
    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=BG)
    sns.lineplot(data=subset_bs, x='step', y='Validation/Out_Degree_Pass', hue='eff_bs', palette='viridis', ax=ax)
    apply_blog_style(fig, ax, "Effect of Batch Size on Out_Degree_Pass (LR=1e-4, Depth=12)", "Step", "Out_Degree_Pass %")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, 'ablation_batch_size_out_degree.png'), dpi=150, facecolor=BG)
    plt.close(fig)
    
    print(f"Styled plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_ablation()
```

**Step 2: Commit**
```bash
git add scripts/ablation/plot_ablation.py
git commit -m "feat(ablation): apply pi216 blog palette to ablation plots"
```

---

### Task 4: Execute Pipeline

**Objective:** Run the full pipeline to generate the metrics and graphs.

**Step 1: Run grid search**
```bash
python scripts/ablation/run_ablation_grid.py
```

**Step 2: Extract metrics**
```bash
python scripts/ablation/extract_metrics.py
```

**Step 3: Generate styled plots**
```bash
python scripts/ablation/plot_ablation.py
```

**Step 4: Verify outputs**
```bash
ls -l ablation_results.csv
ls -l ablation_plots/
```
