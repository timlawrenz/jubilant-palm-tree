import subprocess
import os

configs = [
    # Baseline: BS=16, LR=1e-4, Depth=12
    {"eff_bs": 16, "lr": 1e-4, "depth": 12},
    
    # Experiment 1: Batch Size scaling
    {"eff_bs": 4, "lr": 1e-4, "depth": 12},
    {"eff_bs": 32, "lr": 1e-4, "depth": 12},
    {"eff_bs": 64, "lr": 1e-4, "depth": 12},
    
    # Experiment 2: Learning Rate scaling
    {"eff_bs": 16, "lr": 5e-4, "depth": 12},
    {"eff_bs": 16, "lr": 1e-5, "depth": 12},
    
    # Experiment 3: Network Depth
    {"eff_bs": 16, "lr": 1e-4, "depth": 6},
    {"eff_bs": 16, "lr": 1e-4, "depth": 24},
]

EPOCHS = 50
PHYSICAL_BS = 16

for idx, config in enumerate(configs):
    eff_bs = config["eff_bs"]
    lr = config["lr"]
    depth = config["depth"]
    
    # Calculate gradient accumulation
    # If eff_bs < PHYSICAL_BS, we must reduce physical_bs.
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
        "--force_phase_1"
    ]
    
    subprocess.run(cmd, check=True)

print("\n\nAll ablation grid searches completed successfully!")
