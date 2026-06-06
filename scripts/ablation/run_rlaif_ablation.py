import os
import glob
import subprocess

CONFIGS = [
    {"name": "recon_dom_01_10", "beta_struct": 0.01, "beta_recon": 1.0},
    {"name": "balanced_05_05",  "beta_struct": 0.05, "beta_recon": 0.5},
    {"name": "struct_dom_02_08", "beta_struct": 0.2,  "beta_recon": 0.8}, # Run 9 base
    {"name": "struct_only_10_00", "beta_struct": 1.0,  "beta_recon": 0.0},
]

MAX_EPOCHS = 10
BATCH_SIZE = 2
GRAD_STEPS = 6

def get_latest_checkpoint(ckpt_dir):
    if not os.path.exists(ckpt_dir):
        return None, 0
    ckpts = glob.glob(os.path.join(ckpt_dir, "rlaif_struct_epoch_*.pt"))
    if not ckpts:
        return None, 0
    
    # Extract epoch numbers and find max
    epochs = []
    for c in ckpts:
        try:
            ep = int(c.split("_epoch_")[-1].replace(".pt", ""))
            epochs.append((ep, c))
        except ValueError:
            pass
    if not epochs:
        return None, 0
    
    epochs.sort(key=lambda x: x[0])
    return epochs[-1][1], epochs[-1][0]

def main():
    print("Starting RLAIF Resumable Ablation Grid...")
    
    for config in CONFIGS:
        run_name = f"ablation_{config['name']}"
        ckpt_dir = f"checkpoints/ablation/{config['name']}"
        
        latest_ckpt, latest_epoch = get_latest_checkpoint(ckpt_dir)
        
        if latest_epoch >= MAX_EPOCHS:
            print(f"\\n[SKIP] {run_name} already completed {MAX_EPOCHS} epochs.")
            continue
            
        print(f"\\n{'='*50}")
        if latest_ckpt:
            print(f"=== [RESUMING] {run_name} from Epoch {latest_epoch} ===")
        else:
            print(f"=== [STARTING] {run_name} ===")
        print(f"={'='*49}")
        
        cmd = [
            "python", "src/rlaif/train_rlaif.py",
            "--beta-struct", str(config["beta_struct"]),
            "--beta-recon", str(config["beta_recon"]),
            "--max-epochs", str(MAX_EPOCHS),
            "--batch-size", str(BATCH_SIZE),
            "--grad-steps", str(GRAD_STEPS),
            "--run-name", run_name,
            "--ckpt-dir", ckpt_dir
        ]
        
        if latest_ckpt:
            cmd.extend(["--resume", latest_ckpt])
            
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\\n[ERROR] Run {run_name} failed or was interrupted. State is saved.")
            print("To resume, simply run this script again.")
            exit(1)

    print("\\nAll ablation runs completed successfully!")

if __name__ == "__main__":
    main()
