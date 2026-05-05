import os
from tbparse import SummaryReader

runs_dir = "runs"
all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith("num_dit_run_")]
latest_run = max(all_runs, key=os.path.getmtime)
reader = SummaryReader(latest_run)
df = reader.scalars

epoch_loss_df = df[df["tag"] == "Training/Epoch_Loss"]
if not epoch_loss_df.empty:
    current_epoch = epoch_loss_df["step"].max()
    print(f"Actual Epoch: {current_epoch}")
    
    print("\nLatest Epoch Values:")
    for tag in df["tag"].unique():
        if "Batch" not in tag:
            val = df[(df["tag"] == tag) & (df["step"] == current_epoch)]["value"]
            if not val.empty:
                print(f"{tag}: {val.iloc[0]:.2f}")
