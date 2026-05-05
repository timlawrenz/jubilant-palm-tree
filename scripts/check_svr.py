import os
from tbparse import SummaryReader
import pandas as pd

runs_dir = "runs"
all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith("num_dit_run_")]
latest_run = max(all_runs, key=os.path.getmtime)
reader = SummaryReader(latest_run)
df = reader.scalars

svr = df[(df["tag"] == "Validation/SVR_Perfect_Graphs") & (df["step"] >= 335)]
print(svr[["step", "value"]].to_string(index=False))
