import os
import pandas as pd
from tbparse import SummaryReader

def extract_all_runs(runs_dir="runs", output_csv="ablation_results.csv"):
    all_data = []
    
    for d in os.listdir(runs_dir):
        if d.startswith("ablation_"):
            # Parse hyperparams from name (e.g. ablation_bs64_lr0.0001_depth12_1777...)
            parts = d.split('_')
            bs_str = parts[1].replace("bs", "")
            lr_str = parts[2].replace("lr", "")
            depth_str = parts[3].replace("depth", "")
            
            run_path = os.path.join(runs_dir, d)
            print(f"Reading {run_path}...")
            reader = SummaryReader(run_path)
            df = reader.scalars
            
            if df.empty:
                print(f"  Warning: No scalar data found in {run_path}")
                continue
                
            # Pivot the dataframe so metrics are columns
            df_pivot = df.pivot(index='step', columns='tag', values='value').reset_index()
            
            # Add metadata
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
