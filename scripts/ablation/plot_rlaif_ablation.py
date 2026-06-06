import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tbparse import SummaryReader

def extract_and_plot():
    runs_dir = "runs"
    all_data = []
    
    # Extract
    for d in os.listdir(runs_dir):
        if d.startswith("ablation_"):
            run_path = os.path.join(runs_dir, d)
            print(f"Reading {run_path}...")
            reader = SummaryReader(run_path)
            df = reader.scalars
            if df.empty: continue
                
            df_pivot = df.pivot(index='step', columns='tag', values='value').reset_index()
            df_pivot['run_name'] = d.replace("ablation_", "")
            all_data.append(df_pivot)
            
    if not all_data:
        print("No ablation data found.")
        return
        
    final_df = pd.concat(all_data, ignore_index=True)
    os.makedirs("ablation_plots", exist_ok=True)
    final_df.to_csv("ablation_plots/rlaif_metrics.csv", index=False)
    
    # Plot SVR
    plt.figure(figsize=(10, 6))
    if 'Eval/SVR' in final_df.columns:
        sns.lineplot(data=final_df.dropna(subset=['Eval/SVR']), x='step', y='Eval/SVR', hue='run_name')
        plt.title('SVR vs Training Steps by β Ratio')
        plt.savefig("ablation_plots/svr_comparison.png")
    
    # Plot Density
    plt.figure(figsize=(10, 6))
    if 'Structural/density' in final_df.columns:
        sns.lineplot(data=final_df.dropna(subset=['Structural/density']), x='step', y='Structural/density', hue='run_name')
        plt.title('Edge Density Loss vs Training Steps (Mode Collapse Indicator)')
        plt.savefig("ablation_plots/density_comparison.png")
        
    print("Saved plots to ablation_plots/")

if __name__ == "__main__":
    extract_and_plot()
