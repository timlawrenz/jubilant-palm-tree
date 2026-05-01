import os
from tbparse import SummaryReader

def analyze_runs():
    runs_dir = "runs"
    # Find the most recently modified run directory
    all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
    latest_run = max(all_runs, key=os.path.getmtime)
    
    print(f"Analyzing TensorBoard Data from: {latest_run}")
    
    reader = SummaryReader(latest_run)
    df = reader.scalars
    
    metrics_to_check = [
        "Validation/SVR_Perfect_Graphs",
        "Validation/No_Orphan_Pass", 
        "Validation/Acyclic_Data_Pass",
        "Validation/Out_Degree_Pass",
        "Validation/In_Degree_Pass",
        "Training/Epoch_Loss"
    ]
    
    for metric in metrics_to_check:
        metric_df = df[df['tag'] == metric]
        if not metric_df.empty:
            recent_vals = metric_df['value'].tail(5).tolist()
            avg_recent = sum(recent_vals) / len(recent_vals)
            max_val = metric_df['value'].max()
            print(f"\n{metric}:")
            print(f"  Max hit: {max_val:.2f}")
            print(f"  Last 5 steps average: {avg_recent:.2f}")
            print(f"  Last 5 values: {[f'{v:.2f}' for v in recent_vals]}")

if __name__ == "__main__":
    analyze_runs()
