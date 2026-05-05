import os
from tbparse import SummaryReader

def analyze_latest_run():
    runs_dir = "runs"
    # Find the latest run
    all_runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if d.startswith("num_dit_run_")]
    if not all_runs:
        print("No training runs found.")
        return
        
    latest_run = max(all_runs, key=os.path.getmtime)
    print(f"Analyzing Latest Run: {latest_run}")
    
    reader = SummaryReader(latest_run)
    df = reader.scalars
    
    metrics = [
        "Training/Epoch_Loss",
        "Validation/SVR_Perfect_Graphs",
        "Validation/No_Orphan_Pass", 
        "Validation/Acyclic_Data_Pass",
        "Validation/Out_Degree_Pass",
        "Validation/In_Degree_Pass",
        "Curriculum/Max_Nodes"
    ]
    
    print("\n--- Current Moving Averages (Last 10 Epochs) ---")
    for metric in metrics:
        metric_df = df[df['tag'] == metric]
        if not metric_df.empty:
            recent_data = metric_df.tail(10)['value'].tolist()
            avg = sum(recent_data) / len(recent_data) if recent_data else 0
            
            # Print the current max nodes to contextualize the metrics
            if metric == "Curriculum/Max_Nodes":
                print(f"{metric}: {int(avg)} nodes")
            else:
                print(f"{metric}: {avg:.2f}")

if __name__ == "__main__":
    analyze_latest_run()
