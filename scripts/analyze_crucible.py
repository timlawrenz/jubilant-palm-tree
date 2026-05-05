import os
from tbparse import SummaryReader

def analyze_crucible_run():
    run_dir = "runs/num_dit_run_1777750476"
    print(f"Analyzing Crucible Run: {run_dir}")
    
    if not os.path.exists(run_dir):
        print(f"Error: {run_dir} does not exist.")
        return
        
    reader = SummaryReader(run_dir)
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
    
    for metric in metrics:
        metric_df = df[df['tag'] == metric]
        if not metric_df.empty:
            # We want to see how it performs at the end of Phase 1 (Epoch ~100)
            # and how it performs right now in Phase 2 (Epoch 281)
            epoch_100_data = metric_df[metric_df['step'] <= 100].tail(5)['value'].tolist()
            epoch_latest_data = metric_df.tail(5)['value'].tolist()
            
            avg_100 = sum(epoch_100_data) / len(epoch_100_data) if epoch_100_data else 0
            avg_latest = sum(epoch_latest_data) / len(epoch_latest_data) if epoch_latest_data else 0
            
            print(f"\n{metric}:")
            print(f"  Avg at end of Phase 1 (10-nodes): {avg_100:.2f}")
            print(f"  Avg current Phase 2 (30-nodes):   {avg_latest:.2f}")
            
if __name__ == "__main__":
    analyze_crucible_run()
