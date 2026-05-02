import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_ablation(csv_path="ablation_results.csv", output_dir="ablation_plots"):
    if not os.path.exists(csv_path):
        print(f"{csv_path} not found. Run extraction script first.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    # Clean up column names in case they have prefixes/suffixes
    # We want to match exactly what is in Tensorboard
    
    # 1. Batch Size Ablation
    plt.figure(figsize=(10, 6))
    subset_bs = df[(df['lr'] == 0.0001) & (df['depth'] == 12)]
    sns.lineplot(data=subset_bs, x='step', y='Validation/In_Degree_Pass', hue='eff_bs', palette='viridis')
    plt.title('Effect of Batch Size on In_Degree_Pass (LR=1e-4, Depth=12)')
    plt.savefig(os.path.join(output_dir, 'ablation_batch_size_in_degree.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=subset_bs, x='step', y='Validation/Out_Degree_Pass', hue='eff_bs', palette='viridis')
    plt.title('Effect of Batch Size on Out_Degree_Pass (LR=1e-4, Depth=12)')
    plt.savefig(os.path.join(output_dir, 'ablation_batch_size_out_degree.png'))
    plt.close()
    
    # 2. Depth Ablation
    plt.figure(figsize=(10, 6))
    subset_depth = df[(df['eff_bs'] == 16) & (df['lr'] == 0.0001)]
    sns.lineplot(data=subset_depth, x='step', y='Validation/In_Degree_Pass', hue='depth', palette='magma')
    plt.title('Effect of Network Depth on In_Degree_Pass (BS=16, LR=1e-4)')
    plt.savefig(os.path.join(output_dir, 'ablation_depth_in_degree.png'))
    plt.close()
    
    # 3. Learning Rate Ablation
    plt.figure(figsize=(10, 6))
    subset_lr = df[(df['eff_bs'] == 16) & (df['depth'] == 12)]
    sns.lineplot(data=subset_lr, x='step', y='Validation/In_Degree_Pass', hue='lr', palette='plasma')
    plt.title('Effect of Learning Rate on In_Degree_Pass (BS=16, Depth=12)')
    plt.savefig(os.path.join(output_dir, 'ablation_lr_in_degree.png'))
    plt.close()
    
    print(f"Plots saved to {output_dir}/")

if __name__ == "__main__":
    plot_ablation()
