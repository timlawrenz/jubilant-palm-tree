import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time

from src.models.dataset import ExecutionGraphDataset
from src.models.model import NeuralUniversalMachineDiT
from src.models.loss import compute_flow_matching_loss
from src.models.inference import sample_graph
from src.models.validation import GraphValidator

def get_curriculum_phase(epoch: int) -> int:
    """Returns the max_nodes limit based on the training epoch."""
    if epoch <= 50:
        return 10  # Phase 1: Physics Engine (Tiny graphs)
    elif epoch <= 150:
        return 30  # Phase 2: Control Flow (Medium graphs)
    else:
        return 128 # Phase 3: The Crucible (Full dataset)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Neural Universal Machine Training on {device}...")

    # 1. Initialize Model & Optimizer
    model = NeuralUniversalMachineDiT(
        hidden_dim=256, 
        num_heads=8, 
        depth=12
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    # 2. Initialize TensorBoard Writer
    run_name = f"num_dit_run_{int(time.time())}"
    log_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging initialized. Run: `tensorboard --logdir=runs`")
    
    epochs = 200
    batch_size = 32 # Adjust based on RTX 4090 VRAM

    for epoch in range(1, epochs + 1):
        # --- CURRICULUM UPDATE ---
        current_max_nodes = get_curriculum_phase(epoch)
        
        # Re-initialize dataset to filter by current_max_nodes
        dataset = ExecutionGraphDataset(
            jsonl_path="dataset/compressed/compressed_motifs.jsonl", 
            max_nodes=current_max_nodes,
            augment_permutation=True
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"\n=== Epoch {epoch}/{epochs} | Phase Max Nodes: {current_max_nodes} | Graphs: {len(dataset)} ===")
        
        # --- TRAINING LOOP (Continuous Flow Matching) ---
        model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device inside the loss function normally, but we can do it here for clarity
            for k in ["adjacency", "motifs", "padding_mask"]:
                batch[k] = batch[k].to(device)
            
            optimizer.zero_grad()
            
            # compute_flow_matching_loss handles t sampling, x_t interpolation, and masked MSE
            loss = compute_flow_matching_loss(model, batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Log batch loss
            global_step = (epoch - 1) * len(dataloader) + batch_idx
            writer.add_scalar("Training/Batch_Loss", loss.item(), global_step)
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss: {avg_loss:.4f}")
        writer.add_scalar("Training/Epoch_Loss", avg_loss, epoch)
        writer.add_scalar("Curriculum/Max_Nodes", current_max_nodes, epoch)
        
        # --- VALIDATION LOOP (Discrete Physics Check) ---
        model.eval()
        with torch.no_grad():
            # Grab a single batch of motifs to act as the "LLM Prompt"
            val_batch = next(iter(dataloader))
            val_motifs = val_batch["motifs"].to(device)
            
            # Sample discrete matrices using the Euler solver
            generated_adjacency = sample_graph(model, val_motifs, num_steps=20)
            
            # Pass to the Validation Harness
            val_metrics = GraphValidator.evaluate_batch(generated_adjacency, val_motifs)
            
            print(f"Validation SVR (Perfect Graphs): {val_metrics['perfect_graphs']:.1f}%")
            print(f"  - Out-Degree Pass: {val_metrics['out_degree_pass']:.1f}%")
            print(f"  - In-Degree Pass:  {val_metrics['in_degree_pass']:.1f}%")
            print(f"  - No Orphan Pass:  {val_metrics['no_orphan_pass']:.1f}%")
            
            # Log metrics to TensorBoard
            writer.add_scalar("Validation/SVR_Perfect_Graphs", val_metrics["perfect_graphs"], epoch)
            writer.add_scalar("Validation/Out_Degree_Pass", val_metrics["out_degree_pass"], epoch)
            writer.add_scalar("Validation/In_Degree_Pass", val_metrics["in_degree_pass"], epoch)
            writer.add_scalar("Validation/No_Orphan_Pass", val_metrics["no_orphan_pass"], epoch)
            writer.add_scalar("Validation/Acyclic_Data_Pass", val_metrics["acyclic_data_pass"], epoch)
            writer.add_scalar("Validation/Terminal_Sink_Pass", val_metrics["terminal_sink_pass"], epoch)

    writer.close()
    print("Training complete!")

if __name__ == "__main__":
    train()
