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

def get_curriculum_phase(epoch: int) -> tuple[int, int, int]:
    """
    Returns (max_nodes, physical_batch_size, accumulation_steps).
    Maintains an effective batch size of 16 across all phases to stabilize gradients,
    while scaling down the physical batch size to prevent OOM errors on 8GB VRAM 
    when the adjacency matrices grow.
    """
    if epoch <= 100:
        return 10, 16, 1   # Phase 1: 10x10 matrices. VRAM is fine.
    elif epoch <= 300:
        return 30, 4, 4    # Phase 2: 30x30 matrices. Axial attention reshaping spikes VRAM.
    else:
        return 128, 1, 16  # Phase 3: 128x128 matrices. Extremely heavy.

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
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard logging initialized. Run: `tensorboard --logdir=runs`")
    
    epochs = 400

    for epoch in range(1, epochs + 1):
        # --- CURRICULUM UPDATE ---
        current_max_nodes, physical_batch_size, accumulation_steps = get_curriculum_phase(epoch)
        
        # Re-initialize dataset to filter by current_max_nodes
        dataset = ExecutionGraphDataset(
            jsonl_path="dataset/compressed/compressed_motifs.jsonl", 
            max_nodes=current_max_nodes,
            augment_permutation=True
        )
        dataloader = DataLoader(dataset, batch_size=physical_batch_size, shuffle=True)
        
        print(f"\n=== Epoch {epoch}/{epochs} | Phase Max Nodes: {current_max_nodes} | Graphs: {len(dataset)} | Physical BS: {physical_batch_size} | Accumulation: {accumulation_steps} ===")
        
        # --- TRAINING LOOP (Continuous Flow Matching) ---
        model.train()
        total_loss = 0.0
        
        optimizer.zero_grad() # Move zero_grad outside the batch loop for accumulation
        
        progress_bar = tqdm(dataloader, desc="Training")
        for batch_idx, batch in enumerate(progress_bar):
            for k in ["adjacency", "motifs", "padding_mask"]:
                batch[k] = batch[k].to(device)
            
            # compute_flow_matching_loss handles t sampling, x_t interpolation, and masked MSE/CE
            loss = compute_flow_matching_loss(model, batch)
            
            # Scale loss by accumulation steps
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() # Keep track of unscaled loss for logging
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
