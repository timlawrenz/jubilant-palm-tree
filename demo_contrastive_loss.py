#!/usr/bin/env python3
"""
Demo script for contrastive loss functions.

This script demonstrates the usage of contrastive loss functions
for code-text embedding alignment in Phase 5.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from loss import info_nce_loss, cosine_embedding_loss, simple_contrastive_loss


def demo_contrastive_loss():
    """Demonstrate contrastive loss functions with example embeddings."""
    print("🚀 Contrastive Loss Functions Demo")
    print("=" * 50)
    
    # Simulate code and text embeddings
    batch_size = 4
    embedding_dim = 64
    
    print(f"Batch size: {batch_size}")
    print(f"Embedding dimension: {embedding_dim}")
    print()
    
    # Scenario 1: Well-aligned embeddings (similar vectors)
    print("📋 Scenario 1: Well-aligned embeddings")
    print("-" * 30)
    
    code_embeddings = torch.randn(batch_size, embedding_dim)
    # Text embeddings are similar to code embeddings (good alignment)
    text_embeddings = code_embeddings + 0.1 * torch.randn(batch_size, embedding_dim)
    
    info_nce = info_nce_loss(code_embeddings, text_embeddings)
    cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
    simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
    
    print(f"InfoNCE Loss:        {info_nce.item():.6f} (lower = better)")
    print(f"Cosine Emb Loss:     {cosine_emb.item():.6f} (lower = better)")
    print(f"Simple Contr Loss:   {simple_loss.item():.6f} (lower = better)")
    print()
    
    # Scenario 2: Poorly aligned embeddings (random vectors)
    print("📋 Scenario 2: Poorly aligned embeddings")
    print("-" * 30)
    
    code_embeddings = torch.randn(batch_size, embedding_dim)
    text_embeddings = torch.randn(batch_size, embedding_dim)  # Random, unrelated
    
    info_nce = info_nce_loss(code_embeddings, text_embeddings)
    cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
    simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
    
    print(f"InfoNCE Loss:        {info_nce.item():.6f} (higher = worse)")
    print(f"Cosine Emb Loss:     {cosine_emb.item():.6f} (higher = worse)")
    print(f"Simple Contr Loss:   {simple_loss.item():.6f} (variable)")
    print()
    
    # Scenario 3: Demonstrate training improvement simulation
    print("📋 Scenario 3: Training simulation")
    print("-" * 30)
    
    # Start with random embeddings
    code_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    text_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    print("Training simulation with InfoNCE loss:")
    optimizer = torch.optim.Adam([text_embeddings], lr=0.1)
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Compute loss
        loss = info_nce_loss(code_embeddings, text_embeddings)
        
        # Compute similarities before update
        similarities = F.cosine_similarity(code_embeddings, text_embeddings, dim=1)
        avg_similarity = similarities.mean()
        
        print(f"  Step {step}: Loss = {loss.item():.6f}, Avg Similarity = {avg_similarity.item():.6f}")
        
        # Backpropagate and update
        loss.backward()
        optimizer.step()
    
    print()
    
    # Temperature effect demonstration
    print("📋 Scenario 4: Temperature effect (InfoNCE)")
    print("-" * 30)
    
    code_embeddings = torch.randn(batch_size, embedding_dim)
    text_embeddings = code_embeddings + 0.2 * torch.randn(batch_size, embedding_dim)
    
    temperatures = [0.01, 0.1, 0.5, 1.0]
    for temp in temperatures:
        loss = info_nce_loss(code_embeddings, text_embeddings, temperature=temp)
        print(f"  Temperature {temp:4.2f}: Loss = {loss.item():.6f}")
    
    print()
    
    # Practical usage example
    print("📋 Practical Usage Example")
    print("-" * 30)
    
    print("""
# Example integration with training loop:

import torch
from src.loss import info_nce_loss

# Training loop
for batch in dataloader:
    # Get embeddings from your models
    code_emb = code_encoder(batch.graphs)      # Shape: (batch_size, 64)
    text_emb = text_encoder(batch.texts)       # Shape: (batch_size, 64)
    
    # Compute contrastive loss
    loss = info_nce_loss(code_emb, text_emb, temperature=0.07)
    
    # Standard training step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    """)


def demo_embedding_alignment():
    """Demonstrate how the loss functions encourage alignment."""
    print("🎯 Embedding Alignment Demonstration")
    print("=" * 50)
    
    # Create a scenario where we have specific code-text pairs
    batch_size = 3
    embedding_dim = 8  # Small dimension for easy visualization
    
    # Simulate three methods and their descriptions
    methods = [
        "calculate_total_price",
        "get_user_data", 
        "validate_input"
    ]
    
    descriptions = [
        "calculates the total price including tax",
        "retrieves user information from database",
        "validates the input parameters"
    ]
    
    print("Methods and descriptions:")
    for i, (method, desc) in enumerate(zip(methods, descriptions)):
        print(f"  {i}: {method} -> {desc}")
    print()
    
    # Create embeddings that represent the semantics
    # In practice, these would come from trained encoders
    code_embeddings = torch.tensor([
        [1.0, 0.5, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0],  # price calculation semantic
        [0.0, 0.0, 1.0, 0.5, 0.3, 0.0, 0.0, 0.0],  # data retrieval semantic  
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.8, 0.5],  # validation semantic
    ], dtype=torch.float)
    
    # Well-aligned text embeddings (similar to code embeddings)
    aligned_text_embeddings = torch.tensor([
        [0.9, 0.4, 0.3, 0.1, 0.1, 0.0, 0.0, 0.0],  # price-related
        [0.1, 0.0, 0.8, 0.6, 0.2, 0.0, 0.0, 0.0],  # data-related
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.9, 0.7, 0.4],  # validation-related
    ], dtype=torch.float)
    
    # Misaligned text embeddings (random)
    misaligned_text_embeddings = torch.tensor([
        [0.3, 0.1, 0.8, 0.2, 0.5, 0.1, 0.0, 0.0],  # doesn't match price
        [0.4, 0.6, 0.1, 0.0, 0.0, 0.7, 0.3, 0.2],  # doesn't match data
        [0.1, 0.5, 0.3, 0.8, 0.1, 0.0, 0.2, 0.1],  # doesn't match validation
    ], dtype=torch.float)
    
    # Compute losses for aligned embeddings
    print("Well-aligned embeddings:")
    aligned_info_nce = info_nce_loss(code_embeddings, aligned_text_embeddings)
    aligned_cosine = cosine_embedding_loss(code_embeddings, aligned_text_embeddings)
    
    print(f"  InfoNCE Loss:     {aligned_info_nce.item():.6f}")
    print(f"  Cosine Emb Loss:  {aligned_cosine.item():.6f}")
    
    # Compute similarities for aligned case
    aligned_similarities = F.cosine_similarity(code_embeddings, aligned_text_embeddings, dim=1)
    print(f"  Similarities:     {aligned_similarities.tolist()}")
    print()
    
    # Compute losses for misaligned embeddings  
    print("Misaligned embeddings:")
    misaligned_info_nce = info_nce_loss(code_embeddings, misaligned_text_embeddings)
    misaligned_cosine = cosine_embedding_loss(code_embeddings, misaligned_text_embeddings)
    
    print(f"  InfoNCE Loss:     {misaligned_info_nce.item():.6f}")
    print(f"  Cosine Emb Loss:  {misaligned_cosine.item():.6f}")
    
    # Compute similarities for misaligned case
    misaligned_similarities = F.cosine_similarity(code_embeddings, misaligned_text_embeddings, dim=1)
    print(f"  Similarities:     {misaligned_similarities.tolist()}")
    print()
    
    print("✅ Lower loss values indicate better alignment!")
    print("✅ Higher similarity values indicate better alignment!")


if __name__ == "__main__":
    demo_contrastive_loss()
    print()
    demo_embedding_alignment()
    
    print("\n🎉 Demo completed! The contrastive loss functions are ready for Phase 5 training.")