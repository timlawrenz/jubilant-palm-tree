#!/usr/bin/env python3
"""
Performance Demonstration Script

This script demonstrates the performance improvements from the CPU bottleneck optimizations.
It shows the difference between the old approach (repeated text encoding) and the new 
approach (pre-computed embeddings + optimized DataLoader).
"""

import time
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import AlignmentModel
from data_processing import create_autoregressive_data_loader


def simulate_old_training_approach():
    """Simulate the old training approach with repeated text encoding."""
    print("🔴 Old Approach: Repeated text encoding in training loop")
    print("-" * 60)
    
    # Load alignment model
    print("Loading AlignmentModel...")
    alignment_model = AlignmentModel(input_dim=74, hidden_dim=64)
    
    # Simulate training data
    unique_texts = [
        "method that processes user input data",
        "function to calculate mathematical results", 
        "helper method for data validation",
        "utility function for string processing",
        "algorithm for sorting data efficiently"
    ]
    
    print(f"Simulating training with {len(unique_texts)} unique text descriptions...")
    print("Each text will be encoded 20 times (simulating 20 epochs)...")
    
    start_time = time.time()
    
    # Simulate 20 epochs of training where each text is encoded multiple times
    for epoch in range(20):
        for text in unique_texts:
            # This is what happens in the old training loop - encode same text repeatedly!
            _ = alignment_model.encode_text([text])
    
    old_time = time.time() - start_time
    total_encodings = 20 * len(unique_texts)
    
    print(f"Total text encodings: {total_encodings}")
    print(f"Time taken: {old_time:.2f} seconds")
    print(f"Average time per encoding: {old_time/total_encodings:.3f}s")
    
    return old_time


def simulate_new_training_approach():
    """Simulate the new training approach with pre-computed embeddings."""
    print("\n🟢 New Approach: Pre-computed embeddings")
    print("-" * 60)
    
    # Load alignment model
    print("Loading AlignmentModel...")
    alignment_model = AlignmentModel(input_dim=74, hidden_dim=64)
    
    # Simulate training data
    unique_texts = [
        "method that processes user input data",
        "function to calculate mathematical results", 
        "helper method for data validation", 
        "utility function for string processing",
        "algorithm for sorting data efficiently"
    ]
    
    print(f"Pre-computing embeddings for {len(unique_texts)} unique texts...")
    
    start_time = time.time()
    
    # STEP 1: Pre-compute embeddings once (this is done before training)
    precomputed_embeddings = {}
    for text in unique_texts:
        precomputed_embeddings[text] = alignment_model.encode_text([text])
    
    precompute_time = time.time() - start_time
    print(f"Pre-computation time: {precompute_time:.2f}s")
    
    # STEP 2: Simulate 20 epochs of training using pre-computed embeddings
    print("Simulating 20 epochs of training with pre-computed embeddings...")
    
    training_start = time.time()
    
    for epoch in range(20):
        for text in unique_texts:
            # This is what happens in the new training loop - just lookup!
            _ = precomputed_embeddings[text]  # No computation, just memory lookup
    
    training_time = time.time() - training_start
    total_time = precompute_time + training_time
    
    print(f"Training time (20 epochs): {training_time:.3f}s")
    print(f"Total time (pre-compute + training): {total_time:.2f}s")
    
    return total_time


def demonstrate_dataloader_optimizations():
    """Demonstrate DataLoader optimizations."""
    print("\n🔧 DataLoader Optimizations")
    print("-" * 60)
    
    print("Testing optimized DataLoader configuration...")
    
    try:
        # Test the optimized data loader
        loader = create_autoregressive_data_loader(
            "dataset/train_paired_data.jsonl",
            batch_size=32,
            shuffle=True,
            max_sequence_length=20,
            precomputed_embeddings_path="output/text_embeddings.pt",
            num_workers=4,  # Use multiprocessing
            pin_memory=True  # Use pinned memory for GPU transfer
        )
        
        print("✅ Optimized DataLoader created successfully!")
        print("   - Multi-worker processing enabled")
        print("   - Pinned memory for faster GPU transfer")
        print("   - Persistent workers to avoid worker recreation")
        print("   - Prefetching for overlapped computation")
        
    except Exception as e:
        print(f"DataLoader test: {e}")


def main():
    """Main demonstration function."""
    print("🚀 CPU Bottleneck Optimization Demonstration")
    print("=" * 70)
    print()
    
    # Demonstrate text embedding optimization
    old_time = simulate_old_training_approach()
    new_time = simulate_new_training_approach()
    
    # Calculate improvements
    speedup = old_time / new_time if new_time > 0 else float('inf')
    time_saved = old_time - new_time
    
    print("\n📊 Performance Summary")
    print("=" * 30)
    print(f"Old approach time:    {old_time:.2f}s")
    print(f"New approach time:    {new_time:.2f}s") 
    print(f"Speedup:             {speedup:.1f}x faster")
    print(f"Time saved:          {time_saved:.2f}s")
    print(f"CPU efficiency:      {(time_saved/old_time)*100:.1f}% reduction")
    
    # Demonstrate DataLoader optimizations
    demonstrate_dataloader_optimizations()
    
    print("\n🎯 Key Benefits for GPU Training:")
    print("  • Eliminates CPU bottleneck from repeated text encoding")
    print("  • Frees up CPU cycles for data loading and preprocessing")
    print("  • Allows GPU to run at higher utilization")
    print("  • Reduces training time per epoch significantly")
    print("  • Scales better with larger datasets and vocabularies")
    
    print(f"\n✨ Overall result: Training should be ~{speedup:.0f}x faster with much better GPU utilization!")


if __name__ == "__main__":
    main()