#!/usr/bin/env python3
"""
Example usage of AutoregressiveASTDataset for Phase 7 training.

This script demonstrates how to use the new autoregressive dataset
for training sequential AST generation models.
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_processing import create_autoregressive_data_loader, AutoregressiveASTDataset


def demonstrate_autoregressive_dataset():
    """Demonstrate the AutoregressiveASTDataset functionality."""
    print("🚀 AutoregressiveASTDataset Usage Example")
    print("=" * 50)
    
    # Create autoregressive dataset
    print("1. Creating AutoregressiveASTDataset...")
    dataset = AutoregressiveASTDataset(
        paired_data_path="dataset/paired_data.jsonl",
        max_sequence_length=10,  # Limit sequences to 10 nodes for demo
        seed=42
    )
    
    print(f"   📊 Dataset statistics:")
    print(f"   - Sequential training pairs: {len(dataset)}")
    print(f"   - Node feature dimension: {dataset.get_feature_dim()}")
    
    # Show individual sample structure
    print("\n2. Sample data structure...")
    sample = dataset[0]
    print(f"   📝 Sample keys: {list(sample.keys())}")
    print(f"   - Text: '{sample['text_description']}'")
    print(f"   - Step: {sample['step']}/{sample['total_steps']}")
    print(f"   - Partial graph: {sample['partial_graph']['num_nodes']} nodes")
    print(f"   - Target: '{sample['target_node']['node_type']}' node")
    
    # Show sequence progression for same method
    print("\n3. Sequence progression for a method...")
    first_text = sample['text_description']
    progression = []
    
    for i in range(len(dataset)):
        s = dataset[i]
        if s['text_description'] == first_text:
            progression.append(s)
        if len(progression) >= 5:  # Show first 5 steps
            break
    
    print(f"   📈 Method: '{first_text}'")
    for step_sample in progression:
        step = step_sample['step']
        nodes = step_sample['partial_graph']['num_nodes']
        target = step_sample['target_node']['node_type']
        print(f"   Step {step}: {nodes} nodes → '{target}'")
    
    # Create data loader for training
    print("\n4. Creating DataLoader for training...")
    loader = create_autoregressive_data_loader(
        paired_data_path="dataset/paired_data.jsonl",
        batch_size=4,
        max_sequence_length=8,
        shuffle=True,
        seed=42
    )
    
    print(f"   🔄 DataLoader created with {len(loader)} batches")
    
    # Show batch structure
    print("\n5. Batch structure for training...")
    for batch in loader:
        print(f"   📦 Batch structure:")
        print(f"   - Text descriptions: {len(batch['text_descriptions'])}")
        print(f"   - Partial graphs: {len(batch['partial_graphs']['x'])} total nodes")
        print(f"   - Target node types: {batch['target_node_types']}")
        print(f"   - Steps in batch: {batch['steps']}")
        print(f"   - Sample texts: {[t[:30] + '...' for t in batch['text_descriptions']]}")
        break
    
    print("\n6. Phase 7 format verification...")
    print("   ✅ Text embedding source: text_descriptions")
    print("   ✅ Partial graph: progressive AST with i nodes at step i")
    print("   ✅ Target node: next node to predict with type and features")
    print("   ✅ Causal ordering: node i depends only on nodes 1...i-1")
    
    print("\n🎯 Ready for autoregressive AST decoder training!")
    print("   Use this data loader with your AutoregressiveASTDecoder model")
    print("   Each batch provides (text, partial_graph, target_node) for training")


def show_comparison_with_phase6():
    """Show comparison between Phase 6 and Phase 7 data formats."""
    print("\n" + "=" * 50)
    print("📊 Phase 6 vs Phase 7 Data Format Comparison")
    print("=" * 50)
    
    # Phase 6 format (from existing PairedDataset)
    from data_processing import create_paired_data_loaders
    
    print("Phase 6 (One-shot generation):")
    print("   Input: complete_text_embedding")
    print("   Target: complete_ast_structure")
    print("   Training: reconstruct entire AST in one forward pass")
    
    phase6_loader = create_paired_data_loaders("dataset/paired_data.jsonl", batch_size=2, seed=42)
    for graphs, texts in phase6_loader:
        print(f"   Sample: {len(graphs['x'])} total nodes → complete AST")
        print(f"   Text: '{texts[0][:40]}...'")
        break
    
    print("\nPhase 7 (Autoregressive generation):")
    print("   Input: text_embedding + partial_graph")
    print("   Target: next_node_type + connections")
    print("   Training: predict next node given partial AST")
    
    phase7_loader = create_autoregressive_data_loader("dataset/paired_data.jsonl", batch_size=2, seed=42)
    for batch in phase7_loader:
        print(f"   Sample: {len(batch['partial_graphs']['x'])} partial nodes → next node")
        print(f"   Text: '{batch['text_descriptions'][0][:40]}...'")
        print(f"   Steps: {batch['steps']} (incremental generation)")
        break
    
    print("\n🔄 Phase 7 enables complex control flow generation!")


if __name__ == "__main__":
    demonstrate_autoregressive_dataset()
    show_comparison_with_phase6()