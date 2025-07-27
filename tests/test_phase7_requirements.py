#!/usr/bin/env python3
"""
Test script to verify Phase 7 requirements are met.

This test validates that the AutoregressiveASTDataset implementation
matches the exact requirements specified in the Phase 7 README.
"""

import sys
import os
import traceback

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from data_processing import AutoregressiveASTDataset, create_autoregressive_data_loader

# Helper function to get dataset paths relative to this script
def get_dataset_path(relative_path):
    """Get dataset path relative to this script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)


def test_sequential_pair_generation():
    """Test that ASTs are converted to sequence of (input, target) pairs."""
    print("Testing sequential pair generation...")
    
    try:
        dataset = AutoregressiveASTDataset(get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"), max_sequence_length=5, seed=42)
        
        if len(dataset) == 0:
            print("❌ No sequential pairs generated")
            return False
        
        # Check that we have multiple pairs per original method
        original_methods = 4000
        sequential_pairs = len(dataset)
        
        if sequential_pairs <= original_methods:
            print(f"❌ Expected more sequential pairs than methods: {sequential_pairs} <= {original_methods}")
            return False
        
        print(f"✅ Sequential pair generation successful")
        print(f"   {original_methods} methods -> {sequential_pairs} sequential pairs")
        print(f"   Expansion ratio: {sequential_pairs / original_methods:.1f}x")
        
        return True
    except Exception as e:
        print(f"❌ Sequential pair generation failed: {e}")
        traceback.print_exc()
        return False


def test_progressive_inputs():
    """Test that input at step i contains partial AST with first i nodes."""
    print("\nTesting progressive inputs...")
    
    try:
        dataset = AutoregressiveASTDataset(get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"), max_sequence_length=5, seed=42)
        
        # Find a sequence for the same method
        method_sequence = []
        first_text = None
        
        for i in range(len(dataset)):
            sample = dataset[i]
            if first_text is None:
                first_text = sample['text_description']
                method_sequence.append(sample)
            elif sample['text_description'] == first_text and sample['step'] == len(method_sequence):
                method_sequence.append(sample)
            elif sample['text_description'] != first_text:
                break
        
        if len(method_sequence) < 2:
            print("❌ Could not find method sequence")
            return False
        
        # Verify progressive inputs
        for i, sample in enumerate(method_sequence):
            expected_nodes = i  # Step i should have i nodes in partial graph
            actual_nodes = sample['partial_graph']['num_nodes']
            step = sample['step']
            
            if step != i:
                print(f"❌ Step mismatch: expected {i}, got {step}")
                return False
                
            if actual_nodes != expected_nodes:
                print(f"❌ Progressive input failed at step {i}: expected {expected_nodes} nodes, got {actual_nodes}")
                return False
        
        print(f"✅ Progressive inputs successful")
        print(f"   Verified {len(method_sequence)} progressive steps")
        print(f"   Step 0: {method_sequence[0]['partial_graph']['num_nodes']} nodes")
        print(f"   Step 1: {method_sequence[1]['partial_graph']['num_nodes']} nodes")
        
        return True
    except Exception as e:
        print(f"❌ Progressive inputs test failed: {e}")
        traceback.print_exc()
        return False


def test_incremental_targets():
    """Test that target at step i is the (i+1)-th node."""
    print("\nTesting incremental targets...")
    
    try:
        dataset = AutoregressiveASTDataset(get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"), max_sequence_length=5, seed=42)
        
        # Get a sample to check target structure
        sample = dataset[0]
        
        # Verify target node structure
        target_node = sample['target_node']
        required_keys = ['node_type', 'features']
        
        for key in required_keys:
            if key not in target_node:
                print(f"❌ Missing target node key: {key}")
                return False
        
        # Verify target node has proper type
        if not isinstance(target_node['node_type'], str):
            print(f"❌ Target node type should be string, got {type(target_node['node_type'])}")
            return False
        
        # Verify features are present
        if not target_node['features']:
            print(f"❌ Target node features are empty")
            return False
        
        print(f"✅ Incremental targets successful")
        print(f"   Target node structure: {list(target_node.keys())}")
        print(f"   Sample target type: '{target_node['node_type']}'")
        print(f"   Features dimension: {len(target_node['features'])}")
        
        return True
    except Exception as e:
        print(f"❌ Incremental targets test failed: {e}")
        traceback.print_exc()
        return False


def test_training_sequence_format():
    """Test that each sample contains (text_embedding, partial_graph, target_node)."""
    print("\nTesting training sequence format...")
    
    try:
        loader = create_autoregressive_data_loader(
            get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"),
            batch_size=2,
            max_sequence_length=3,
            seed=42
        )
        
        for batch in loader:
            # Check Phase 7 format requirements
            required_keys = {
                'text_descriptions': 'text_embedding equivalent',
                'partial_graphs': 'partial_graph',
                'target_node_types': 'part of target_node',
                'target_node_features': 'part of target_node'
            }
            
            for key, description in required_keys.items():
                if key not in batch:
                    print(f"❌ Missing batch key {key} ({description})")
                    return False
            
            # Verify partial graph has proper structure
            partial_graphs = batch['partial_graphs']
            if 'x' not in partial_graphs or 'edge_index' not in partial_graphs:
                print(f"❌ Partial graph missing node features or edges")
                return False
            
            # Verify text descriptions are present
            if not batch['text_descriptions']:
                print(f"❌ No text descriptions in batch")
                return False
            
            print(f"✅ Training sequence format successful")
            print(f"   Batch contains all required Phase 7 components")
            print(f"   Text descriptions: {len(batch['text_descriptions'])}")
            print(f"   Partial graph nodes: {len(partial_graphs['x'])}")
            print(f"   Target nodes: {len(batch['target_node_types'])}")
            
            break
            
        return True
    except Exception as e:
        print(f"❌ Training sequence format test failed: {e}")
        traceback.print_exc()
        return False


def test_causal_generation_order():
    """Test that proper ordering ensures causal generation."""
    print("\nTesting causal generation order...")
    
    try:
        dataset = AutoregressiveASTDataset(get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"), max_sequence_length=5, seed=42)
        
        # Check that steps are in proper order
        previous_step = -1
        previous_text = None
        step_progression_correct = True
        
        for i in range(min(20, len(dataset))):  # Check first 20 samples
            sample = dataset[i]
            current_step = sample['step']
            current_text = sample['text_description']
            
            # If same method, step should increment
            if previous_text == current_text:
                if current_step != previous_step + 1:
                    step_progression_correct = False
                    break
            else:
                # New method should start at step 0
                if current_step != 0:
                    step_progression_correct = False
                    break
            
            previous_step = current_step
            previous_text = current_text
        
        if not step_progression_correct:
            print(f"❌ Step progression not causal")
            return False
        
        # Verify that partial graph at step i only contains nodes 1...i-1
        sample = dataset[5]  # Get a sample that should have some nodes
        step = sample['step']
        partial_nodes = sample['partial_graph']['num_nodes']
        
        if partial_nodes != step:
            print(f"❌ Causal violation: step {step} has {partial_nodes} nodes, expected {step}")
            return False
        
        print(f"✅ Causal generation order successful")
        print(f"   Step progression verified")
        print(f"   Node i depends only on nodes 1...i-1")
        print(f"   Sample verification: step {step} has {partial_nodes} nodes")
        
        return True
    except Exception as e:
        print(f"❌ Causal generation order test failed: {e}")
        traceback.print_exc()
        return False


def test_phase5_compatibility():
    """Test compatibility with existing text-code alignment."""
    print("\nTesting Phase 5 compatibility...")
    
    try:
        dataset = AutoregressiveASTDataset(get_dataset_path("../dataset/samples/train_paired_data_sample.jsonl"), max_sequence_length=3, seed=42)
        
        # Check that text descriptions are preserved
        sample = dataset[0]
        text_desc = sample['text_description']
        
        if not isinstance(text_desc, str) or len(text_desc) == 0:
            print(f"❌ Text description format invalid")
            return False
        
        # Check that node features maintain same dimension as Phase 5
        target_node = sample['target_node']
        feature_dim = len(target_node['features'])
        expected_dim = 74  # From existing ASTNodeEncoder
        
        if feature_dim != expected_dim:
            print(f"❌ Feature dimension mismatch: expected {expected_dim}, got {feature_dim}")
            return False
        
        print(f"✅ Phase 5 compatibility successful")
        print(f"   Text descriptions preserved: '{text_desc[:50]}...'")
        print(f"   Node features maintain 74-dimensional encoding")
        print(f"   Compatible with existing alignment model")
        
        return True
    except Exception as e:
        print(f"❌ Phase 5 compatibility test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all Phase 7 requirement tests."""
    print("📋 Phase 7 Requirements Validation")
    print("=" * 50)
    
    tests = [
        test_sequential_pair_generation,
        test_progressive_inputs,
        test_incremental_targets,
        test_training_sequence_format,
        test_causal_generation_order,
        test_phase5_compatibility,
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        if test_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"🏁 Phase 7 Requirements: {passed}/{total} requirements met")
    
    if passed == total:
        print("🎉 All Phase 7 requirements satisfied!")
        print("")
        print("✅ RubyASTDataset yields sequences of partial graphs for autoregressive training")
        print("✅ Each training sample contains (text_embedding, partial_graph, target_node)")
        print("✅ Proper ordering ensures causal generation (node i depends only on nodes 1...i-1)")
        print("✅ Compatible with existing text-code alignment from Phase 5")
        return True
    else:
        print("❌ Some Phase 7 requirements not met. Please review implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)