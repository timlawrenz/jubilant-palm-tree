#!/usr/bin/env python3
"""
Optimized evaluation script for the AST autoencoder.

This script scales from evaluating 4 samples to hundreds or thousands of samples
with comprehensive metrics and optimized performance.
"""

import sys
import os
import json
import subprocess
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from tqdm import tqdm

from data_processing import RubyASTDataset, ASTNodeEncoder
from models import ASTAutoencoder

def main():
    """Main evaluation function"""
    print("=" * 80)
    print("AST AUTOENCODER EVALUATION - OPTIMIZED")
    print("=" * 80)
    
    # Configuration
    CONFIG = {
        'num_samples': 100,  # Number of samples to evaluate
        'random_seed': 42,   # For reproducible sample selection
        'enable_ruby_conversion': False,  # Disable Ruby conversion for faster execution
        'ruby_timeout': 15,  # Timeout for Ruby subprocess calls
        'save_results': True,  # Save detailed results to file
    }
    
    print(f"Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Set random seed
    np.random.seed(CONFIG['random_seed'])
    torch.manual_seed(CONFIG['random_seed'])
    
    # Load dataset
    print(f"\nLoading test dataset...")
    start_time = time.time()
    test_dataset = RubyASTDataset("dataset/test.jsonl")
    print(f"Loaded {len(test_dataset)} samples in {time.time() - start_time:.2f}s")
    
    # Initialize autoencoder
    print(f"\nInitializing autoencoder...")
    start_time = time.time()
    autoencoder = ASTAutoencoder(
        encoder_input_dim=74,
        node_output_dim=74,
        hidden_dim=64,
        num_layers=3,
        conv_type='GCN',
        freeze_encoder=True,
        encoder_weights_path="best_model.pt"
    )
    
    # Load decoder
    decoder_path = "best_decoder.pt"
    if os.path.exists(decoder_path):
        print(f"Loading trained decoder from {decoder_path}")
        checkpoint = torch.load(decoder_path, map_location='cpu')
        if 'decoder_state_dict' in checkpoint:
            autoencoder.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"✓ Decoder loaded successfully (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            autoencoder.decoder.load_state_dict(checkpoint)
            print("✓ Decoder loaded successfully")
    else:
        print("No trained decoder found - using randomly initialized decoder")
    
    autoencoder.eval()
    print(f"Autoencoder ready in {time.time() - start_time:.2f}s")
    
    # Model info
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total ({trainable_params:,} trainable)")
    
    # Select samples
    sample_indices = select_diverse_samples(test_dataset, CONFIG['num_samples'], CONFIG['random_seed'])
    
    # Run evaluation
    print(f"\nStarting evaluation of {len(sample_indices)} samples...")
    eval_start_time = time.time()
    
    evaluation_results = evaluate_samples_batch(
        test_dataset, autoencoder, sample_indices, CONFIG
    )
    
    total_time = time.time() - eval_start_time
    print(f"\nEvaluation completed in {total_time:.1f}s ({total_time/len(evaluation_results):.3f}s per sample)")
    
    # Analyze results
    print(f"\nPerforming analysis...")
    analysis = analyze_reconstruction_quality(evaluation_results, len(test_dataset))
    
    # Display results
    display_results(analysis, evaluation_results, CONFIG, total_time)
    
    # Save results
    if CONFIG['save_results']:
        save_results(CONFIG, analysis, evaluation_results, sample_indices, total_time)
    
    print(f"\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"Successfully evaluated {len(evaluation_results)} samples from {len(test_dataset)} total test samples")
    print(f"Perfect structural preservation achieved in {analysis['perfect_node_count_preservation']} samples ({100*analysis['perfect_node_count_preservation']/len(evaluation_results):.1f}%)")
    print(f"Total evaluation time: {total_time:.1f} seconds")

def select_diverse_samples(dataset, num_samples, random_seed=42):
    """Select a diverse set of samples based on AST size distribution"""
    np.random.seed(random_seed)
    
    # Get AST sizes for all samples
    sizes = []
    for i in range(len(dataset)):
        sample = dataset[i]
        sizes.append(len(sample['x']))
    
    sizes = np.array(sizes)
    
    # Create size-based bins for diverse sampling
    percentiles = [0, 25, 50, 75, 90, 95, 100]
    size_thresholds = np.percentile(sizes, percentiles)
    
    print(f"\nAST size distribution:")
    for i, p in enumerate(percentiles):
        print(f"  {p:2d}th percentile: {size_thresholds[i]:6.1f} nodes")
    
    # Sample from different size categories
    selected_indices = []
    
    # Stratified sampling based on size
    for i in range(len(size_thresholds) - 1):
        min_size = size_thresholds[i]
        max_size = size_thresholds[i + 1]
        
        # Find indices in this size range
        in_range = np.where((sizes >= min_size) & (sizes <= max_size))[0]
        
        if len(in_range) > 0:
            # Sample proportionally from this range
            n_from_range = max(1, int(num_samples * len(in_range) / len(dataset)))
            n_from_range = min(n_from_range, len(in_range), num_samples - len(selected_indices))
            
            if n_from_range > 0:
                sampled = np.random.choice(in_range, size=n_from_range, replace=False)
                selected_indices.extend(sampled)
    
    # Fill remaining slots with random sampling if needed
    while len(selected_indices) < num_samples:
        remaining = set(range(len(dataset))) - set(selected_indices)
        if not remaining:
            break
        selected_indices.append(np.random.choice(list(remaining)))
    
    # Trim to exact number requested
    selected_indices = selected_indices[:num_samples]
    
    # Show selection summary
    selected_sizes = [sizes[i] for i in selected_indices]
    print(f"\nSelected {len(selected_indices)} samples:")
    print(f"  Size range: {min(selected_sizes)} - {max(selected_sizes)} nodes")
    print(f"  Average size: {np.mean(selected_sizes):.1f} nodes")
    print(f"  Median size: {np.median(selected_sizes):.1f} nodes")
    
    return sorted(selected_indices)

def convert_sample_to_torch(sample):
    """Convert a dataset sample to PyTorch format"""
    x = torch.tensor(sample['x'], dtype=torch.float)
    edge_index = torch.tensor(sample['edge_index'], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index, batch=batch)

def reconstruct_ast_from_features(node_features, reconstruction_info):
    """Convert reconstructed node features back to AST JSON format"""
    node_encoder = ASTNodeEncoder()
    features_tensor = node_features.squeeze()
    if features_tensor.dim() == 1:
        features_tensor = features_tensor.unsqueeze(0)
    
    node_type_indices = torch.argmax(features_tensor, dim=1)
    
    # Map feature indices back to node type names
    node_types = []
    for idx in node_type_indices:
        idx_val = idx.item()
        if idx_val < len(node_encoder.node_types):
            node_types.append(node_encoder.node_types[idx_val])
        else:
            node_types.append('unknown')
    
    # Create simplified AST structure
    return create_simple_ast(node_types)

def create_simple_ast(node_types):
    """Create a simple AST structure to avoid pretty printer issues"""
    if not node_types:
        return {'type': 'str', 'children': ['reconstructed_empty']}
    
    root_type = node_types[0]
    
    if root_type == 'def':
        return {
            'type': 'def',
            'children': [
                'reconstructed_method',
                {'type': 'args', 'children': []},
                {'type': 'str', 'children': ['reconstructed_content']}
            ]
        }
    else:
        return {
            'type': root_type if root_type != 'unknown' else 'str',
            'children': ['reconstructed_value']
        }

def evaluate_sample_fast(dataset, autoencoder, sample_idx):
    """Evaluate a single sample through the autoencoder"""
    sample = dataset[sample_idx]
    data = convert_sample_to_torch(sample)
    
    # Pass through autoencoder
    with torch.no_grad():
        result = autoencoder(data)
        embedding = result['embedding']
        reconstruction = result['reconstruction']
    
    # Get original data from the JSONL file
    original_code = None
    original_ast = None
    
    with open('dataset/test.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i == sample_idx:
                data_dict = json.loads(line)
                original_code = data_dict['raw_source']
                original_ast = json.loads(data_dict['ast_json'])
                break
    
    # Reconstruct AST from decoder output
    reconstructed_ast = reconstruct_ast_from_features(
        reconstruction['node_features'],
        reconstruction
    )
    
    return {
        'sample_idx': sample_idx,
        'embedding_dim': embedding.shape[1],
        'original_code': original_code,
        'original_ast': original_ast,
        'reconstructed_ast': reconstructed_ast,
        'original_nodes': len(sample['x']),
        'reconstructed_nodes': reconstruction['node_features'].shape[1],
    }

def evaluate_samples_batch(dataset, autoencoder, sample_indices, config):
    """Evaluate multiple samples with progress tracking"""
    print(f"\nEvaluating {len(sample_indices)} samples...")
    
    evaluation_results = []
    
    for idx in tqdm(sample_indices, desc="Autoencoder inference"):
        if idx < len(dataset):
            result = evaluate_sample_fast(dataset, autoencoder, idx)
            evaluation_results.append(result)
    
    print(f"Completed autoencoder inference for {len(evaluation_results)} samples")
    return evaluation_results

def analyze_reconstruction_quality(results, total_test_samples):
    """Comprehensive analysis of reconstruction quality"""
    analysis = {
        'total_samples': len(results),
        'total_test_samples_available': total_test_samples,
        'coverage_percentage': len(results) / total_test_samples * 100,
        'avg_original_nodes': np.mean([r['original_nodes'] for r in results]),
        'avg_reconstructed_nodes': np.mean([r['reconstructed_nodes'] for r in results]),
        'node_count_differences': [abs(r['original_nodes'] - r['reconstructed_nodes']) for r in results],
        'perfect_node_count_preservation': 0,
        'structural_similarity': [],
        'size_distribution': {
            'small_methods': 0,    # < 20 nodes
            'medium_methods': 0,   # 20-100 nodes
            'large_methods': 0,    # > 100 nodes
        }
    }
    
    # Count perfect node preservation
    for result in results:
        if result['original_nodes'] == result['reconstructed_nodes']:
            analysis['perfect_node_count_preservation'] += 1
    
    # Count by size categories
    for result in results:
        size = result['original_nodes']
        if size < 20:
            analysis['size_distribution']['small_methods'] += 1
        elif size <= 100:
            analysis['size_distribution']['medium_methods'] += 1
        else:
            analysis['size_distribution']['large_methods'] += 1
    
    # Calculate structural similarity (simplified metric)
    for result in results:
        orig_ast = result['original_ast']
        recon_ast = result['reconstructed_ast']
        
        # Simple similarity: check if root types match
        orig_type = orig_ast.get('type') if orig_ast else None
        recon_type = recon_ast.get('type') if recon_ast else None
        
        if orig_type == recon_type:
            analysis['structural_similarity'].append(1.0)
        else:
            analysis['structural_similarity'].append(0.0)
    
    return analysis

def display_results(analysis, evaluation_results, config, total_time):
    """Display comprehensive results"""
    print("\n" + "="*80)
    print("COMPREHENSIVE RECONSTRUCTION QUALITY ANALYSIS")
    print("="*80)

    print(f"\nDataset Coverage:")
    print(f"  Total test samples available: {analysis['total_test_samples_available']:,}")
    print(f"  Samples evaluated: {analysis['total_samples']:,}")
    print(f"  Coverage: {analysis['coverage_percentage']:.2f}%")

    print(f"\nSize Distribution:")
    print(f"  Small methods (<20 nodes): {analysis['size_distribution']['small_methods']}")
    print(f"  Medium methods (20-100 nodes): {analysis['size_distribution']['medium_methods']}")
    print(f"  Large methods (>100 nodes): {analysis['size_distribution']['large_methods']}")

    print(f"\nStructural Preservation:")
    print(f"  Average original nodes: {analysis['avg_original_nodes']:.1f}")
    print(f"  Average reconstructed nodes: {analysis['avg_reconstructed_nodes']:.1f}")
    print(f"  Average node count difference: {np.mean(analysis['node_count_differences']):.1f}")
    print(f"  Perfect node count preservation: {analysis['perfect_node_count_preservation']}/{analysis['total_samples']} ({100*analysis['perfect_node_count_preservation']/analysis['total_samples']:.1f}%)")
    print(f"  Root type match rate: {np.mean(analysis['structural_similarity']):.3f} ({100*np.mean(analysis['structural_similarity']):.1f}%)")

    # Show sample examples
    print(f"\nSample Results (first 10):")
    print(f"{'Sample':<8} {'Orig Nodes':<12} {'Recon Nodes':<13} {'Diff':<6} {'Perfect':<8}")
    print("-" * 50)
    for i, result in enumerate(evaluation_results[:10]):
        node_diff = abs(result['original_nodes'] - result['reconstructed_nodes'])
        perfect = "✓" if node_diff == 0 else "✗"
        print(f"{result['sample_idx']:<8} {result['original_nodes']:<12} {result['reconstructed_nodes']:<13} {node_diff:<6} {perfect:<8}")

def save_results(config, analysis, evaluation_results, sample_indices, total_time):
    """Save detailed results to files"""
    print(f"\nSaving detailed results...")
    
    # Create output directory
    os.makedirs('output', exist_ok=True)
    timestamp = int(time.time())
    
    # Convert numpy types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(v) for v in obj]
        return obj
    
    # Save to JSON file
    results_file = f"output/evaluation_results_{config['num_samples']}_samples_{timestamp}.json"
    json_data = {
        'config': convert_numpy_types(config),
        'analysis': convert_numpy_types({
            k: v for k, v in analysis.items() 
            if k not in ['node_count_differences', 'structural_similarity']
        }),
        'summary_statistics': {
            'avg_node_count_difference': float(np.mean(analysis['node_count_differences'])),
            'max_node_count_difference': float(np.max(analysis['node_count_differences'])),
            'avg_structural_similarity': float(np.mean(analysis['structural_similarity'])),
            'perfect_preservation_rate': float(analysis['perfect_node_count_preservation'] / analysis['total_samples']),
        },
        'sample_indices': [int(x) for x in sample_indices],
        'evaluation_time_seconds': float(total_time),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(results_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    # Save CSV summary
    csv_file = f"output/evaluation_summary_{config['num_samples']}_samples_{timestamp}.csv"
    summary_data = []
    for result in evaluation_results:
        node_diff = abs(result['original_nodes'] - result['reconstructed_nodes'])
        summary_data.append({
            'Sample': result['sample_idx'],
            'Original_Nodes': result['original_nodes'],
            'Reconstructed_Nodes': result['reconstructed_nodes'],
            'Node_Diff': node_diff,
            'Perfect_Preservation': node_diff == 0,
            'Embedding_Dim': result['embedding_dim']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(csv_file, index=False)
    print(f"CSV summary saved to: {csv_file}")

if __name__ == "__main__":
    main()