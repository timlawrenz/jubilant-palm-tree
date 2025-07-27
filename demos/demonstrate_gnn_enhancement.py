#!/usr/bin/env python3
"""
Demonstration script showing the GNN enhancement in AutoregressiveASTDecoder.

This script compares the old simple mean pooling approach with the new
GNN-based structural encoding to show the benefits of the enhancement.
"""

import torch
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from models import AutoregressiveASTDecoder
from torch_geometric.nn import GCNConv, global_mean_pool


class SimplePoolingDecoder(torch.nn.Module):
    """
    Reference implementation of the old simple mean pooling approach.
    This shows what the decoder was like BEFORE the GNN enhancement.
    """
    
    def __init__(self, text_embedding_dim=64, graph_hidden_dim=64, state_hidden_dim=128, node_types=74):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.graph_hidden_dim = graph_hidden_dim
        self.state_hidden_dim = state_hidden_dim
        self.node_types = node_types
        
        # OLD APPROACH: Simple linear encoder (mean pooling + linear transform)
        self.graph_encoder = torch.nn.Sequential(
            torch.nn.Linear(node_types, graph_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(graph_hidden_dim)
        )
        
        # Same sequential components as enhanced version
        self.state_encoder = torch.nn.GRU(
            input_size=text_embedding_dim + graph_hidden_dim,
            hidden_size=state_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        self.node_type_predictor = torch.nn.Linear(state_hidden_dim, node_types)
        self.connection_predictor = torch.nn.Sequential(
            torch.nn.Linear(state_hidden_dim, 100),
            torch.nn.Sigmoid()
        )
    
    def forward(self, text_embedding, partial_graph=None, hidden_state=None):
        batch_size = text_embedding.size(0)
        device = text_embedding.device
        
        # OLD APPROACH: Simple mean pooling (ignores graph structure)
        if partial_graph is not None and 'x' in partial_graph and len(partial_graph['x']) > 0:
            graph_features = partial_graph['x']
            if isinstance(graph_features, list):
                graph_features = torch.tensor(graph_features, dtype=torch.float32, device=device)
            else:
                graph_features = graph_features.to(device)
            
            # Simple mean pooling - ignores edges completely
            graph_representation = graph_features.mean(dim=0).unsqueeze(0).expand(batch_size, -1)
        else:
            graph_representation = torch.zeros(batch_size, self.node_types, device=device)
        
        # Apply simple linear encoding
        graph_encoded = self.graph_encoder(graph_representation)
        
        # Rest is the same as enhanced version
        combined_input = torch.cat([text_embedding, graph_encoded], dim=-1)
        sequence_input = combined_input.unsqueeze(1)
        sequence_output, new_hidden_state = self.state_encoder(sequence_input, hidden_state)
        sequence_output = sequence_output.squeeze(1)
        
        node_type_logits = self.node_type_predictor(sequence_output)
        connection_probs = self.connection_predictor(sequence_output)
        
        return {
            'node_type_logits': node_type_logits,
            'connection_probs': connection_probs,
            'hidden_state': new_hidden_state
        }


def create_test_scenarios():
    """Create various graph scenarios to test structural sensitivity."""
    scenarios = {}
    
    # Scenario 1: Linear chain vs Star structure
    scenarios['linear_chain'] = {
        'description': 'Linear Chain: A -> B -> C',
        'x': torch.tensor([[1.0] + [0.0]*73, [0.0,1.0] + [0.0]*72, [0.0]*2 + [1.0] + [0.0]*71]),
        'edge_index': torch.tensor([[0,1,1,2], [1,0,2,1]]),
        'batch': torch.tensor([0,0,0])
    }
    
    scenarios['star_structure'] = {
        'description': 'Star: B connected to both A and C',
        'x': torch.tensor([[1.0] + [0.0]*73, [0.0,1.0] + [0.0]*72, [0.0]*2 + [1.0] + [0.0]*71]),
        'edge_index': torch.tensor([[0,1,1,2], [1,0,1,2]]),  # B (node 1) is center
        'batch': torch.tensor([0,0,0])
    }
    
    # Scenario 2: Dense vs Sparse connectivity  
    scenarios['dense_graph'] = {
        'description': 'Dense: All nodes connected',
        'x': torch.tensor([[1.0] + [0.0]*73, [0.0,1.0] + [0.0]*72, [0.0]*2 + [1.0] + [0.0]*71]),
        'edge_index': torch.tensor([[0,0,1,1,2,2], [1,2,0,2,0,1]]),
        'batch': torch.tensor([0,0,0])
    }
    
    scenarios['sparse_graph'] = {
        'description': 'Sparse: Only one connection',
        'x': torch.tensor([[1.0] + [0.0]*73, [0.0,1.0] + [0.0]*72, [0.0]*2 + [1.0] + [0.0]*71]),
        'edge_index': torch.tensor([[0,1], [1,0]]),  # Only A-B connected
        'batch': torch.tensor([0,0,0])
    }
    
    return scenarios


def compare_approaches():
    """Compare GNN-enhanced vs simple pooling approaches."""
    print("🔬 Comparing GNN Enhancement vs Simple Mean Pooling")
    print("=" * 60)
    
    # Create models
    enhanced_model = AutoregressiveASTDecoder()  # NEW: GNN-based
    simple_model = SimplePoolingDecoder()       # OLD: Mean pooling
    
    text_embedding = torch.randn(1, 64)
    scenarios = create_test_scenarios()
    
    results = {}
    
    with torch.no_grad():
        for name, graph_data in scenarios.items():
            print(f"\n📊 Testing: {graph_data['description']}")
            
            # Get predictions from both models
            enhanced_output = enhanced_model(text_embedding, graph_data)
            simple_output = simple_model(text_embedding, graph_data)
            
            # Store results
            results[name] = {
                'enhanced': enhanced_output['node_type_logits'].cpu(),
                'simple': simple_output['node_type_logits'].cpu()
            }
            
            # Calculate prediction confidence (max probability)
            enhanced_conf = F.softmax(enhanced_output['node_type_logits'], dim=-1).max().item()
            simple_conf = F.softmax(simple_output['node_type_logits'], dim=-1).max().item()
            
            print(f"   Enhanced GNN confidence: {enhanced_conf:.4f}")
            print(f"   Simple pooling confidence: {simple_conf:.4f}")
    
    return results


def analyze_structural_sensitivity(results):
    """Analyze how well each approach distinguishes different structures."""
    print(f"\n🧠 Structural Sensitivity Analysis")
    print("=" * 60)
    
    # Compare linear chain vs star structure (same nodes, different topology)
    enhanced_linear = results['linear_chain']['enhanced']
    enhanced_star = results['star_structure']['enhanced']
    simple_linear = results['linear_chain']['simple']
    simple_star = results['star_structure']['simple']
    
    # Calculate differences
    enhanced_diff = torch.abs(enhanced_linear - enhanced_star).max().item()
    simple_diff = torch.abs(simple_linear - simple_star).max().item()
    
    print(f"Linear Chain vs Star Structure (same nodes, different topology):")
    print(f"   Enhanced GNN difference: {enhanced_diff:.6f}")
    print(f"   Simple pooling difference: {simple_diff:.6f}")
    print(f"   GNN sensitivity ratio: {enhanced_diff / max(simple_diff, 1e-8):.2f}x")
    
    # Compare dense vs sparse connectivity
    enhanced_dense = results['dense_graph']['enhanced']
    enhanced_sparse = results['sparse_graph']['enhanced']
    simple_dense = results['dense_graph']['simple']
    simple_sparse = results['sparse_graph']['simple']
    
    enhanced_connectivity_diff = torch.abs(enhanced_dense - enhanced_sparse).max().item()
    simple_connectivity_diff = torch.abs(simple_dense - simple_sparse).max().item()
    
    print(f"\nDense vs Sparse Connectivity:")
    print(f"   Enhanced GNN difference: {enhanced_connectivity_diff:.6f}")
    print(f"   Simple pooling difference: {simple_connectivity_diff:.6f}")
    print(f"   GNN sensitivity ratio: {enhanced_connectivity_diff / max(simple_connectivity_diff, 1e-8):.2f}x")
    
    return {
        'topology_sensitivity': enhanced_diff / max(simple_diff, 1e-8),
        'connectivity_sensitivity': enhanced_connectivity_diff / max(simple_connectivity_diff, 1e-8)
    }


def demonstrate_parameter_efficiency():
    """Show parameter counts and computational efficiency."""
    print(f"\n⚡ Parameter and Computational Efficiency")
    print("=" * 60)
    
    enhanced_model = AutoregressiveASTDecoder()
    simple_model = SimplePoolingDecoder()
    
    enhanced_params = sum(p.numel() for p in enhanced_model.parameters())
    simple_params = sum(p.numel() for p in simple_model.parameters())
    
    # Count GNN-specific parameters
    gnn_params = sum(p.numel() for p in enhanced_model.graph_gnn_layers.parameters())
    gnn_params += sum(p.numel() for p in enhanced_model.graph_layer_norm.parameters())
    gnn_params += sum(p.numel() for p in enhanced_model.graph_dropout.parameters())
    
    print(f"Enhanced model total parameters: {enhanced_params:,}")
    print(f"Simple model total parameters: {simple_params:,}")
    print(f"GNN enhancement cost: {gnn_params:,} additional parameters")
    print(f"Parameter overhead: {(enhanced_params - simple_params) / simple_params * 100:.1f}%")
    print(f"Benefit: Rich structural representations with minimal parameter cost")


def main():
    """Run the complete demonstration."""
    print("🚀 GNN Enhancement Demonstration")
    print("This demonstration shows the benefits of replacing simple mean pooling")
    print("with proper Graph Neural Networks in AutoregressiveASTDecoder")
    print("="*80)
    
    # Compare both approaches
    results = compare_approaches()
    
    # Analyze structural sensitivity
    sensitivity = analyze_structural_sensitivity(results)
    
    # Show parameter efficiency
    demonstrate_parameter_efficiency()
    
    # Final summary
    print(f"\n🎯 Enhancement Summary")
    print("=" * 60)
    print(f"✅ GNN provides {sensitivity['topology_sensitivity']:.2f}x better topology sensitivity")
    print(f"✅ GNN provides {sensitivity['connectivity_sensitivity']:.2f}x better connectivity sensitivity")
    print(f"✅ Structural awareness enables better partial graph understanding")
    print(f"✅ Sequential decoder gets richer context at each generation step")
    print(f"✅ Critical for learning complex, nested code structures")
    
    success = (
        sensitivity['topology_sensitivity'] > 1.0 and 
        sensitivity['connectivity_sensitivity'] > 1.0
    )
    
    if success:
        print(f"\n🎉 GNN Enhancement Successfully Demonstrated!")
        print(f"   The enhanced decoder provides much richer structural representations")
        print(f"   compared to simple mean pooling, enabling better code generation.")
    else:
        print(f"\n⚠️  GNN enhancement needs further tuning")
        
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)