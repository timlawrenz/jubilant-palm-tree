"""
Graph Neural Network models for Ruby code complexity prediction.

This module contains PyTorch Geometric models for learning from
Ruby AST structures.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class RubyComplexityGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting Ruby method complexity.
    
    This model uses Graph Convolutional Networks to learn from
    Abstract Syntax Tree representations of Ruby methods.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GCN layers
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            
        # Last layer
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output layer for complexity prediction
        self.predictor = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing graph
            
        Returns:
            Complexity prediction
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers with ReLU activation
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Predict complexity
        return self.predictor(x)