"""
Graph Neural Network models for Ruby code complexity prediction.

This module contains PyTorch Geometric models for learning from
Ruby AST structures.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch


class RubyComplexityGNN(torch.nn.Module):
    """
    Graph Neural Network for predicting Ruby method complexity.
    
    This model uses Graph Convolutional Networks (GCN) or GraphSAGE layers
    to learn from Abstract Syntax Tree representations of Ruby methods.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, 
                 conv_type: str = 'GCN', dropout: float = 0.1):
        """
        Initialize the GNN model.
        
        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of convolutional layers
            conv_type: Type of convolution ('GCN' or 'SAGE')
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        if conv_type not in ['GCN', 'SAGE']:
            raise ValueError("conv_type must be either 'GCN' or 'SAGE'")
        
        self.num_layers = num_layers
        self.conv_type = conv_type
        self.dropout = dropout
        self.convs = torch.nn.ModuleList()
        
        # Select convolution layer type
        ConvLayer = GCNConv if conv_type == 'GCN' else SAGEConv
        
        # First layer
        self.convs.append(ConvLayer(input_dim, hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(ConvLayer(hidden_dim, hidden_dim))
            
        # Last layer
        if num_layers > 1:
            self.convs.append(ConvLayer(hidden_dim, hidden_dim))
        
        # Output layer for complexity prediction
        self.predictor = torch.nn.Linear(hidden_dim, 1)
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing graph
            
        Returns:
            Complexity prediction tensor of shape (batch_size, 1)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply convolution layers with ReLU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x, batch)
        
        # Predict complexity
        return self.predictor(x)
    
    def get_model_info(self) -> str:
        """
        Get information about the model configuration.
        
        Returns:
            String describing the model architecture
        """
        return (f"RubyComplexityGNN({self.conv_type}, "
                f"layers={self.num_layers}, "
                f"dropout={self.dropout})")