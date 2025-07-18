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
        
    def forward(self, data: Data, return_embedding: bool = False) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: PyTorch Geometric Data object containing graph
            return_embedding: If True, return graph embedding instead of prediction
            
        Returns:
            Complexity prediction tensor of shape (batch_size, 1) or
            Graph embedding tensor of shape (batch_size, hidden_dim) if return_embedding=True
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply convolution layers with ReLU activation and dropout
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:  # No activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        embedding = global_mean_pool(x, batch)
        
        if return_embedding:
            return embedding
        
        # Predict complexity
        return self.predictor(embedding)
    
    def get_model_info(self) -> str:
        """
        Get information about the model configuration.
        
        Returns:
            String describing the model architecture
        """
        return (f"RubyComplexityGNN({self.conv_type}, "
                f"layers={self.num_layers}, "
                f"dropout={self.dropout})")


class ASTDecoder(torch.nn.Module):
    """
    GNN-based decoder for reconstructing Abstract Syntax Trees from embeddings.
    
    This module takes a graph embedding and autoregressively generates node features
    and edge structure to reconstruct an AST.
    """
    
    def __init__(self, embedding_dim: int, output_node_dim: int, hidden_dim: int = 64, 
                 num_layers: int = 3, max_nodes: int = 100):
        """
        Initialize the AST decoder.
        
        Args:
            embedding_dim: Dimension of input graph embedding
            output_node_dim: Dimension of output node features
            hidden_dim: Hidden layer dimension
            num_layers: Number of decoder layers
            max_nodes: Maximum number of nodes to generate
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.output_node_dim = output_node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_nodes = max_nodes
        
        # Transform embedding to initial hidden state
        self.embedding_transform = torch.nn.Linear(embedding_dim, hidden_dim)
        
        # GNN layers for iterative refinement
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Output projections
        self.node_output = torch.nn.Linear(hidden_dim, output_node_dim)
        self.edge_predictor = torch.nn.Linear(hidden_dim * 2, 1)  # For edge existence
        
    def forward(self, embedding: torch.Tensor, target_num_nodes: int = None) -> dict:
        """
        Forward pass to decode embedding into AST structure.
        
        Args:
            embedding: Graph embedding tensor of shape (batch_size, embedding_dim)
            target_num_nodes: Target number of nodes to generate (for training)
            
        Returns:
            Dictionary containing generated node features and edge information
        """
        batch_size = embedding.size(0)
        
        # Use target_num_nodes if provided, otherwise use default max_nodes
        num_nodes = target_num_nodes if target_num_nodes is not None else self.max_nodes
        
        # Initialize node features from embedding
        # Broadcast embedding to all nodes
        initial_features = self.embedding_transform(embedding)  # (batch_size, hidden_dim)
        node_features = initial_features.unsqueeze(1).expand(batch_size, num_nodes, self.hidden_dim)
        
        # Create a simple fully connected graph for initial processing
        # This is a simplified approach - in practice, you'd want more sophisticated edge generation
        edge_list = []
        batch_indices = []
        
        for b in range(batch_size):
            # Create edges for this batch
            for i in range(num_nodes):
                for j in range(i + 1, min(i + 3, num_nodes)):  # Connect to next 2 nodes
                    edge_list.extend([[b * num_nodes + i, b * num_nodes + j],
                                    [b * num_nodes + j, b * num_nodes + i]])
            batch_indices.extend([b] * num_nodes)
        
        # Determine device from embedding to ensure tensor consistency
        device = embedding.device
        
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
        else:
            # No edges case
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
        
        batch_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)
        
        # Flatten node features for GNN processing
        x = node_features.reshape(-1, self.hidden_dim)  # (batch_size * num_nodes, hidden_dim)
        
        # Apply GNN layers for refinement
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Generate final node features
        output_node_features = self.node_output(x)  # (batch_size * num_nodes, output_node_dim)
        
        # Reshape back to batch format
        output_node_features = output_node_features.reshape(batch_size, num_nodes, self.output_node_dim)
        
        return {
            'node_features': output_node_features,
            'edge_index': edge_index,
            'batch': batch_tensor,
            'num_nodes_per_graph': [num_nodes] * batch_size
        }


class ASTAutoencoder(torch.nn.Module):
    """
    Autoencoder for Abstract Syntax Trees using Graph Neural Networks.
    
    Combines the existing RubyComplexityGNN (as encoder) with the new ASTDecoder
    to create an autoencoder that can reconstruct ASTs from learned embeddings.
    """
    
    def __init__(self, encoder_input_dim: int, node_output_dim: int, 
                 hidden_dim: int = 64, num_layers: int = 3, 
                 conv_type: str = 'GCN', dropout: float = 0.1,
                 freeze_encoder: bool = False, encoder_weights_path: str = None):
        """
        Initialize the AST autoencoder.
        
        Args:
            encoder_input_dim: Input dimension for encoder (node feature dimension)
            node_output_dim: Output dimension for decoder node features
            hidden_dim: Hidden dimension for both encoder and decoder
            num_layers: Number of layers in both encoder and decoder
            conv_type: Type of convolution for encoder ('GCN' or 'SAGE')
            dropout: Dropout rate for encoder
            freeze_encoder: Whether to freeze encoder weights
            encoder_weights_path: Path to pre-trained encoder weights
        """
        super().__init__()
        
        # Initialize encoder (RubyComplexityGNN without prediction head)
        self.encoder = RubyComplexityGNN(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout
        )
        
        # Load pre-trained weights if provided and adjust encoder config if needed
        self.encoder_weights_path = encoder_weights_path
        if encoder_weights_path is not None:
            try:
                checkpoint = torch.load(encoder_weights_path, map_location='cpu')
                # Check if checkpoint contains model config and use it to create compatible encoder
                if 'model_config' in checkpoint:
                    saved_config = checkpoint['model_config']
                    # Recreate encoder with saved configuration if it differs from current
                    if (saved_config.get('conv_type', conv_type) != conv_type or
                        saved_config.get('hidden_dim', hidden_dim) != hidden_dim or
                        saved_config.get('num_layers', num_layers) != num_layers or
                        saved_config.get('dropout', dropout) != dropout):
                        print(f"Adjusting encoder config to match saved model: conv_type={saved_config.get('conv_type', conv_type)}")
                        self.encoder = RubyComplexityGNN(
                            input_dim=encoder_input_dim,
                            hidden_dim=saved_config.get('hidden_dim', hidden_dim),
                            num_layers=saved_config.get('num_layers', num_layers),
                            conv_type=saved_config.get('conv_type', conv_type),
                            dropout=saved_config.get('dropout', dropout)
                        )
                        # Update hidden_dim for decoder compatibility
                        hidden_dim = saved_config.get('hidden_dim', hidden_dim)
                
                self.encoder.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded encoder weights from {encoder_weights_path}")
            except FileNotFoundError:
                print(f"Warning: Could not find encoder weights at {encoder_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load encoder weights: {e}")
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            print("Encoder weights frozen")
        
        # Initialize decoder
        self.decoder = ASTDecoder(
            embedding_dim=hidden_dim,
            output_node_dim=node_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        
        self.hidden_dim = hidden_dim
        self.freeze_encoder = freeze_encoder
        
    def forward(self, data: Data) -> dict:
        """
        Forward pass through the autoencoder.
        
        Args:
            data: PyTorch Geometric Data object containing input AST
            
        Returns:
            Dictionary containing reconstructed AST information
        """
        # Encode: AST -> embedding
        embedding = self.encoder(data, return_embedding=True)
        
        # Determine target number of nodes from input data
        batch_size = embedding.size(0)
        if hasattr(data, 'batch'):
            # Count nodes per graph in batch
            unique_batch, counts = torch.unique(data.batch, return_counts=True)
            target_nodes = int(counts[0].item())  # Use first graph's size as target
        else:
            # Single graph case
            target_nodes = data.x.size(0)
        
        # Decode: embedding -> AST
        reconstruction = self.decoder(embedding, target_num_nodes=target_nodes)
        
        return {
            'embedding': embedding,
            'reconstruction': reconstruction
        }
    
    def get_model_info(self) -> str:
        """
        Get information about the autoencoder configuration.
        
        Returns:
            String describing the model architecture
        """
        encoder_info = self.encoder.get_model_info()
        decoder_info = f"ASTDecoder(embedding_dim={self.hidden_dim})"
        freeze_status = " [FROZEN]" if self.freeze_encoder else ""
        
        return (f"ASTAutoencoder(\n"
                f"  encoder: {encoder_info}{freeze_status}\n"
                f"  decoder: {decoder_info}\n"
                f")")