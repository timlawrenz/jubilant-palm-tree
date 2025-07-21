"""
Graph Neural Network models for Ruby code complexity prediction.

This module contains PyTorch Geometric models for learning from
Ruby AST structures.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data, Batch
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False


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


class SimpleTextEncoder(torch.nn.Module):
    """
    Simple text encoder as fallback when sentence-transformers is not available.
    
    This provides a basic text encoding mechanism using character-level features
    and a simple neural network. Used as fallback for testing when internet
    access is not available.
    """
    
    def __init__(self, output_dim: int = 384, max_length: int = 100):
        """
        Initialize the simple text encoder.
        
        Args:
            output_dim: Output embedding dimension
            max_length: Maximum text length to consider
        """
        super().__init__()
        self.output_dim = output_dim
        self.max_length = max_length
        
        # Character embedding (256 ASCII characters)
        self.char_embedding = torch.nn.Embedding(256, 64)
        
        # Simple RNN for text processing
        self.rnn = torch.nn.LSTM(64, 128, batch_first=True, bidirectional=True)
        
        # Output projection
        self.output_proj = torch.nn.Linear(256, output_dim)
        
    def encode(self, texts: list, convert_to_tensor: bool = True) -> torch.Tensor:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of text strings
            convert_to_tensor: Whether to return tensor (for compatibility)
            
        Returns:
            Text embeddings tensor
        """
        batch_size = len(texts)
        
        # Convert texts to character indices
        char_sequences = []
        for text in texts:
            # Convert to lowercase and get character codes
            chars = [min(ord(c), 255) for c in text.lower()[:self.max_length]]
            # Pad to max_length
            chars.extend([0] * (self.max_length - len(chars)))
            char_sequences.append(chars[:self.max_length])
        
        # Convert to tensor and move to same device as model
        char_tensor = torch.tensor(char_sequences, dtype=torch.long)
        char_tensor = char_tensor.to(next(self.parameters()).device)
        
        # Embed characters
        embedded = self.char_embedding(char_tensor)  # (batch, seq_len, embed_dim)
        
        # Process with RNN
        rnn_output, (hidden, _) = self.rnn(embedded)
        
        # Use last hidden state (concatenated forward and backward)
        final_hidden = torch.cat([hidden[0], hidden[1]], dim=1)  # (batch, 256)
        
        # Project to output dimension
        embeddings = self.output_proj(final_hidden)
        
        return embeddings
    
    def get_sentence_embedding_dimension(self) -> int:
        """Get embedding dimension for compatibility."""
        return self.output_dim


class AlignmentModel(torch.nn.Module):
    """
    Dual-encoder model for aligning text descriptions with code embeddings.
    
    This model combines a frozen RubyComplexityGNN (code encoder) with a 
    sentence-transformers text encoder to create aligned embeddings in the
    same 64-dimensional space.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3,
                 conv_type: str = 'GCN', dropout: float = 0.1,
                 text_model_name: str = 'all-MiniLM-L6-v2',
                 code_encoder_weights_path: str = 'best_encoder_model.pt'):
        """
        Initialize the alignment model.
        
        Args:
            input_dim: Input dimension for code encoder (node feature dimension)
            hidden_dim: Hidden dimension for both encoders (default: 64)
            num_layers: Number of layers in code encoder
            conv_type: Type of convolution for code encoder ('GCN' or 'SAGE')
            dropout: Dropout rate for code encoder
            text_model_name: Name of the sentence-transformers model to use
            code_encoder_weights_path: Path to pre-trained code encoder weights (default: 'best_encoder_model.pt')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initialize frozen code encoder (RubyComplexityGNN without prediction head)
        self.code_encoder = RubyComplexityGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            dropout=dropout
        )
        
        # Load pre-trained weights if provided
        if code_encoder_weights_path is not None:
            try:
                checkpoint = torch.load(code_encoder_weights_path, map_location='cpu')
                # Handle both direct state dict and checkpoint format
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                # Load state dict, ignoring predictor weights if present
                model_state = {}
                for key, value in state_dict.items():
                    if not key.startswith('predictor'):
                        model_state[key] = value
                
                self.code_encoder.load_state_dict(model_state, strict=False)
                print(f"Loaded code encoder weights from {code_encoder_weights_path}")
            except FileNotFoundError:
                print(f"Warning: Could not find code encoder weights at {code_encoder_weights_path}")
            except Exception as e:
                print(f"Warning: Could not load code encoder weights: {e}")
        
        # Freeze code encoder parameters
        for param in self.code_encoder.parameters():
            param.requires_grad = False
        print("Code encoder weights frozen")
        
        # Initialize text encoder
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.text_encoder = SentenceTransformer(text_model_name)
                self.text_encoder_type = "sentence_transformers"
                print(f"Using SentenceTransformer: {text_model_name}")
            except Exception as e:
                print(f"Warning: Could not load SentenceTransformer ({e}), using fallback")
                self.text_encoder = SimpleTextEncoder(output_dim=384)
                self.text_encoder_type = "simple"
        else:
            print("SentenceTransformers not available, using simple text encoder")
            self.text_encoder = SimpleTextEncoder(output_dim=384)
            self.text_encoder_type = "simple"
        
        # Get text encoder output dimension
        text_dim = self.text_encoder.get_sentence_embedding_dimension()
        
        # Projection head to align text embeddings to code embedding space
        # Small MLP for better capacity: Linear(384 -> 256) -> ReLU() -> Linear(256 -> 64)
        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(text_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, hidden_dim)
        )
        
        print(f"Text encoder output dim: {text_dim}, projecting to: {hidden_dim}")
        
    def encode_code(self, data: Data) -> torch.Tensor:
        """
        Encode graph data to embeddings using the frozen code encoder.
        
        Args:
            data: PyTorch Geometric Data object containing graph
            
        Returns:
            Code embeddings tensor of shape (batch_size, hidden_dim)
        """
        with torch.no_grad():  # Code encoder is frozen
            return self.code_encoder(data, return_embedding=True)
    
    def encode_text(self, texts: list) -> torch.Tensor:
        """
        Encode text descriptions to embeddings using the text encoder.
        
        Args:
            texts: List of text descriptions
            
        Returns:
            Text embeddings tensor of shape (batch_size, hidden_dim)
        """
        # Get text embeddings from sentence transformer
        text_embeddings = self.text_encoder.encode(texts, convert_to_tensor=True)
        
        # Clone tensor to create a normal tensor for autograd (SentenceTransformer creates inference tensors)
        text_embeddings = text_embeddings.clone()
        
        # Project to code embedding space
        projected_embeddings = self.text_projection(text_embeddings)
        
        return projected_embeddings
    
    def forward(self, data: Data, texts: list) -> dict:
        """
        Forward pass through both encoders.
        
        Args:
            data: PyTorch Geometric Data object containing graphs
            texts: List of text descriptions (same length as batch size)
            
        Returns:
            Dictionary containing:
                - 'code_embeddings': Code embeddings (batch_size, hidden_dim)
                - 'text_embeddings': Text embeddings (batch_size, hidden_dim)
        """
        # Encode code
        code_embeddings = self.encode_code(data)
        
        # Encode text
        text_embeddings = self.encode_text(texts)
        
        # Ensure embeddings are on the same device
        if code_embeddings.device != text_embeddings.device:
            text_embeddings = text_embeddings.to(code_embeddings.device)
        
        return {
            'code_embeddings': code_embeddings,
            'text_embeddings': text_embeddings
        }
    
    def get_model_info(self) -> str:
        """
        Get information about the alignment model configuration.
        
        Returns:
            String describing the model architecture
        """
        code_info = self.code_encoder.get_model_info()
        
        if self.text_encoder_type == "sentence_transformers":
            # Try to get model name from _model_config, fallback to transformer config, or use generic name
            model_name = self.text_encoder._model_config.get('_name_or_path')
            if model_name is None:
                # Try to get from transformer module config
                try:
                    model_name = self.text_encoder[0].auto_model.config._name_or_path
                except (AttributeError, IndexError):
                    model_name = "SentenceTransformer"
            text_info = f"SentenceTransformer({model_name})"
        else:
            text_info = f"SimpleTextEncoder(dim={self.text_encoder.output_dim})"
            
        # Handle Sequential projection (MLP) vs single Linear layer
        if isinstance(self.text_projection, torch.nn.Sequential):
            first_layer = self.text_projection[0]
            last_layer = self.text_projection[2]
            projection_info = f"MLP({first_layer.in_features} -> 256 -> {last_layer.out_features})"
        else:
            projection_info = f"Linear({self.text_projection.in_features} -> {self.text_projection.out_features})"
        
        return (f"AlignmentModel(\n"
                f"  code_encoder: {code_info} [FROZEN]\n"
                f"  text_encoder: {text_info}\n"
                f"  projection: {projection_info}\n"
                f")")