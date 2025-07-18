"""
Data processing utilities for Ruby method datasets.

This module provides functions to load, preprocess, and prepare Ruby method
data for GNN training. Includes custom Dataset class for AST to graph conversion.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union


def load_methods_json(filepath: str) -> List[Dict[str, Any]]:
    """
    Load Ruby methods from JSON file.
    
    Args:
        filepath: Path to the JSON file containing method data
        
    Returns:
        List of method dictionaries
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def methods_to_dataframe(methods: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert list of method dictionaries to a structured format.
    
    Args:
        methods: List of method dictionaries
        
    Returns:
        List of method dictionaries (pass-through for compatibility)
    """
    return methods


def filter_methods_by_length(methods: List[Dict[str, Any]], min_lines: int = 5, max_lines: int = 100) -> List[Dict[str, Any]]:
    """
    Filter methods by source code length.
    
    Args:
        methods: List of method dictionaries
        min_lines: Minimum number of lines
        max_lines: Maximum number of lines
        
    Returns:
        Filtered list of methods
    """
    filtered = []
    for method in methods:
        if 'raw_source' in method:
            line_count = len(method['raw_source'].split('\n'))
            if min_lines <= line_count <= max_lines:
                method['line_count'] = line_count
                filtered.append(method)
    return filtered
    """
    Filter methods by source code length.
    
    Args:
        df: DataFrame containing method data
        min_lines: Minimum number of lines
        max_lines: Maximum number of lines
        
    Returns:
        Filtered DataFrame
    """
    df['line_count'] = df['raw_source'].apply(lambda x: len(x.split('\n')))
    return df[(df['line_count'] >= min_lines) & (df['line_count'] <= max_lines)]


class ASTNodeEncoder:
    """
    Encoder for mapping AST node types to feature vectors.
    
    This class maintains a vocabulary of AST node types found in Ruby code
    and maps them to dense feature vectors for GNN processing.
    """
    
    def __init__(self):
        """Initialize the node encoder with common Ruby AST node types."""
        # Common Ruby AST node types based on the parser gem
        self.node_types = [
            'def', 'defs', 'args', 'arg', 'begin', 'end', 'lvasgn', 'ivasgn', 'gvasgn',
            'cvasgn', 'send', 'block', 'if', 'unless', 'while', 'until', 'for', 'case',
            'when', 'rescue', 'ensure', 'retry', 'break', 'next', 'redo', 'return',
            'yield', 'super', 'zsuper', 'lambda', 'proc', 'and', 'or', 'not', 'true',
            'false', 'nil', 'self', 'int', 'float', 'str', 'sym', 'regexp', 'array',
            'hash', 'pair', 'splat', 'kwsplat', 'block_pass', 'const', 'cbase',
            'lvar', 'ivar', 'gvar', 'cvar', 'casgn', 'masgn', 'mlhs', 'op_asgn',
            'and_asgn', 'or_asgn', 'back_ref', 'nth_ref', 'class', 'sclass', 'module',
            'defined?', 'alias', 'undef', 'range', 'irange', 'erange', 'regopt'
        ]
        
        # Create mapping from node type to index
        self.type_to_idx = {node_type: idx for idx, node_type in enumerate(self.node_types)}
        self.unknown_idx = len(self.node_types)  # Index for unknown node types
        self.vocab_size = len(self.node_types) + 1  # +1 for unknown
        
    def encode_node_type(self, node_type: str) -> int:
        """
        Encode a node type to its integer index.
        
        Args:
            node_type: The AST node type string
            
        Returns:
            Integer index for the node type
        """
        return self.type_to_idx.get(node_type, self.unknown_idx)
    
    def create_node_features(self, node_type: str) -> List[float]:
        """
        Create feature vector for a node type.
        
        Args:
            node_type: The AST node type string
            
        Returns:
            Feature vector as list of floats
        """
        # Simple one-hot encoding for now
        features = [0.0] * self.vocab_size
        idx = self.encode_node_type(node_type)
        features[idx] = 1.0
        return features


class ASTGraphConverter:
    """
    Converter for transforming AST JSON to graph representation.
    
    This class parses the AST JSON structure and converts it into
    a graph format suitable for GNN processing.
    """
    
    def __init__(self):
        """Initialize the AST to graph converter."""
        self.node_encoder = ASTNodeEncoder()
        self.reset()
    
    def reset(self):
        """Reset the converter state for processing a new AST."""
        self.nodes = []  # List of node features
        self.edges = []  # List of edge tuples (parent_idx, child_idx)
        self.node_count = 0
    
    def parse_ast_json(self, ast_json: str) -> Dict[str, Any]:
        """
        Parse AST JSON string and convert to graph representation.
        
        Args:
            ast_json: JSON string representing the AST
            
        Returns:
            Dictionary containing node features and edge indices
        """
        self.reset()
        
        try:
            ast_data = json.loads(ast_json)
            self._process_node(ast_data, parent_idx=None)
            
            # Convert to appropriate format
            if not self.nodes:
                # Handle empty AST case
                node_features = [[0.0] * self.node_encoder.vocab_size]
                edge_index = [[], []]  # Empty edge list
            else:
                node_features = self.nodes
                if self.edges:
                    # Transpose edge list to [2, num_edges] format
                    edge_index = [[], []]
                    for parent, child in self.edges:
                        edge_index[0].append(parent)
                        edge_index[1].append(child)
                else:
                    edge_index = [[], []]
            
            return {
                'x': node_features,
                'edge_index': edge_index,
                'num_nodes': len(self.nodes) if self.nodes else 1
            }
            
        except (json.JSONDecodeError, Exception) as e:
            # Handle malformed JSON or other errors gracefully
            return {
                'x': [[0.0] * self.node_encoder.vocab_size],
                'edge_index': [[], []],
                'num_nodes': 1
            }
    
    def _process_node(self, node: Union[Dict, List, str, int, float, None], parent_idx: Optional[int] = None) -> int:
        """
        Recursively process an AST node and its children.
        
        Args:
            node: The AST node (dict, list, or primitive)
            parent_idx: Index of the parent node
            
        Returns:
            Index of the current node
        """
        if isinstance(node, dict) and 'type' in node:
            # This is an AST node with a type
            node_type = node['type']
            current_idx = self.node_count
            self.node_count += 1
            
            # Create node features
            features = self.node_encoder.create_node_features(node_type)
            self.nodes.append(features)
            
            # Add edge from parent to current node
            if parent_idx is not None:
                self.edges.append((parent_idx, current_idx))
            
            # Process children
            if 'children' in node:
                for child in node['children']:
                    self._process_node(child, current_idx)
            
            return current_idx
            
        elif isinstance(node, list):
            # Process list of nodes
            for child in node:
                self._process_node(child, parent_idx)
            return parent_idx if parent_idx is not None else -1
            
        else:
            # Leaf node (string, int, float, None)
            if parent_idx is not None:
                current_idx = self.node_count
                self.node_count += 1
                
                # Create a generic leaf node
                leaf_type = 'leaf_' + type(node).__name__
                features = self.node_encoder.create_node_features(leaf_type)
                self.nodes.append(features)
                
                # Add edge from parent to leaf
                self.edges.append((parent_idx, current_idx))
                
                return current_idx
            return -1


def load_jsonl_file(filepath: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        
    Returns:
        List of dictionaries from the JSONL file
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines
    return data


class RubyASTDataset:
    """
    Dataset class for loading Ruby AST data and converting to graph format.
    
    This class loads JSONL files containing Ruby method data and converts
    the AST representations to graph objects suitable for GNN training.
    """
    
    def __init__(self, jsonl_path: str, transform=None):
        """
        Initialize the dataset.
        
        Args:
            jsonl_path: Path to the JSONL file containing method data
            transform: Optional transform to apply to each sample
        """
        self.jsonl_path = jsonl_path
        self.transform = transform
        self.converter = ASTGraphConverter()
        
        # Load the data
        self.data = load_jsonl_file(jsonl_path)
        
        print(f"Loaded {len(self.data)} samples from {jsonl_path}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing graph data and target
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}")
        
        sample = self.data[idx]
        
        # Convert AST to graph
        graph_data = self.converter.parse_ast_json(sample['ast_json'])
        
        # Create the data object
        result = {
            'x': graph_data['x'],
            'edge_index': graph_data['edge_index'],
            'y': [sample['complexity_score']],
            'num_nodes': graph_data['num_nodes'],
            'id': sample.get('id', f'sample_{idx}'),
            'repo_name': sample.get('repo_name', ''),
            'file_path': sample.get('file_path', '')
        }
        
        # Apply transform if provided
        if self.transform:
            result = self.transform(result)
        
        return result
    
    def get_feature_dim(self) -> int:
        """Return the dimension of node features."""
        return self.converter.node_encoder.vocab_size


def collate_graphs(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for batching graph data.
    
    Args:
        batch: List of graph data dictionaries
        
    Returns:
        Batched graph data
    """
    if not batch:
        raise ValueError("Cannot collate empty batch")
    
    # Collect all node features and edge indices
    all_x = []
    all_edge_index = [[], []]  # [source_nodes, target_nodes]
    all_y = []
    batch_idx = []
    node_offset = 0
    
    metadata = {
        'ids': [],
        'repo_names': [],
        'file_paths': []
    }
    
    for i, sample in enumerate(batch):
        # Node features
        all_x.extend(sample['x'])
        
        # Edge indices (offset by current node count)
        edges = sample['edge_index']
        if len(edges[0]) > 0:  # Only offset if there are edges
            for j in range(len(edges[0])):
                all_edge_index[0].append(edges[0][j] + node_offset)
                all_edge_index[1].append(edges[1][j] + node_offset)
        
        # Target values
        all_y.extend(sample['y'])
        
        # Batch indices for each node
        num_nodes = sample['num_nodes']
        batch_idx.extend([i] * num_nodes)
        node_offset += num_nodes
        
        # Metadata
        metadata['ids'].append(sample['id'])
        metadata['repo_names'].append(sample['repo_name'])
        metadata['file_paths'].append(sample['file_path'])
    
    return {
        'x': all_x,
        'edge_index': all_edge_index,
        'y': all_y,
        'batch': batch_idx,
        'num_graphs': len(batch),
        'metadata': metadata
    }


class SimpleDataLoader:
    """
    Simple DataLoader implementation for batching data.
    
    This provides a basic implementation that can be used when PyTorch
    DataLoader is not available, and can easily be replaced with the real
    PyTorch DataLoader when dependencies are installed.
    """
    
    def __init__(self, dataset, batch_size: int = 1, shuffle: bool = False, collate_fn=None):
        """
        Initialize the DataLoader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            collate_fn: Function to collate samples into batches
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or collate_graphs
        
        # Create indices
        self.indices = list(range(len(dataset)))
        if shuffle:
            import random
            random.shuffle(self.indices)
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def __iter__(self):
        """Iterate over batches."""
        for i in range(0, len(self.dataset), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)


def create_data_loaders(train_path: str, val_path: str, batch_size: int = 32, shuffle: bool = True):
    """
    Create train and validation data loaders.
    
    Args:
        train_path: Path to training JSONL file
        val_path: Path to validation JSONL file
        batch_size: Batch size for both loaders
        shuffle: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = RubyASTDataset(train_path)
    val_dataset = RubyASTDataset(val_path)
    
    train_loader = SimpleDataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = SimpleDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader