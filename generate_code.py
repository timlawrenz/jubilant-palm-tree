#!/usr/bin/env python3
"""
Code Generation Script

This script provides a simple command-line interface for the text-to-code model.
It loads the trained AlignmentModel and ASTDecoder, takes natural language input,
and generates Ruby code.

Usage:
    python generate_code.py "calculate total price with tax"
    python generate_code.py --interactive
"""

import sys
import os
import argparse
import json
import subprocess
import tempfile
import torch
import torch.nn.functional as F

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import AlignmentModel, ASTDecoder, AutoregressiveASTDecoder


def create_empty_graph():
    """Create an empty graph for starting autoregressive generation."""
    return {
        'x': [],  # Node features
        'edge_index': [[], []],  # Edge connections
        'batch': [],  # Batch indices
        'num_nodes': 0
    }


def add_node_to_graph(partial_graph, new_node):
    """
    Add a new node to the partial graph.
    
    Args:
        partial_graph: Current partial graph state
        new_node: Dictionary with 'node_type', 'connections', 'features'
        
    Returns:
        Updated partial graph with new node added
    """
    # Create a copy of the current graph
    updated_graph = {
        'x': partial_graph['x'].copy() if isinstance(partial_graph['x'], list) else partial_graph['x'],
        'edge_index': [partial_graph['edge_index'][0].copy(), partial_graph['edge_index'][1].copy()],
        'batch': partial_graph['batch'].copy() if isinstance(partial_graph['batch'], list) else partial_graph['batch'],
        'num_nodes': partial_graph['num_nodes']
    }
    
    # Add new node features
    if isinstance(updated_graph['x'], list):
        updated_graph['x'].append(new_node['features'])
    else:
        # Convert to list if it's a tensor
        updated_graph['x'] = updated_graph['x'].tolist()
        updated_graph['x'].append(new_node['features'])
    
    # Add connections from new node to existing nodes
    new_node_idx = updated_graph['num_nodes']
    for connection_idx in new_node.get('connections', []):
        if connection_idx < new_node_idx:  # Only connect to existing nodes
            # Add bidirectional edges
            updated_graph['edge_index'][0].extend([new_node_idx, connection_idx])
            updated_graph['edge_index'][1].extend([connection_idx, new_node_idx])
    
    # Update batch index for new node (assume single batch for generation)
    if isinstance(updated_graph['batch'], list):
        updated_graph['batch'].append(0)
    else:
        updated_graph['batch'] = updated_graph['batch'].tolist()
        updated_graph['batch'].append(0)
    
    # Update node count
    updated_graph['num_nodes'] += 1
    
    return updated_graph


def get_node_features(node_type_idx, feature_dim=74):
    """
    Get one-hot encoded features for a node type.
    
    Args:
        node_type_idx: Index of the node type
        feature_dim: Total feature dimension
        
    Returns:
        List of feature values (one-hot encoded)
    """
    features = [0.0] * feature_dim
    if 0 <= node_type_idx < feature_dim:
        features[node_type_idx] = 1.0
    return features


def is_end_token(node_type_idx, end_token_idx=0):
    """
    Check if the node type represents end of generation.
    
    Args:
        node_type_idx: Index of the predicted node type
        end_token_idx: Index that represents end of sequence
        
    Returns:
        True if generation should stop
    """
    # For simplicity, we'll use index 0 as end token
    # In practice, this might be a specific "END" token or when we reach certain node types
    return node_type_idx == end_token_idx


def build_complete_ast(generated_nodes):
    """
    Build complete AST structure from generated nodes.
    
    Args:
        generated_nodes: List of generated node dictionaries
        
    Returns:
        Dictionary representing the complete AST for further processing
    """
    if not generated_nodes:
        # Return minimal AST structure
        return {
            'node_features': torch.zeros(1, 1, 74),  # Batch of 1, 1 node, 74 features
            'edge_index': torch.zeros(2, 0, dtype=torch.long),  # No edges
            'num_nodes': 1
        }
    
    # Convert nodes to tensor format expected by existing pipeline
    node_features = []
    edge_index = [[], []]
    
    for i, node in enumerate(generated_nodes):
        node_features.append(node['features'])
        
        # Add connections
        for connection_idx in node.get('connections', []):
            if connection_idx < i:  # Only connect to previous nodes
                edge_index[0].extend([i, connection_idx])
                edge_index[1].extend([connection_idx, i])
    
    # Convert to tensors
    node_features_tensor = torch.tensor(node_features, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) if edge_index[0] else torch.zeros(2, 0, dtype=torch.long)
    
    return {
        'node_features': node_features_tensor,
        'edge_index': edge_index_tensor,
        'num_nodes': len(generated_nodes),
        'generated_nodes': generated_nodes  # Preserve original nodes with connection info
    }


def generate_ast_autoregressive(model, text_embedding, max_length=50, 
                              temperature=1.0, top_k=5):
    """
    Generate AST using autoregressive decoder with sampling
    
    Args:
        model: Trained AutoregressiveASTDecoder
        text_embedding: (1, 64) - Text description embedding
        max_length: Maximum number of nodes to generate
        temperature: Sampling temperature for diversity
        top_k: Top-k sampling for node type selection
    """
    
    model.eval()
    device = next(model.parameters()).device
    text_embedding = text_embedding.to(device)
    
    # Initialize generation
    partial_graph = create_empty_graph()
    hidden_state = None
    generated_nodes = []
    
    with torch.no_grad():
        for step in range(max_length):
            # Forward pass
            outputs = model(text_embedding, partial_graph, hidden_state)
            
            # Sample next node type with temperature and top-k
            node_type_logits = outputs['node_type_logits'] / temperature
            
            # Top-k sampling for diversity
            if top_k > 0 and top_k < node_type_logits.size(-1):
                top_k_indices = torch.topk(node_type_logits, top_k).indices
                top_k_probs = F.softmax(
                    torch.gather(node_type_logits, 1, top_k_indices), 
                    dim=1
                )
                
                sampled_idx = torch.multinomial(top_k_probs, 1)
                next_node_type = top_k_indices.gather(1, sampled_idx).item()
            else:
                # Regular sampling from full distribution
                node_type_probs = F.softmax(node_type_logits, dim=1)
                next_node_type = torch.multinomial(node_type_probs, 1).item()
            
            # Sample connections to existing nodes
            connection_probs = outputs['connection_probs']
            connections = (connection_probs > 0.5).nonzero(as_tuple=True)[1]
            
            # Create new node
            new_node = {
                'node_type': next_node_type,
                'connections': connections.tolist()[:partial_graph['num_nodes']],  # Only connect to existing nodes
                'features': get_node_features(next_node_type)
            }
            
            # Check for end-of-generation
            if is_end_token(next_node_type) or len(generated_nodes) >= max_length - 1:
                break
                
            # Update partial graph
            partial_graph = add_node_to_graph(partial_graph, new_node)
            generated_nodes.append(new_node)
            
            # Update hidden state
            hidden_state = outputs['hidden_state']
    
    return build_complete_ast(generated_nodes)


class CodeGenerator:
    """Main code generation class."""
    
    def __init__(self, alignment_model_path="models/best_alignment_model.pt", 
                 decoder_model_path="models/best_decoder.pt",
                 code_encoder_path="models/best_model.pt"):
        """
        Initialize the code generator.
        
        Args:
            alignment_model_path: Path to trained AlignmentModel
            decoder_model_path: Path to trained ASTDecoder  
            code_encoder_path: Path to trained code encoder weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🚀 Loading Code Generation Models...")
        print("=" * 50)
        
        # Load AlignmentModel
        print("📦 Loading AlignmentModel...")
        self.alignment_model = self._load_alignment_model(
            alignment_model_path, code_encoder_path
        )
        print("✅ AlignmentModel loaded successfully")
        
        # Load ASTDecoder
        print("📦 Loading ASTDecoder...")
        self.decoder = self._load_ast_decoder(decoder_model_path)
        print("✅ ASTDecoder loaded successfully")
        
        print("\n🎯 Code Generator ready!")
        
    def _load_alignment_model(self, model_path, code_encoder_path):
        """Load the AlignmentModel with proper configuration."""
        # Create model with same configuration as training
        alignment_model = AlignmentModel(
            input_dim=74,  # Based on dataset
            hidden_dim=64,
            text_model_name='all-MiniLM-L6-v2',  # Use same model as training
            code_encoder_weights_path=code_encoder_path
        )
        
        # Load trained weights if available
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            alignment_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded alignment model weights from {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Could not find alignment model at {model_path}")
            print("   Using randomly initialized weights (for testing only)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load alignment model weights: {e}")
            print("   Using randomly initialized weights (for testing only)")
            
        alignment_model.eval()
        
        return alignment_model.to(self.device)
    
    def _load_ast_decoder(self, model_path):
        """Load the ASTDecoder with proper configuration."""
        decoder = ASTDecoder(
            embedding_dim=64,
            output_node_dim=74,
            hidden_dim=64
        )
        
        # Load trained weights if available
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            print(f"✅ Loaded AST decoder weights from {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Could not find AST decoder at {model_path}")
            print("   Using randomly initialized weights (for testing only)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load AST decoder weights: {e}")
            print("   Using randomly initialized weights (for testing only)")
            
        decoder.eval()
        
        return decoder.to(self.device)
    
    def text_to_embedding(self, text):
        """Convert natural language text to 64-dimensional embedding."""
        with torch.no_grad():
            embedding = self.alignment_model.encode_text([text])
        return embedding
    
    def embedding_to_ast(self, embedding, target_nodes=15):
        """Convert embedding to AST structure."""
        with torch.no_grad():
            reconstruction = self.decoder(embedding, target_num_nodes=target_nodes)
        return reconstruction
    
    def ast_to_json(self, reconstruction, method_name="generated_method"):
        """
        Convert decoder output to AST JSON format expected by Ruby pretty printer.
        
        This implementation uses the proper tree-building algorithm that takes the sequence
        of generated nodes and their predicted parent-child connections and assembles them
        into the correct nested JSON structure.
        """
        # Import the node type mapping
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from data_processing import ASTNodeEncoder
        
        # Initialize the encoder to get node type names
        encoder = ASTNodeEncoder()
        
        # Check if we have the modern reconstruction format with generated nodes
        if 'generated_nodes' in reconstruction:
            generated_nodes = reconstruction['generated_nodes']
        else:
            # Fallback: Try to extract from node_features (though this won't have connection info)
            # This provides backward compatibility with older reconstruction formats
            node_features = reconstruction['node_features'][0]  # First batch item
            num_nodes = node_features.shape[0]
            
            # Convert features back to node types (best effort)
            generated_nodes = []
            for i in range(num_nodes):
                # Find the node type with highest probability
                node_type_idx = torch.argmax(node_features[i]).item()
                generated_nodes.append({
                    'node_type': node_type_idx,
                    'connections': [],  # No connection info available
                    'features': node_features[i].tolist()
                })
        
        # If no nodes generated, create minimal structure
        if not generated_nodes:
            return json.dumps({
                "type": "def",
                "children": [
                    method_name,
                    {"type": "args", "children": []},
                    {"type": "nil", "children": []}
                ]
            })
        
        # Step 1: Initialize node objects
        node_objects = []
        for i, node_data in enumerate(generated_nodes):
            node_type_idx = node_data['node_type']
            
            # Convert node type index to name
            if 0 <= node_type_idx < len(encoder.node_types):
                node_type_name = encoder.node_types[node_type_idx]
            else:
                node_type_name = "unknown"
            
            # Create node object
            node_obj = {
                "type": node_type_name,
                "children": [],
                "_connections": node_data.get('connections', []),  # Temporary field for building
                "_index": i  # Temporary field for building
            }
            node_objects.append(node_obj)
        
        # Step 2: Build tree structure by connecting nodes to their parents
        # The first node is always the root
        if not node_objects:
            return json.dumps({"type": "nil", "children": []})
        
        root = node_objects[0]
        
        # For each subsequent node, find its parent and add it to parent's children
        for i in range(1, len(node_objects)):
            current_node = node_objects[i]
            connections = current_node["_connections"]
            
            # Find the parent (typically the most recent connection, or previous node as fallback)
            parent_idx = None
            if connections:
                # Use the last connection as the parent (most recent in sequence)
                parent_idx = connections[-1] if connections[-1] < i else None
            
            # Fallback: if no valid parent connection, attach to the previous node
            if parent_idx is None and i > 0:
                parent_idx = i - 1
            
            # Add current node to parent's children
            if parent_idx is not None and 0 <= parent_idx < len(node_objects):
                parent_node = node_objects[parent_idx]
                parent_node["children"].append(current_node)
        
        # Step 3: Clean up temporary fields and adjust structure for Ruby AST format
        def clean_node(node):
            # Remove temporary fields
            if "_connections" in node:
                del node["_connections"]
            if "_index" in node:
                del node["_index"]
            
            # Recursively clean children
            for child in node["children"]:
                if isinstance(child, dict):
                    clean_node(child)
            
            # Apply Ruby AST conventions based on node type
            node_type = node["type"]
            
            # Handle specific node types that need special structure
            if node_type == "def" and len(node["children"]) == 0:
                # Ensure def nodes have proper structure: name, args, body
                node["children"] = [
                    method_name,
                    {"type": "args", "children": []},
                    {"type": "nil", "children": []} if not node["children"] else node["children"][0]
                ]
            elif node_type == "args" and len(node["children"]) == 0:
                # Args node can be empty
                pass
            elif node_type == "send" and len(node["children"]) < 2:
                # Send nodes need at least receiver and method name
                while len(node["children"]) < 2:
                    node["children"].append("unknown")
            elif node_type in ["int", "str", "sym"] and len(node["children"]) == 0:
                # Literal nodes need a value
                if node_type == "int":
                    node["children"] = [42]
                elif node_type == "str":
                    node["children"] = ["generated"]
                elif node_type == "sym":
                    node["children"] = ["generated"]
            elif node_type in ["lvar", "ivar", "gvar", "cvar"] and len(node["children"]) == 0:
                # Variable nodes need a name
                node["children"] = ["var"]
            elif node_type == "const" and len(node["children"]) == 0:
                # Constant nodes need scope and name
                node["children"] = [None, "Generated"]
        
        # Clean the tree starting from root
        clean_node(root)
        
        # Step 4: Ensure the root is a proper method definition
        if root["type"] != "def":
            # Wrap in a method definition if the root isn't already one
            root = {
                "type": "def",
                "children": [
                    method_name,
                    {"type": "args", "children": []},
                    root
                ]
            }
        
        return json.dumps(root)
    
    def ruby_prettify(self, ast_json):
        """Call Ruby pretty printer to convert AST JSON to Ruby code."""
        try:
            # Create temporary file for AST JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(ast_json)
                temp_file = f.name
            
            # Set up Ruby environment
            ruby_env = dict(os.environ)
            if os.path.exists('.env-ruby'):
                # Source Ruby environment if available
                ruby_env.update({
                    'PATH': '/home/runner/.local/share/gem/ruby/3.2.0/bin:' + os.environ.get('PATH', ''),
                    'GEM_PATH': '/home/runner/.local/share/gem/ruby/3.2.0:' + os.environ.get('GEM_PATH', '')
                })
            
            # Call Ruby pretty printer
            result = subprocess.run([
                'bundle', 'exec', 'ruby', 'scripts/pretty_print_ast.rb', temp_file
            ], capture_output=True, text=True, env=ruby_env, cwd=os.path.dirname(__file__))
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"Ruby pretty printer failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error in Ruby pretty printing: {e}")
            return None
    
    def generate_code(self, text_prompt, method_name=None):
        """
        Complete pipeline: text -> embedding -> AST -> Ruby code.
        
        Args:
            text_prompt: Natural language description
            method_name: Optional method name (auto-generated if None)
            
        Returns:
            Generated Ruby code as string
        """
        print(f"🔍 Generating code for: '{text_prompt}'")
        print("-" * 50)
        
        # Step 1: Text to embedding
        print("1. Converting text to embedding...")
        embedding = self.text_to_embedding(text_prompt)
        print(f"   ✅ Generated embedding shape: {embedding.shape}")
        
        # Step 2: Embedding to AST
        print("2. Converting embedding to AST...")
        reconstruction = self.embedding_to_ast(embedding)
        print(f"   ✅ Generated AST with {reconstruction['node_features'].shape[1]} nodes")
        
        # Step 3: AST to JSON
        print("3. Converting AST to JSON...")
        if method_name is None:
            # Generate method name from text prompt
            words = text_prompt.lower().replace(' ', '_').replace('-', '_')
            method_name = ''.join(c for c in words if c.isalnum() or c == '_')[:20]
            if not method_name or not method_name[0].isalpha():
                method_name = "generated_method"
        
        ast_json = self.ast_to_json(reconstruction, method_name)
        print("   ✅ Generated AST JSON")
        
        # Step 4: JSON to Ruby code
        print("4. Converting JSON to Ruby code...")
        ruby_code = self.ruby_prettify(ast_json)
        
        if ruby_code:
            print("   ✅ Generated Ruby code")
            return ruby_code
        else:
            print("   ❌ Failed to generate Ruby code")
            return None


class AutoregressiveCodeGenerator:
    """Enhanced code generator using autoregressive AST decoder."""
    
    def __init__(self, alignment_model_path="models/best_alignment_model.pt", 
                 autoregressive_decoder_path="best_autoregressive_decoder.pt",
                 code_encoder_path="models/best_model.pt"):
        """
        Initialize the autoregressive code generator.
        
        Args:
            alignment_model_path: Path to trained AlignmentModel
            autoregressive_decoder_path: Path to trained AutoregressiveASTDecoder  
            code_encoder_path: Path to trained code encoder weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print("🚀 Loading Autoregressive Code Generation Models...")
        print("=" * 50)
        
        # Load AlignmentModel
        print("📦 Loading AlignmentModel...")
        self.alignment_model = self._load_alignment_model(
            alignment_model_path, code_encoder_path
        )
        print("✅ AlignmentModel loaded successfully")
        
        # Load AutoregressiveASTDecoder
        print("📦 Loading AutoregressiveASTDecoder...")
        self.autoregressive_decoder = self._load_autoregressive_decoder(autoregressive_decoder_path)
        print("✅ AutoregressiveASTDecoder loaded successfully")
        
        print("\n🎯 Autoregressive Code Generator ready!")
        
    def _load_alignment_model(self, model_path, code_encoder_path):
        """Load the AlignmentModel with proper configuration."""
        # Create model with same configuration as training
        alignment_model = AlignmentModel(
            input_dim=74,  # Based on dataset
            hidden_dim=64,
            text_model_name='all-MiniLM-L6-v2',  # Use same model as training
            code_encoder_weights_path=code_encoder_path
        )
        
        # Load trained weights if available
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            alignment_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded alignment model weights from {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Could not find alignment model at {model_path}")
            print("   Using randomly initialized weights (for testing only)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load alignment model weights: {e}")
            print("   Using randomly initialized weights (for testing only)")
            
        alignment_model.eval()
        
        return alignment_model.to(self.device)
    
    def _load_autoregressive_decoder(self, model_path):
        """Load the AutoregressiveASTDecoder with proper configuration."""
        decoder = AutoregressiveASTDecoder(
            text_embedding_dim=64,
            graph_hidden_dim=64,
            state_hidden_dim=128,
            node_types=74,
            sequence_model='GRU'
        )
        
        # Load trained weights if available
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            decoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded autoregressive decoder weights from {model_path}")
        except FileNotFoundError:
            print(f"⚠️  Warning: Could not find autoregressive decoder at {model_path}")
            print("   Using randomly initialized weights (for testing only)")
        except Exception as e:
            print(f"⚠️  Warning: Could not load autoregressive decoder weights: {e}")
            print("   Using randomly initialized weights (for testing only)")
            
        decoder.eval()
        
        return decoder.to(self.device)
    
    def text_to_embedding(self, text):
        """Convert natural language text to 64-dimensional embedding."""
        with torch.no_grad():
            embedding = self.alignment_model.encode_text([text])
        return embedding
    
    def embedding_to_ast(self, embedding, max_length=50, temperature=1.0, top_k=5):
        """Convert embedding to AST structure using autoregressive generation."""
        return generate_ast_autoregressive(
            self.autoregressive_decoder,
            embedding,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
    
    def ast_to_json(self, reconstruction, method_name="generated_method"):
        """
        Convert decoder output to AST JSON format expected by Ruby pretty printer.
        
        This implementation uses the proper tree-building algorithm that takes the sequence
        of generated nodes and their predicted parent-child connections and assembles them
        into the correct nested JSON structure.
        """
        # Import the node type mapping
        import sys
        import os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from data_processing import ASTNodeEncoder
        
        # Initialize the encoder to get node type names
        encoder = ASTNodeEncoder()
        
        # Check if we have the modern reconstruction format with generated nodes
        if 'generated_nodes' in reconstruction:
            generated_nodes = reconstruction['generated_nodes']
        else:
            # Fallback: Try to extract from node_features (though this won't have connection info)
            # This provides backward compatibility with older reconstruction formats
            node_features = reconstruction['node_features'][0]  # First batch item
            num_nodes = node_features.shape[0]
            
            # Convert features back to node types (best effort)
            generated_nodes = []
            for i in range(num_nodes):
                # Find the node type with highest probability
                node_type_idx = torch.argmax(node_features[i]).item()
                generated_nodes.append({
                    'node_type': node_type_idx,
                    'connections': [],  # No connection info available
                    'features': node_features[i].tolist()
                })
        
        # If no nodes generated, create minimal structure
        if not generated_nodes:
            return json.dumps({
                "type": "def",
                "children": [
                    method_name,
                    {"type": "args", "children": []},
                    {"type": "nil", "children": []}
                ]
            })
        
        # Step 1: Initialize node objects
        node_objects = []
        for i, node_data in enumerate(generated_nodes):
            node_type_idx = node_data['node_type']
            
            # Convert node type index to name
            if 0 <= node_type_idx < len(encoder.node_types):
                node_type_name = encoder.node_types[node_type_idx]
            else:
                node_type_name = "unknown"
            
            # Create node object
            node_obj = {
                "type": node_type_name,
                "children": [],
                "_connections": node_data.get('connections', []),  # Temporary field for building
                "_index": i  # Temporary field for building
            }
            node_objects.append(node_obj)
        
        # Step 2: Build tree structure by connecting nodes to their parents
        # The first node is always the root
        if not node_objects:
            return json.dumps({"type": "nil", "children": []})
        
        root = node_objects[0]
        
        # For each subsequent node, find its parent and add it to parent's children
        for i in range(1, len(node_objects)):
            current_node = node_objects[i]
            connections = current_node["_connections"]
            
            # Find the parent (typically the most recent connection, or previous node as fallback)
            parent_idx = None
            if connections:
                # Use the last connection as the parent (most recent in sequence)
                parent_idx = connections[-1] if connections[-1] < i else None
            
            # Fallback: if no valid parent connection, attach to the previous node
            if parent_idx is None and i > 0:
                parent_idx = i - 1
            
            # Add current node to parent's children
            if parent_idx is not None and 0 <= parent_idx < len(node_objects):
                parent_node = node_objects[parent_idx]
                parent_node["children"].append(current_node)
        
        # Step 3: Clean up temporary fields and adjust structure for Ruby AST format
        def clean_node(node):
            # Remove temporary fields
            if "_connections" in node:
                del node["_connections"]
            if "_index" in node:
                del node["_index"]
            
            # Recursively clean children
            for child in node["children"]:
                if isinstance(child, dict):
                    clean_node(child)
            
            # Apply Ruby AST conventions based on node type
            node_type = node["type"]
            
            # Handle specific node types that need special structure
            if node_type == "def" and len(node["children"]) == 0:
                # Ensure def nodes have proper structure: name, args, body
                node["children"] = [
                    method_name,
                    {"type": "args", "children": []},
                    {"type": "nil", "children": []} if not node["children"] else node["children"][0]
                ]
            elif node_type == "args" and len(node["children"]) == 0:
                # Args node can be empty
                pass
            elif node_type == "send" and len(node["children"]) < 2:
                # Send nodes need at least receiver and method name
                while len(node["children"]) < 2:
                    node["children"].append("unknown")
            elif node_type in ["int", "str", "sym"] and len(node["children"]) == 0:
                # Literal nodes need a value
                if node_type == "int":
                    node["children"] = [42]
                elif node_type == "str":
                    node["children"] = ["generated"]
                elif node_type == "sym":
                    node["children"] = ["generated"]
            elif node_type in ["lvar", "ivar", "gvar", "cvar"] and len(node["children"]) == 0:
                # Variable nodes need a name
                node["children"] = ["var"]
            elif node_type == "const" and len(node["children"]) == 0:
                # Constant nodes need scope and name
                node["children"] = [None, "Generated"]
        
        # Clean the tree starting from root
        clean_node(root)
        
        # Step 4: Ensure the root is a proper method definition
        if root["type"] != "def":
            # Wrap in a method definition if the root isn't already one
            root = {
                "type": "def",
                "children": [
                    method_name,
                    {"type": "args", "children": []},
                    root
                ]
            }
        
        return json.dumps(root)
    
    def ruby_prettify(self, ast_json):
        """Call Ruby pretty printer to convert AST JSON to Ruby code."""
        try:
            # Create temporary file for AST JSON
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                f.write(ast_json)
                temp_file = f.name
            
            # Set up Ruby environment
            ruby_env = dict(os.environ)
            if os.path.exists('.env-ruby'):
                # Source Ruby environment if available
                ruby_env.update({
                    'PATH': '/home/runner/.local/share/gem/ruby/3.2.0/bin:' + os.environ.get('PATH', ''),
                    'GEM_PATH': '/home/runner/.local/share/gem/ruby/3.2.0:' + os.environ.get('GEM_PATH', '')
                })
            
            # Call Ruby pretty printer
            result = subprocess.run([
                'bundle', 'exec', 'ruby', 'scripts/pretty_print_ast.rb', temp_file
            ], capture_output=True, text=True, env=ruby_env, cwd=os.path.dirname(__file__))
            
            # Clean up temp file
            os.unlink(temp_file)
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                raise Exception(f"Ruby pretty printer failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Error in Ruby pretty printing: {e}")
            return None
    
    def generate_code(self, text_prompt, method_name=None, **kwargs):
        """
        Complete pipeline: text -> embedding -> AST -> Ruby code.
        
        Args:
            text_prompt: Natural language description
            method_name: Optional method name (auto-generated if None)
            **kwargs: Generation control parameters
                - max_length: Maximum number of nodes to generate (default: 50)
                - temperature: Sampling temperature for diversity (default: 1.0)
                - top_k: Top-k sampling for node type selection (default: 5)
            
        Returns:
            Generated Ruby code as string
        """
        print(f"🔍 Generating code for: '{text_prompt}'")
        print("-" * 50)
        
        # Extract generation parameters
        max_length = kwargs.get('max_length', 50)
        temperature = kwargs.get('temperature', 1.0)
        top_k = kwargs.get('top_k', 5)
        
        print(f"   🎛️  Generation settings: max_length={max_length}, temperature={temperature}, top_k={top_k}")
        
        # Step 1: Text to embedding
        print("1. Converting text to embedding...")
        embedding = self.text_to_embedding(text_prompt)
        print(f"   ✅ Generated embedding shape: {embedding.shape}")
        
        # Step 2: Embedding to AST (NEW: autoregressive generation)
        print("2. Converting embedding to AST (autoregressive)...")
        reconstruction = self.embedding_to_ast(
            embedding,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k
        )
        print(f"   ✅ Generated AST with {reconstruction['num_nodes']} nodes")
        
        # Step 3: AST to JSON
        print("3. Converting AST to JSON...")
        if method_name is None:
            # Generate method name from text prompt
            words = text_prompt.lower().replace(' ', '_').replace('-', '_')
            method_name = ''.join(c for c in words if c.isalnum() or c == '_')[:20]
            if not method_name or not method_name[0].isalpha():
                method_name = "generated_method"
        
        ast_json = self.ast_to_json(reconstruction, method_name)
        print("   ✅ Generated AST JSON")
        
        # Step 4: JSON to Ruby code
        print("4. Converting JSON to Ruby code...")
        ruby_code = self.ruby_prettify(ast_json)
        
        if ruby_code:
            print("   ✅ Generated Ruby code")
            return ruby_code
        else:
            print("   ❌ Failed to generate Ruby code")
            return None


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate Ruby code from natural language descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enhanced autoregressive generation (default, recommended)
  python generate_code.py "calculate total price with tax"
  python generate_code.py "find user by email address" --method-name find_user
  python generate_code.py "validate user input" --temperature 0.5 --top-k 3
  python generate_code.py "process data creatively" --temperature 1.2 --top-k 10
  python generate_code.py "complex validation method" --max-length 100
  
  # Standard generation (not recommended due to known issues)
  python generate_code.py "calculate total price" --use-standard
  
  # Interactive mode
  python generate_code.py --interactive
  python generate_code.py --interactive --temperature 0.8
        """
    )
    
    parser.add_argument(
        'prompt',
        nargs='?',
        help='Natural language description of the method to generate'
    )
    
    parser.add_argument(
        '--method-name',
        help='Override the generated method name'
    )
    
    parser.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Start interactive mode'
    )
    
    parser.add_argument(
        '--alignment-model',
        default='models/best_alignment_model.pt',
        help='Path to trained AlignmentModel (default: models/best_alignment_model.pt)'
    )
    
    parser.add_argument(
        '--decoder-model',
        default='models/best_decoder.pt',
        help='Path to trained ASTDecoder (default: models/best_decoder.pt)'
    )
    
    parser.add_argument(
        '--autoregressive-decoder',
        default='best_autoregressive_decoder.pt',
        help='Path to trained AutoregressiveASTDecoder (default: best_autoregressive_decoder.pt)'
    )
    
    parser.add_argument(
        '--code-encoder',
        default='models/best_model.pt',
        help='Path to trained code encoder (default: models/best_model.pt)'
    )
    
    parser.add_argument(
        '--use-standard',
        action='store_true',
        help='Use standard one-shot decoder (not recommended due to known issues)'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=50,
        help='Maximum number of nodes to generate (default: 50)'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature for diversity (default: 1.0)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Top-k sampling for node type selection (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.prompt:
        parser.error("Either provide a prompt or use --interactive mode")
    
    try:
        # Initialize code generator - now using autoregressive by default
        if args.use_standard:
            print("⚠️  Using Standard Code Generator (known issues with repetitive output)")
            generator = CodeGenerator(
                alignment_model_path=args.alignment_model,
                decoder_model_path=args.decoder_model,
                code_encoder_path=args.code_encoder
            )
            generation_kwargs = {}  # Standard generator doesn't support extra parameters
        else:
            print("🚀 Using Enhanced Autoregressive Code Generator (Recommended)")
            generator = AutoregressiveCodeGenerator(
                alignment_model_path=args.alignment_model,
                autoregressive_decoder_path=args.autoregressive_decoder,
                code_encoder_path=args.code_encoder
            )
            # Prepare generation kwargs for autoregressive generator
            generation_kwargs = {
                'max_length': args.max_length,
                'temperature': args.temperature,
                'top_k': args.top_k
            }
        
        if args.interactive:
            # Interactive mode
            print(f"\n🎨 Interactive Code Generation")
            if not args.use_standard:
                print("🔥 Enhanced with autoregressive generation controls!")
                print(f"   Settings: max_length={args.max_length}, temperature={args.temperature}, top_k={args.top_k}")
            print("Type 'quit' or 'exit' to stop")
            print("=" * 50)
            
            while True:
                try:
                    prompt = input("\n🤖 Describe the method you want to generate: ").strip()
                    
                    if prompt.lower() in ['quit', 'exit', 'q']:
                        print("👋 Goodbye!")
                        break
                    
                    if not prompt:
                        continue
                    
                    # Generate code
                    ruby_code = generator.generate_code(prompt, args.method_name, **generation_kwargs)
                    
                    if ruby_code:
                        print(f"\n🎉 Generated Ruby Code:")
                        print("=" * 30)
                        print(ruby_code)
                        print("=" * 30)
                    else:
                        print("❌ Failed to generate code. Please try a different prompt.")
                        
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        else:
            # Single prompt mode
            ruby_code = generator.generate_code(args.prompt, args.method_name, **generation_kwargs)
            
            if ruby_code:
                print(f"\n🎉 Generated Ruby Code:")
                print("=" * 30)
                print(ruby_code)
                print("=" * 30)
            else:
                print("❌ Failed to generate code")
                sys.exit(1)
    
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()