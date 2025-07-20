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

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models import AlignmentModel, ASTDecoder


class CodeGenerator:
    """Main code generation class."""
    
    def __init__(self, alignment_model_path="best_alignment_model.pt", 
                 decoder_model_path="best_decoder.pt",
                 code_encoder_path="best_model.pt"):
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
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        alignment_model.load_state_dict(checkpoint['model_state_dict'])
        alignment_model.eval()
        
        return alignment_model.to(self.device)
    
    def _load_ast_decoder(self, model_path):
        """Load the ASTDecoder with proper configuration."""
        decoder = ASTDecoder(
            embedding_dim=64,
            output_node_dim=74,
            hidden_dim=64
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
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
        
        This is a simplified implementation that creates basic Ruby method structures.
        In a full implementation, this would need to properly interpret the node features
        to generate meaningful AST structures.
        """
        node_features = reconstruction['node_features'][0]  # First batch item
        edge_index = reconstruction['edge_index']
        num_nodes = node_features.shape[0]
        
        # For simplicity, create a basic method structure
        # In practice, you'd analyze node_features to determine types and content
        
        # Generate simple method arguments based on number of nodes
        args = []
        if num_nodes > 5:
            args = ["arg1", "arg2"]
        elif num_nodes > 3:
            args = ["arg"]
        
        # Create basic method body - this is very simplified
        body_statements = []
        
        # Add some basic statements based on method name hints
        if any(word in method_name.lower() for word in ['calculate', 'compute', 'total']):
            if args:
                body_statements.append({
                    "type": "send",
                    "children": [
                        {"type": "lvar", "children": [args[0]]},
                        "+",
                        {"type": "lvar", "children": [args[1]]} if len(args) > 1 else {"type": "int", "children": [1]}
                    ]
                })
            else:
                body_statements.append({
                    "type": "int",
                    "children": [42]
                })
        elif any(word in method_name.lower() for word in ['get', 'fetch', 'find']):
            body_statements.append({
                "type": "send",
                "children": [
                    {"type": "const", "children": [None, "SomeClass"]},
                    "find",
                    {"type": "lvar", "children": [args[0]]} if args else {"type": "int", "children": [1]}
                ]
            })
        else:
            # Default simple return
            body_statements.append({
                "type": "str",
                "children": ["result"]
            })
        
        # Construct method AST
        method_ast = {
            "type": "def",
            "children": [
                method_name,
                {
                    "type": "args",
                    "children": [{"type": "arg", "children": [arg]} for arg in args]
                } if args else {"type": "args", "children": []},
                {
                    "type": "begin",
                    "children": body_statements
                } if len(body_statements) > 1 else body_statements[0]
            ]
        }
        
        return json.dumps(method_ast)
    
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


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Generate Ruby code from natural language descriptions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_code.py "calculate total price with tax"
  python generate_code.py "find user by email address" --method-name find_user
  python generate_code.py --interactive
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
        default='best_alignment_model.pt',
        help='Path to trained AlignmentModel (default: best_alignment_model.pt)'
    )
    
    parser.add_argument(
        '--decoder-model',
        default='best_decoder.pt',
        help='Path to trained ASTDecoder (default: best_decoder.pt)'
    )
    
    parser.add_argument(
        '--code-encoder',
        default='best_model.pt',
        help='Path to trained code encoder (default: best_model.pt)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.interactive and not args.prompt:
        parser.error("Either provide a prompt or use --interactive mode")
    
    try:
        # Initialize code generator
        generator = CodeGenerator(
            alignment_model_path=args.alignment_model,
            decoder_model_path=args.decoder_model,
            code_encoder_path=args.code_encoder
        )
        
        if args.interactive:
            # Interactive mode
            print("\n🎨 Interactive Code Generation")
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
                    ruby_code = generator.generate_code(prompt, args.method_name)
                    
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
            ruby_code = generator.generate_code(args.prompt, args.method_name)
            
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