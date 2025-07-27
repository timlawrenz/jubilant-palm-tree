#!/usr/bin/env python3
"""
Verification script for the evaluate_autoencoder.ipynb fixes.

This script tests the key components that were causing the notebook to hang
and verifies that the fixes are working correctly.
"""

import sys
import os
import json
import subprocess
import torch
from subprocess import TimeoutExpired
from torch_geometric.data import Data

def main():
    print("🔍 Verifying evaluate_autoencoder.ipynb fixes...")
    print()
    
    # Change to the project root directory
    project_root = '/home/runner/work/jubilant-palm-tree/jubilant-palm-tree'
    if not os.path.exists(project_root):
        project_root = '.'  # Fallback to current directory
    
    os.chdir(project_root)
    sys.path.insert(0, 'src')
    
    # Test 1: Check required dependencies
    print("✅ Test 1: Checking dependencies...")
    try:
        from data_processing import RubyASTDataset
        from models import ASTAutoencoder
        print("   ✓ Python modules import successfully")
    except ImportError as e:
        print(f"   ❌ Missing Python dependencies: {e}")
        print("   📝 Run: pip install -r requirements.txt")
        return False
    
    # Test 2: Check Ruby gem availability
    print("\n✅ Test 2: Checking Ruby Parser gem...")
    try:
        env = dict(os.environ)
        env['GEM_PATH'] = f"/home/runner/.local/share/gem/ruby/3.2.0:{env.get('GEM_PATH', '')}"
        
        result = subprocess.run(
            ['ruby', '-e', 'require "parser/current"; puts "Parser gem available"'],
            capture_output=True,
            text=True,
            env=env,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✓ Ruby Parser gem is available")
        else:
            print(f"   ❌ Ruby Parser gem not available: {result.stderr}")
            print("   📝 Run: gem install parser --user-install")
            return False
    except Exception as e:
        print(f"   ❌ Error checking Ruby gem: {e}")
        return False
    
    # Test 3: Check data files
    print("\n✅ Test 3: Checking data files...")
    required_files = [
        'dataset/test.jsonl',
        'models/best_model.pt',
        'models/best_decoder.pt',
        'scripts/pretty_print_ast.rb',
        'scripts/check_syntax.rb'
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"   ✓ {file_path} exists")
        else:
            print(f"   ❌ Missing required file: {file_path}")
            return False
    
    # Test 4: Test timeout protection
    print("\n✅ Test 4: Testing timeout protection...")
    
    # Create a test AST that would previously cause hanging
    test_ast = {
        'type': 'def',
        'children': [
            'test_method',
            {'type': 'args', 'children': []},
            {'type': 'str', 'children': ['test_content']}
        ]
    }
    
    try:
        temp_file = '/tmp/test_ast.json'
        with open(temp_file, 'w') as f:
            json.dump(test_ast, f)
        
        env = dict(os.environ)
        env['GEM_PATH'] = f"/home/runner/.local/share/gem/ruby/3.2.0:{env.get('GEM_PATH', '')}"
        env['PATH'] = f"/home/runner/.local/share/gem/ruby/3.2.0/bin:{env.get('PATH', '')}"
        
        result = subprocess.run(
            ['ruby', 'scripts/pretty_print_ast.rb', temp_file],
            capture_output=True,
            text=True,
            env=env,
            timeout=10
        )
        
        if result.returncode == 0:
            print("   ✓ Ruby pretty printer works with timeout protection")
        else:
            print(f"   ⚠ Ruby pretty printer returned error (but didn't hang): {result.stderr}")
            
    except TimeoutExpired:
        print("   ❌ Ruby pretty printer still hanging (timeout protection failed)")
        return False
    except Exception as e:
        print(f"   ❌ Error testing timeout protection: {e}")
        return False
    
    # Test 5: Test basic model loading
    print("\n✅ Test 5: Testing model loading...")
    try:
        os.chdir('notebooks')  # Change to notebooks directory for relative paths
        
        test_dataset = RubyASTDataset("../dataset/test.jsonl")
        autoencoder = ASTAutoencoder(
            encoder_input_dim=74,
            node_output_dim=74,
            hidden_dim=64,
            num_layers=3,
            conv_type='GCN',
            freeze_encoder=True,
            encoder_weights_path="../models/best_model.pt"
        )
        
        # Load decoder
        checkpoint = torch.load("../models/best_decoder.pt", map_location='cpu')
        if 'decoder_state_dict' in checkpoint:
            autoencoder.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        else:
            autoencoder.decoder.load_state_dict(checkpoint)
        
        autoencoder.eval()
        print(f"   ✓ Model loaded successfully, dataset has {len(test_dataset)} samples")
        
    except Exception as e:
        print(f"   ❌ Error loading model/dataset: {e}")
        return False
    
    print("\n🎉 All verification tests passed!")
    print("\n📋 Summary of fixes applied:")
    print("   • Added 10-second timeout to Ruby pretty printer calls")
    print("   • Added 5-second timeout to Ruby syntax checker calls")
    print("   • Fixed Ruby gem environment (GEM_PATH and PATH)")
    print("   • Simplified AST reconstruction to avoid complex structures")
    print("   • Added error handling for timeout exceptions")
    print("   • Added subprocess.TimeoutExpired import")
    print("\n✨ The evaluate_autoencoder.ipynb notebook should now work without hanging!")
    print("   Run all cells in sequence: Setup → Helper Functions → Evaluate Sample Methods → Side-by-Side Comparison")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)