#!/usr/bin/env python3
"""
Test script to reproduce and validate the device mismatch issue fix.

This script specifically tests the device handling in ASTDecoder when 
running on CUDA vs CPU devices.
"""

import sys
import os
import torch
from torch_geometric.data import Data

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from models import ASTAutoencoder


def test_device_mismatch():
    """Test that reproduces the device mismatch issue."""
    print("🔧 Testing Device Mismatch Issue")
    print("-" * 40)
    
    # Test both CPU and CUDA if available
    devices_to_test = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda'))
        print("🖥️  CUDA available, testing both CPU and CUDA devices")
    else:
        print("⚠️  CUDA not available, testing CPU device handling only")
    
    all_passed = True
    
    for device in devices_to_test:
        print(f"\n📱 Testing on device: {device}")
        
        # Create sample data
        x = torch.randn(20, 74, dtype=torch.float)
        edge_index = torch.randint(0, 20, (2, 30), dtype=torch.long)
        batch = torch.zeros(20, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, batch=batch)
        
        print(f"   Input data device: {data.x.device}")
        
        # Create autoencoder and move to device
        autoencoder = ASTAutoencoder(
            encoder_input_dim=74,
            node_output_dim=74,
            hidden_dim=64,
            num_layers=3,
            conv_type='GCN',
            dropout=0.1,
            freeze_encoder=True
        ).to(device)
        
        print(f"   Model device: {next(autoencoder.parameters()).device}")
        
        # Move data to device  
        data = data.to(device)
        print(f"   Data moved to device: {data.x.device}")
        
        try:
            # This should trigger the device mismatch error before the fix
            with torch.no_grad():
                result = autoencoder(data)
            
            print(f"   ✅ Forward pass successful on {device}!")
            print(f"   ✅ Embedding shape: {result['embedding'].shape}")
            print(f"   ✅ Embedding device: {result['embedding'].device}")
            print(f"   ✅ Reconstruction node features shape: {result['reconstruction']['node_features'].shape}")
            print(f"   ✅ Reconstruction device: {result['reconstruction']['node_features'].device}")
            
        except RuntimeError as e:
            if "Expected all tensors to be on the same device" in str(e):
                print(f"   ❌ Device mismatch error occurred on {device}: {e}")
                all_passed = False
            else:
                print(f"   ❌ Unexpected error on {device}: {e}")
                raise e
    
    return all_passed


def test_decoder_device_consistency():
    """Test that the decoder creates all tensors on the correct device."""
    print("\n🔍 Testing Decoder Device Consistency")
    print("-" * 40)
    
    devices_to_test = [torch.device('cpu')]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device('cuda'))
    
    all_passed = True
    
    for device in devices_to_test:
        print(f"\n📱 Testing on device: {device}")
        
        from models import ASTDecoder
        
        decoder = ASTDecoder(
            embedding_dim=64,
            output_node_dim=74,
            hidden_dim=64,
            num_layers=2
        ).to(device)
        
        # Create embedding on device
        embedding = torch.randn(2, 64).to(device)
        print(f"   Input embedding device: {embedding.device}")
        
        with torch.no_grad():
            result = decoder(embedding, target_num_nodes=10)
        
        print(f"   ✅ Node features device: {result['node_features'].device}")
        print(f"   ✅ Edge index device: {result['edge_index'].device}")
        print(f"   ✅ Batch tensor device: {result['batch'].device}")
        
        # Verify all tensors are on the same device
        try:
            assert result['node_features'].device == device
            assert result['edge_index'].device == device
            assert result['batch'].device == device
            print(f"   ✅ All decoder outputs on correct device ({device})!")
        except AssertionError:
            print(f"   ❌ Device mismatch in decoder outputs on {device}")
            all_passed = False
    
    return all_passed


def main():
    """Run device mismatch tests."""
    print("🧪 Device Mismatch Test Suite")
    print("=" * 50)
    
    tests = [
        test_device_mismatch,
        test_decoder_device_consistency
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"❌ {test.__name__} failed")
        except Exception as e:
            print(f"❌ {test.__name__} failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"🏁 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All device tests passed!")
    else:
        print("⚠️  Some device tests failed.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()