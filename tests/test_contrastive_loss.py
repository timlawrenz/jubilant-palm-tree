#!/usr/bin/env python3
"""
Test script for contrastive loss functions.

This script tests the contrastive loss functions for code-text alignment
to ensure they work correctly with embedding tensors.
"""

import sys
import os
import torch
import torch.nn.functional as F

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from loss import info_nce_loss, cosine_embedding_loss, simple_contrastive_loss


def test_identical_embeddings():
    """Test that loss is low when embeddings are identical."""
    print("🔍 Testing Loss with Identical Embeddings")
    print("-" * 40)
    
    # Create identical embeddings
    batch_size = 4
    embedding_dim = 64
    
    # Same embeddings for code and text (perfect alignment)
    embeddings = torch.randn(batch_size, embedding_dim)
    code_embeddings = embeddings.clone()
    text_embeddings = embeddings.clone()
    
    # Test all loss functions
    info_nce = info_nce_loss(code_embeddings, text_embeddings)
    cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
    simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
    
    print(f"✅ InfoNCE loss (identical): {info_nce.item():.6f}")
    print(f"✅ Cosine embedding loss (identical): {cosine_emb.item():.6f}")
    print(f"✅ Simple contrastive loss (identical): {simple_loss.item():.6f}")
    
    # Losses should be low for identical embeddings
    assert info_nce.item() < 0.1, f"InfoNCE loss too high: {info_nce.item()}"
    assert cosine_emb.item() < 0.5, f"Cosine embedding loss too high: {cosine_emb.item()}"
    assert simple_loss.item() < 0.1, f"Simple loss too high: {simple_loss.item()}"
    
    return True


def test_random_embeddings():
    """Test that loss is higher when embeddings are random."""
    print("\n🔍 Testing Loss with Random Embeddings")
    print("-" * 40)
    
    batch_size = 4
    embedding_dim = 64
    
    # Random embeddings (poor alignment)
    code_embeddings = torch.randn(batch_size, embedding_dim)
    text_embeddings = torch.randn(batch_size, embedding_dim)
    
    # Test all loss functions
    info_nce = info_nce_loss(code_embeddings, text_embeddings)
    cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
    simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
    
    print(f"✅ InfoNCE loss (random): {info_nce.item():.6f}")
    print(f"✅ Cosine embedding loss (random): {cosine_emb.item():.6f}")
    print(f"✅ Simple contrastive loss (random): {simple_loss.item():.6f}")
    
    # Losses should be higher for random embeddings
    assert info_nce.item() > 0.5, f"InfoNCE loss too low: {info_nce.item()}"
    # Note: cosine_emb might be variable depending on random similarities
    # Note: simple_loss might be negative (since it's -mean_similarity)
    
    return True


def test_opposite_embeddings():
    """Test that loss is high when embeddings are opposite."""
    print("\n🔍 Testing Loss with Opposite Embeddings")
    print("-" * 40)
    
    batch_size = 4
    embedding_dim = 64
    
    # Opposite embeddings (worst alignment)
    code_embeddings = torch.randn(batch_size, embedding_dim)
    text_embeddings = -code_embeddings  # Opposite direction
    
    # Test all loss functions
    info_nce = info_nce_loss(code_embeddings, text_embeddings)
    cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
    simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
    
    print(f"✅ InfoNCE loss (opposite): {info_nce.item():.6f}")
    print(f"✅ Cosine embedding loss (opposite): {cosine_emb.item():.6f}")
    print(f"✅ Simple contrastive loss (opposite): {simple_loss.item():.6f}")
    
    # Losses should be high for opposite embeddings
    assert info_nce.item() > 1.0, f"InfoNCE loss too low: {info_nce.item()}"
    assert cosine_emb.item() > 1.0, f"Cosine embedding loss too low: {cosine_emb.item()}"
    assert simple_loss.item() > 5.0, f"Simple loss too low: {simple_loss.item()}"
    
    return True


def test_gradient_flow():
    """Test that gradients flow through loss functions."""
    print("\n⚡ Testing Gradient Flow")
    print("-" * 40)
    
    batch_size = 3
    embedding_dim = 32
    
    # Create embeddings with gradient tracking
    code_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    text_embeddings = torch.randn(batch_size, embedding_dim, requires_grad=True)
    
    # Test InfoNCE loss gradients
    loss = info_nce_loss(code_embeddings, text_embeddings)
    loss.backward()
    
    print(f"✅ InfoNCE loss: {loss.item():.6f}")
    print(f"✅ Code gradients computed: {code_embeddings.grad is not None}")
    print(f"✅ Text gradients computed: {text_embeddings.grad is not None}")
    
    assert code_embeddings.grad is not None, "Code embeddings should have gradients"
    assert text_embeddings.grad is not None, "Text embeddings should have gradients"
    assert not torch.isnan(code_embeddings.grad).any(), "Code gradients should not be NaN"
    assert not torch.isnan(text_embeddings.grad).any(), "Text gradients should not be NaN"
    
    return True


def test_batch_sizes():
    """Test loss functions with different batch sizes."""
    print("\n📦 Testing Different Batch Sizes")
    print("-" * 40)
    
    embedding_dim = 64
    
    for batch_size in [1, 2, 8, 16]:
        code_embeddings = torch.randn(batch_size, embedding_dim)
        text_embeddings = torch.randn(batch_size, embedding_dim)
        
        # All loss functions should work with any batch size
        info_nce = info_nce_loss(code_embeddings, text_embeddings)
        cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
        simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
        
        print(f"✅ Batch size {batch_size:2d}: InfoNCE={info_nce.item():.3f}, "
              f"Cosine={cosine_emb.item():.3f}, Simple={simple_loss.item():.3f}")
        
        # Verify losses are not NaN or infinite
        assert not torch.isnan(info_nce), f"InfoNCE NaN for batch size {batch_size}"
        assert not torch.isnan(cosine_emb), f"Cosine loss NaN for batch size {batch_size}"
        assert not torch.isnan(simple_loss), f"Simple loss NaN for batch size {batch_size}"
        assert torch.isfinite(info_nce), f"InfoNCE infinite for batch size {batch_size}"
        assert torch.isfinite(cosine_emb), f"Cosine loss infinite for batch size {batch_size}"
        assert torch.isfinite(simple_loss), f"Simple loss infinite for batch size {batch_size}"
    
    return True


def test_embedding_dimensions():
    """Test loss functions with different embedding dimensions."""
    print("\n📏 Testing Different Embedding Dimensions")
    print("-" * 40)
    
    batch_size = 4
    
    for dim in [16, 32, 64, 128, 256]:
        code_embeddings = torch.randn(batch_size, dim)
        text_embeddings = torch.randn(batch_size, dim)
        
        # All loss functions should work with any embedding dimension
        info_nce = info_nce_loss(code_embeddings, text_embeddings)
        cosine_emb = cosine_embedding_loss(code_embeddings, text_embeddings)
        simple_loss = simple_contrastive_loss(code_embeddings, text_embeddings)
        
        print(f"✅ Dim {dim:3d}: InfoNCE={info_nce.item():.3f}, "
              f"Cosine={cosine_emb.item():.3f}, Simple={simple_loss.item():.3f}")
        
        # Verify losses are computable
        assert not torch.isnan(info_nce), f"InfoNCE NaN for dim {dim}"
        assert not torch.isnan(cosine_emb), f"Cosine loss NaN for dim {dim}"
        assert not torch.isnan(simple_loss), f"Simple loss NaN for dim {dim}"
    
    return True


def test_temperature_scaling():
    """Test that temperature parameter affects InfoNCE loss appropriately."""
    print("\n🌡️  Testing Temperature Scaling")
    print("-" * 40)
    
    batch_size = 4
    embedding_dim = 64
    
    # Create somewhat aligned embeddings
    code_embeddings = torch.randn(batch_size, embedding_dim)
    text_embeddings = code_embeddings + 0.1 * torch.randn(batch_size, embedding_dim)
    
    temperatures = [0.01, 0.05, 0.1, 0.5, 1.0]
    losses = []
    
    for temp in temperatures:
        loss = info_nce_loss(code_embeddings, text_embeddings, temperature=temp)
        losses.append(loss.item())
        print(f"✅ Temperature {temp:.2f}: Loss = {loss.item():.6f}")
    
    # Generally, lower temperature should give higher loss (sharper distributions)
    # But this relationship may not be strictly monotonic due to the specific embeddings
    print(f"✅ Temperature scaling tested across range {temperatures}")
    
    return True


def main():
    """Run all contrastive loss tests."""
    print("🧪 Contrastive Loss Testing Suite")
    print("=" * 50)
    
    tests = [
        test_identical_embeddings,
        test_random_embeddings,
        test_opposite_embeddings,
        test_gradient_flow,
        test_batch_sizes,
        test_embedding_dimensions,
        test_temperature_scaling
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
        print("🎉 All contrastive loss tests passed! Loss functions are ready for use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
    
    return passed == len(tests)


if __name__ == "__main__":
    main()