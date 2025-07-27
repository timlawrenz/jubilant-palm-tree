#!/usr/bin/env python3
"""
Test suite for AlignmentModel - Dual encoder for text-code alignment.
"""

import sys
import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from models import AlignmentModel
from data_processing import create_data_loaders

# Helper function to get dataset paths relative to this script
def get_dataset_path(relative_path):
    """Get dataset path relative to this script location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, relative_path)

def create_sample_graph():
    """Create a sample graph for testing."""
    # Simple graph with 5 nodes
    x = torch.randn(5, 4)  # 5 nodes with 4 features each
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4],
                              [1, 0, 2, 1, 3, 2, 4, 3]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index)

def test_alignment_model_initialization():
    """Test AlignmentModel initialization."""
    print("🔍 Testing AlignmentModel Initialization")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        print(f"✅ Model initialized successfully")
        print(f"   Code encoder frozen: {not any(p.requires_grad for p in model.code_encoder.parameters())}")
        print(f"   Text projection trainable: {any(p.requires_grad for p in model.text_projection.parameters())}")
        print(f"   Model info:\n{model.get_model_info()}")
        return True
    except Exception as e:
        print(f"❌ Model initialization failed: {e}")
        return False

def test_single_sample_encoding():
    """Test encoding single graph and text."""
    print("\n🔍 Testing Single Sample Encoding")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        
        # Create sample data
        graph = create_sample_graph()
        text = ["calculate total price"]
        
        # Test individual encoders
        code_embedding = model.encode_code(graph)
        text_embedding = model.encode_text(text)
        
        print(f"✅ Code embedding shape: {code_embedding.shape}")
        print(f"✅ Text embedding shape: {text_embedding.shape}")
        print(f"   Code embedding sample: {code_embedding[0][:5].tolist()}")
        print(f"   Text embedding sample: {text_embedding[0][:5].tolist()}")
        
        # Check dimensions match
        assert code_embedding.shape[1] == text_embedding.shape[1] == 64
        print(f"✅ Embedding dimensions match: {code_embedding.shape[1]}D")
        
        return True
    except Exception as e:
        print(f"❌ Single sample encoding failed: {e}")
        return False

def test_forward_pass():
    """Test complete forward pass with graph and text."""
    print("\n🔍 Testing Forward Pass")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        
        # Create sample data
        graph = create_sample_graph()
        texts = ["calculate total price"]
        
        # Forward pass
        outputs = model.forward(graph, texts)
        
        print(f"✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   Code embeddings shape: {outputs['code_embeddings'].shape}")
        print(f"   Text embeddings shape: {outputs['text_embeddings'].shape}")
        
        # Check outputs
        assert 'code_embeddings' in outputs
        assert 'text_embeddings' in outputs
        assert outputs['code_embeddings'].shape == outputs['text_embeddings'].shape
        print(f"✅ Output shapes match: {outputs['code_embeddings'].shape}")
        
        return True
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

def test_batch_processing():
    """Test batch processing with multiple graphs and texts."""
    print("\n🔍 Testing Batch Processing")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        
        # Create batch of graphs
        graphs = [create_sample_graph() for _ in range(3)]
        batch = Batch.from_data_list(graphs)
        
        # Create batch of texts
        texts = [
            "calculate total price",
            "get user information", 
            "process payment method"
        ]
        
        # Forward pass with batch
        outputs = model.forward(batch, texts)
        
        print(f"✅ Batch processing successful")
        print(f"   Batch size: {len(texts)}")
        print(f"   Code embeddings shape: {outputs['code_embeddings'].shape}")
        print(f"   Text embeddings shape: {outputs['text_embeddings'].shape}")
        
        # Check batch dimensions
        assert outputs['code_embeddings'].shape[0] == len(texts)
        assert outputs['text_embeddings'].shape[0] == len(texts)
        assert outputs['code_embeddings'].shape[1] == 64
        assert outputs['text_embeddings'].shape[1] == 64
        print(f"✅ Batch dimensions correct: {len(texts)} samples, 64D embeddings")
        
        return True
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
        return False

def test_embedding_similarity():
    """Test that embeddings can be compared using cosine similarity."""
    print("\n🔍 Testing Embedding Similarity")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        
        # Create sample data
        graph = create_sample_graph()
        texts = [
            "calculate total price",
            "process user data",
            "calculate total price"  # Duplicate for similarity test
        ]
        
        # Get embeddings
        text_embeddings = model.encode_text(texts)
        
        # Calculate cosine similarities
        text_norm = F.normalize(text_embeddings, p=2, dim=1)
        similarities = torch.mm(text_norm, text_norm.t())
        
        print(f"✅ Similarity computation successful")
        print(f"   Text 0 vs Text 0: {similarities[0, 0].item():.4f}")
        print(f"   Text 0 vs Text 1: {similarities[0, 1].item():.4f}")
        print(f"   Text 0 vs Text 2: {similarities[0, 2].item():.4f}")
        
        # Check that identical texts have high similarity
        assert similarities[0, 2].item() > 0.9, "Identical texts should have high similarity"
        print(f"✅ Identical texts have high similarity: {similarities[0, 2].item():.4f}")
        
        return True
    except Exception as e:
        print(f"❌ Embedding similarity test failed: {e}")
        return False

def test_device_compatibility():
    """Test that model works with different devices."""
    print("\n🔍 Testing Device Compatibility")
    print("----------------------------------------")
    
    try:
        model = AlignmentModel(input_dim=4, hidden_dim=64)
        
        # Create sample data
        graph = create_sample_graph()
        texts = ["calculate total price"]
        
        # Test CPU
        outputs_cpu = model.forward(graph, texts)
        print(f"✅ CPU processing successful")
        print(f"   Code embeddings device: {outputs_cpu['code_embeddings'].device}")
        print(f"   Text embeddings device: {outputs_cpu['text_embeddings'].device}")
        
        # Test GPU if available
        if torch.cuda.is_available():
            model_gpu = model.cuda()
            graph_gpu = graph.cuda()
            outputs_gpu = model_gpu.forward(graph_gpu, texts)
            print(f"✅ GPU processing successful")
            print(f"   Code embeddings device: {outputs_gpu['code_embeddings'].device}")
            print(f"   Text embeddings device: {outputs_gpu['text_embeddings'].device}")
        else:
            print("⚠️  GPU not available, skipping GPU test")
        
        return True
    except Exception as e:
        print(f"❌ Device compatibility test failed: {e}")
        return False

def test_with_real_data():
    """Test with real data from the dataset if available."""
    print("\n🔍 Testing with Real Data")
    print("----------------------------------------")
    
    try:
        # Try to load real data
        try:
            train_loader, _ = create_data_loaders(
                train_path=get_dataset_path("../dataset/samples/train_sample.jsonl"),
                val_path=get_dataset_path("../dataset/samples/validation_sample.jsonl"),
                batch_size=2,
                shuffle=False
            )
            
            # Get a sample batch
            sample_batch = next(iter(train_loader))
            
            # Handle data format - the data loader returns a dictionary with lists
            if isinstance(sample_batch, dict):
                # Convert lists to tensors and create Data object
                x = torch.tensor(sample_batch['x'], dtype=torch.float32)
                edge_index = torch.tensor(sample_batch['edge_index'], dtype=torch.long)
                batch = torch.tensor(sample_batch['batch'], dtype=torch.long)
                
                # Create Data object
                from torch_geometric.data import Data
                real_graph = Data(x=x, edge_index=edge_index, batch=batch)
            else:
                real_graph = sample_batch
            
            print(f"✅ Loaded real data")
            print(f"   Graph nodes: {real_graph.x.shape[0]}")
            print(f"   Graph features: {real_graph.x.shape[1]}")
            print(f"   Batch size: {real_graph.batch.max().item() + 1}")
            
            # Create model with correct input dimension
            model = AlignmentModel(input_dim=real_graph.x.shape[1], hidden_dim=64)
            
            # Create sample texts for the batch
            batch_size = real_graph.batch.max().item() + 1
            texts = [f"sample method {i}" for i in range(batch_size)]
            
            # Forward pass
            outputs = model.forward(real_graph, texts)
            
            print(f"✅ Real data processing successful")
            print(f"   Code embeddings shape: {outputs['code_embeddings'].shape}")
            print(f"   Text embeddings shape: {outputs['text_embeddings'].shape}")
            
        except FileNotFoundError:
            print("⚠️  Real dataset not found, using synthetic data")
            # Fallback to synthetic test
            return test_batch_processing()
        
        return True
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 AlignmentModel Testing Suite")
    print("==================================================")
    
    tests = [
        test_alignment_model_initialization,
        test_single_sample_encoding,
        test_forward_pass,
        test_batch_processing,
        test_embedding_similarity,
        test_device_compatibility,
        test_with_real_data
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n==================================================")
    print(f"🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AlignmentModel is ready for use.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    main()